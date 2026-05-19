"""
BSD DeepNight ROS2 노드 (두 카메라 지원)
=========================================
MORAI Camera-1(우측) + Camera-2(좌측) → SGLDet 추론 → BSD 경고 + 시각화

Subscribed topics:
  /morai/camera_right/image_raw (sensor_msgs/Image)  ← Camera-1 (우측)
  /morai/camera_left/image_raw  (sensor_msgs/Image)  ← Camera-2 (좌측)

Published topics:
  /bsd/detections      (vision_msgs/Detection2DArray)
  /bsd/warning         (std_msgs/String : "SAFE"/"WARNING"/"DANGER")
  /bsd/visualization_right (sensor_msgs/Image)
  /bsd/visualization_left  (sensor_msgs/Image)

실행 예시:
  ros2 run bsd_deepnight detector_node
  ros2 launch bsd_deepnight bsd_detector.launch.py weights:=$(pwd)/checkpoints/best_model.pt
"""

import os
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String, Header
from vision_msgs.msg import (
    Detection2DArray, Detection2D,
    BoundingBox2D, ObjectHypothesisWithPose,
)
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge

import cv2
import numpy as np

# 프로젝트 루트 path 추가 (BSDsystem/ 모듈 import 위해)
# ros2_ws/src/bsd_deepnight/bsd_deepnight/ → ../../../../ = BSDsystem/
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.calibration import CameraCalibration
from src.inference.detector        import SGLDetInference
from src.inference.bsd_interface   import BSDInterface
from models.sort_tracker           import SORTTracker


class BSDDetectorNode(Node):
    """SGLDet + SORT + BSD 통합 ROS2 노드 (좌·우 BSD 카메라 동시 처리)."""

    def __init__(self):
        super().__init__("bsd_detector_node")

        # ── 파라미터 선언 ────────────────────────────────────────────
        self.declare_parameter("weights",          "yolov8m.pt")
        self.declare_parameter("camera_config",    "configs/camera_config.yaml")
        # 두 카메라 토픽 (MORAI frameID 기반)
        self.declare_parameter("topic_right",  "/morai/camera_right/image_raw")
        self.declare_parameter("img_size",         640)
        self.declare_parameter("conf_threshold",   0.25)
        self.declare_parameter("iou_threshold",    0.45)
        self.declare_parameter("device",           "cuda")
        self.declare_parameter("publish_vis",      True)
        self.declare_parameter("undistort",        False)   # MORAI 시뮬은 왜곡 없음
        self.declare_parameter("project_root",     str(PROJECT_ROOT))

        weights       = self.get_parameter("weights").value
        camera_config = self.get_parameter("camera_config").value
        topic_right   = self.get_parameter("topic_right").value
        img_size      = self.get_parameter("img_size").value
        conf_thres    = self.get_parameter("conf_threshold").value
        iou_thres     = self.get_parameter("iou_threshold").value
        device        = self.get_parameter("device").value
        self.publish_vis = self.get_parameter("publish_vis").value
        self.undistort   = self.get_parameter("undistort").value
        proj_root        = self.get_parameter("project_root").value

        # 절대경로 보정
        if not os.path.isabs(weights):
            weights = os.path.join(proj_root, weights)
        if not os.path.isabs(camera_config):
            camera_config = os.path.join(proj_root, camera_config)

        self.get_logger().info(f"weights       : {weights}")
        self.get_logger().info(f"camera_config : {camera_config}")
        self.get_logger().info(f"topic_right   : {topic_right}")
        self.get_logger().info(f"device        : {device}")

        # ── 모델 초기화 ──────────────────────────────────────────────
        self.detector = SGLDetInference(
            weights=weights,
            img_size=img_size,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            device=device,
            mode="auto",
        )

        # SORT 트래커 (우측 카메라 1대 사용)
        self.tracker_right = SORTTracker(max_age=3, min_hits=1, iou_threshold=0.3)

        # BSD 인터페이스 (두 카메라 CoordTransformer 포함)
        self.bsd = BSDInterface(camera_config)

        # 왜곡 보정 (MORAI 시뮬레이션은 dist_coeffs=[0,0,0,0,0] → 비활성화)
        if self.undistort:
            self.calib = CameraCalibration(camera_config)
            self.get_logger().info("카메라 왜곡 보정 활성화")
        else:
            self.calib = None

        self.bridge = CvBridge()

        # ── QoS 설정 (실시간 카메라: BEST_EFFORT) ───────────────────
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ── Subscriber / Publisher ───────────────────────────────────
        # 우측 BSD 카메라만 구독 (단일 카메라 모드)
        self.sub_right = self.create_subscription(
            Image, topic_right,
            lambda msg: self._image_callback(msg, side="right"),
            sensor_qos,
        )

        self.pub_detections = self.create_publisher(
            Detection2DArray, "/bsd/detections", 10,
        )
        self.pub_warning = self.create_publisher(
            String, "/bsd/warning", 10,
        )
        if self.publish_vis:
            self.pub_vis_right = self.create_publisher(
                Image, "/bsd/visualization", 5,
            )

        self._latest_level = "SAFE"
        self.frame_count   = 0
        self.create_timer(5.0, self._log_stats)

        self.get_logger().info("BSD DeepNight 노드 시작 완료 (우측 BSD 카메라)")

    # ── 메인 콜백 ────────────────────────────────────────────────────────

    def _image_callback(self, msg: Image, side: str) -> None:
        """카메라 이미지 수신 콜백 (right/left 공용)."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"[{side}] 이미지 변환 실패: {e}")
            return

        # 1. 왜곡 보정 (옵션)
        if self.calib is not None:
            frame = self.calib.undistort(frame)

        # 2. SGLDet 추론
        detections = self.detector.detect(frame)

        # 3. SORT 추적
        tracker = self.tracker_right if side == "right" else self.tracker_left
        sort_input  = BSDInterface.format_sort_input(detections)
        sort_output = tracker.update(sort_input)
        _, track_ids = BSDInterface.parse_sort_output(sort_output, detections)

        if len(track_ids) != len(detections):
            track_ids = list(range(len(detections)))

        # 4. BSD 경고 판단
        h, w = frame.shape[:2]
        tracked_objs, any_danger = self.bsd.process(
            detections,
            side=side,
            tracked_ids=track_ids,
            img_w=w,
            img_h=h,
        )

        # 5. 경고 레벨 업데이트 및 발행
        levels = [o.alert_level for o in tracked_objs]
        self._latest_level = self._max_level(levels)

        self._publish_detections(msg.header, tracked_objs)
        self._publish_warning()

        # 6. 시각화
        if self.publish_vis:
            bsd_idx = [i for i, o in enumerate(tracked_objs) if o.is_bsd]
            vis = self.detector.visualize(frame, detections, bsd_idx, track_ids)

            level_str = self._latest_level
            color = (0, 0, 255) if level_str == "DANGER" \
                    else (0, 165, 255) if level_str == "WARNING" \
                    else (0, 200, 0)
            cv2.putText(vis, f"RIGHT BSD: {level_str}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            try:
                vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
                vis_msg.header = msg.header
                self.pub_vis_right.publish(vis_msg)
            except Exception as e:
                self.get_logger().warn(f"vis publish 실패: {e}")

        self.frame_count += 1

    # ── 발행 헬퍼 ────────────────────────────────────────────────────────

    def _publish_warning(self) -> None:
        """최신 경고 레벨 발행."""
        msg = String()
        msg.data = self._latest_level
        self.pub_warning.publish(msg)

    def _publish_detections(
        self, header: Header, tracked_objs: list,
    ) -> None:
        """vision_msgs/Detection2DArray 발행."""
        det_array = Detection2DArray()
        det_array.header = header

        for obj in tracked_objs:
            det = Detection2D()
            det.header = header

            x1, y1, x2, y2 = obj.bbox
            bbox = BoundingBox2D()
            bbox.center = Pose2D(
                x=float((x1 + x2) / 2),
                y=float((y1 + y2) / 2),
                theta=0.0,
            )
            bbox.size_x = float(x2 - x1)
            bbox.size_y = float(y2 - y1)
            det.bbox = bbox

            hyp = ObjectHypothesisWithPose()
            try:
                hyp.hypothesis.class_id = obj.cls_name
                hyp.hypothesis.score    = float(obj.conf)
            except AttributeError:
                # 구버전 vision_msgs 호환
                hyp.id    = obj.cls_id      # type: ignore
                hyp.score = float(obj.conf) # type: ignore

            det.results.append(hyp)
            det.id = f"{obj.side}_{obj.track_id}"

            det_array.detections.append(det)

        self.pub_detections.publish(det_array)

    # ── 유틸 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _max_level(levels: list[str]) -> str:
        """경고 레벨 최대값 반환."""
        if "DANGER" in levels:
            return "DANGER"
        if "WARNING" in levels:
            return "WARNING"
        return "SAFE"

    def _log_stats(self) -> None:
        self.get_logger().info(
            f"5초 처리 프레임: {self.frame_count}  현재 레벨: {self._latest_level}"
        )
        self.frame_count = 0


# ── 엔트리포인트 ────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = BSDDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
