#!/usr/bin/env python3
"""
MORAI GT 기반 BSD 데이터 수집기
================================
[기존 방식 문제]
  Instance mask bbox + YOLOv8(COCO) → IoU 매칭으로 클래스 결정
  → 야간 저조도에서 YOLOv8 오탐, 매칭 실패 빈발 (교수님 지적 반영)

[이 버전: GT 직접 활용]
  /Object_topic (morai_msgs/ObjectStatusList) → 클래스 + 3D 위치 (GT)
  /Ego_topic    (morai_msgs/EgoVehicleStatus)  → 자차 위치 + heading
  Instance Mask (/image_jpeg/compressed)       → 2D bbox 추출

  클래스 (2개):
    npc_list        → 0: vehicle   (차량 전체: 승용차/트럭/버스 모두)
    pedestrian_list → 1: pedestrian

  매칭:
    GT 객체의 3D 위치 → 이미지 투영 → instance bbox 안에 포함 여부 확인
    YOLOv8 없이 100% GT 기반 클래스 할당

저장:
  data/morai/{condition}/images/ *.jpg
  data/morai/{condition}/labels/ *.txt  (YOLO: class cx cy w h)

실행:
  python3 collect_data.py --condition night
  python3 collect_data.py --condition dusk --auto --interval 0.5
"""

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage

from morai_msgs.msg import ObjectStatusList, EgoVehicleStatus


# ── 클래스 정의 (2클래스) ────────────────────────────────────────────────────

BSD_CLASSES  = {0: "vehicle", 1: "pedestrian"}
CLASS_COLORS = {0: (0, 200, 0), 1: (255, 128, 0)}


# ── 카메라 파라미터 (camera_config.yaml 기준 — MORAI Fisheye Equidistant) ─────
# 투영 모델 : Equidistant fisheye  (r = f · θ)
# 왜곡 계수 : [k1,k2,k3,k4] = [0,0,0,0] (MORAI는 이상적 모델)
# fx 산출  : (W/2) / (FOV_h/2 rad) = 640 / 1.5621 ≈ 409.73 (수평 FOV 179°)

FX, FY  = 409.73, 409.73
CX, CY  = 640.0, 360.0
IMG_W   = 1280
IMG_H   = 720
FOV_HALF_RAD = np.deg2rad(89.5)   # 수평 FOV 179° / 2 — 광선 가시 한계

MOUNT       = np.array([2.150, 0.900, 0.550])  # 우측 BSD 카메라 장착 위치 (m)
PITCH_DEG   = 15.0
YAW_DEG     = 90.0

SYNC_TOL    = 0.15    # 동기화 허용 오차 (초)
MIN_AREA    = 800     # instance mask 최소 픽셀 면적

# Fisheye 유효 영역 반지름 = fx · θ_max ≈ 409.73 · 1.562 ≈ 640
FISHEYE_CX, FISHEYE_CY, FISHEYE_R = 640, 360, 640


# ── 카메라 변환 행렬 사전 계산 ───────────────────────────────────────────────

def _Rz(d):
    t = np.deg2rad(d)
    return np.array([[np.cos(t), -np.sin(t), 0],
                     [np.sin(t),  np.cos(t), 0],
                     [0, 0, 1]], dtype=float)

def _Ry(d):
    t = np.deg2rad(d)
    return np.array([[ np.cos(t), 0, np.sin(t)],
                     [0, 1, 0],
                     [-np.sin(t), 0, np.cos(t)]], dtype=float)

_R_CAM_BASE = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]], dtype=float)
_R_C2V = _Rz(YAW_DEG) @ _Ry(PITCH_DEG) @ _R_CAM_BASE
_R_V2C = _R_C2V.T


def world_to_pixel(obj_pos, ego_pos, ego_heading_deg):
    """
    ENU 월드 좌표 → fisheye 이미지 픽셀 (Equidistant 모델).

    투영 수식 (OpenCV cv2.fisheye 표준):
        r_xy  = sqrt(X² + Y²)
        θ     = atan2(r_xy, Z)              # 광축에서 광선까지 각도
        u     = fx · (θ / r_xy) · X + cx    # k1..k4 = 0
        v     = fy · (θ / r_xy) · Y + cy

    핀홀 투영(u = fx·X/Z + cx)과 달리 cam[2] < 0 (광축 뒤)이라도
    θ < FOV/2 인 한 가시. 단 θ > 89.5°(FOV 한계)면 None.

    MORAI heading 가정: 0 = North(북쪽), 시계방향 양수.
    # TODO: `ros2 topic echo /Ego_topic` 으로 실제 확인 필요.
    #   차량이 북쪽을 향할 때 heading ≈ 0 이면 맞음.
    #   동쪽(East)이 0이라면 h = np.deg2rad(ego_heading_deg - 90) 으로 수정.
    """
    dx = obj_pos.x - ego_pos.x   # East
    dy = obj_pos.y - ego_pos.y   # North
    dz = obj_pos.z - ego_pos.z   # Up

    h = np.deg2rad(ego_heading_deg)
    veh = np.array([
        dx * np.sin(h) + dy * np.cos(h),   # X 전방
        dx * np.cos(h) - dy * np.sin(h),   # Y 우측
        dz,
    ])

    cam = _R_V2C @ (veh - MOUNT)   # 카메라 프레임 (X=우, Y=하, Z=광축)
    X, Y, Z = cam[0], cam[1], cam[2]

    r_xy  = np.hypot(X, Y)
    theta = np.arctan2(r_xy, Z)    # 광축에서 광선까지 각도 [0, π]

    if theta > FOV_HALF_RAD:        # FOV 밖 (카메라 뒤편 포함)
        return None

    if r_xy < 1e-9:                 # 광축 위의 점
        u, v = int(CX), int(CY)
    else:
        f = theta / r_xy            # equidistant 스케일 (k=0)
        u = int(FX * f * X + CX)
        v = int(FY * f * Y + CY)

    return (u, v) if (0 <= u < IMG_W and 0 <= v < IMG_H) else None


# ── Instance Mask → 2D bbox ───────────────────────────────────────────────────

def extract_bboxes(mask_bgr):
    """Instance mask → [(x1,y1,x2,y2), ...] 픽셀 bbox 목록."""
    white   = np.all(mask_bgr >= 240, axis=2)
    obj_msk = (~white).astype(np.uint8) * 255

    roi = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(roi, (FISHEYE_CX, FISHEYE_CY), FISHEYE_R, 255, -1)
    obj_msk = cv2.bitwise_and(obj_msk, roi)

    n, _, stats, _ = cv2.connectedComponentsWithStats(obj_msk, connectivity=8)
    bboxes = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        x1 = stats[i, cv2.CC_STAT_LEFT]
        y1 = stats[i, cv2.CC_STAT_TOP]
        x2 = x1 + stats[i, cv2.CC_STAT_WIDTH]
        y2 = y1 + stats[i, cv2.CC_STAT_HEIGHT]
        bboxes.append((x1, y1, x2, y2))
    return bboxes


def assign_classes(obj_msg, bboxes, ego_pos, ego_heading):
    """
    GT 객체 3D 위치 → 이미지 투영 → instance bbox에 클래스 할당.

    Returns:
        [(cls_id, (x1,y1,x2,y2)), ...]
    """
    assigned = {}   # bbox_idx → cls_id

    # pedestrian 우선 처리
    for obj in obj_msg.pedestrian_list:
        px = world_to_pixel(obj.position, ego_pos, ego_heading)
        if px is None:
            continue
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            if i not in assigned and x1 <= px[0] <= x2 and y1 <= px[1] <= y2:
                assigned[i] = 1
                break

    # npc (차량 전체 → vehicle 0)
    for obj in obj_msg.npc_list:
        px = world_to_pixel(obj.position, ego_pos, ego_heading)
        if px is None:
            continue
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            if i not in assigned and x1 <= px[0] <= x2 and y1 <= px[1] <= y2:
                assigned[i] = 0
                break

    return [(cls, bboxes[i]) for i, cls in assigned.items()]


# ── YOLO 라벨 저장 ────────────────────────────────────────────────────────────

def save_sample(rgb, labeled, img_path, lbl_path):
    cv2.imwrite(str(img_path), rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
    lines = []
    for cls, (x1, y1, x2, y2) in labeled:
        cx = ((x1 + x2) / 2) / IMG_W
        cy = ((y1 + y2) / 2) / IMG_H
        w  = (x2 - x1) / IMG_W
        h  = (y2 - y1) / IMG_H
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── ROS2 노드 ─────────────────────────────────────────────────────────────────

class GTCollector(Node):

    def __init__(self, condition, output_root, auto_save, interval):
        super().__init__("gt_collector")
        self.condition  = condition
        self.auto_save  = auto_save
        self.interval   = interval
        self._last_save = 0.0
        self._lock      = threading.Lock()
        self._saved     = 0

        base = Path(output_root) / condition
        self.img_dir = base / "images"
        self.lbl_dir = base / "labels"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)

        self._rgb_buf  = None   # (stamp, bgr)
        self._mask_buf = None
        self._obj_buf  = None
        self._ego_buf  = None

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(CompressedImage,  "/image_jpeg/rgb",
                                 self._cb_rgb,  qos)
        self.create_subscription(CompressedImage,  "/image_jpeg/compressed",
                                 self._cb_mask, qos)
        self.create_subscription(ObjectStatusList, "/Object_topic",
                                 self._cb_obj,  qos)
        self.create_subscription(EgoVehicleStatus, "/Ego_topic",
                                 self._cb_ego,  qos)

        self.get_logger().info(
            f"GT Collector 시작 | 조건: {condition} | "
            f"{'자동' if auto_save else '수동(스페이스바)'} | "
            f"클래스: {BSD_CLASSES}"
        )

    def _stamp(self, msg):
        s = msg.header.stamp
        return s.sec + s.nanosec * 1e-9

    def _decode(self, msg):
        arr = np.frombuffer(msg.data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _cb_rgb(self, msg):
        img = self._decode(msg)
        if img is not None:
            with self._lock:
                self._rgb_buf = (self._stamp(msg), img)

    def _cb_mask(self, msg):
        img = self._decode(msg)
        if img is not None:
            with self._lock:
                self._mask_buf = (self._stamp(msg), img)

    def _cb_obj(self, msg):
        with self._lock:
            self._obj_buf = (self._stamp(msg), msg)

    def _cb_ego(self, msg):
        with self._lock:
            self._ego_buf = (self._stamp(msg), msg)

    def _get_synced(self):
        with self._lock:
            bufs = [self._rgb_buf, self._mask_buf,
                    self._obj_buf, self._ego_buf]
            if any(b is None for b in bufs):
                return None
            bufs = list(bufs)

        if max(b[0] for b in bufs) - min(b[0] for b in bufs) > SYNC_TOL:
            return None
        return bufs[0][1], bufs[1][1], bufs[2][1], bufs[3][1]

    def _process_and_save(self):
        data = self._get_synced()
        if data is None:
            return False

        rgb, mask, obj_msg, ego_msg = data
        bboxes = extract_bboxes(mask)
        if not bboxes:
            return False

        labeled = assign_classes(obj_msg, bboxes,
                                 ego_msg.position, ego_msg.heading)
        if not labeled:
            self.get_logger().warn("GT 매칭 객체 없음 — 스킵")
            return False

        ts = int(time.time() * 1000)
        save_sample(rgb, labeled,
                    self.img_dir / f"{ts}.jpg",
                    self.lbl_dir / f"{ts}.txt")
        self._saved += 1
        summary = " | ".join(f"{BSD_CLASSES[c]}" for c, _ in labeled)
        self.get_logger().info(f"[{self._saved}] {ts}.jpg | {summary}")
        return True

    def run_auto(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            now = time.time()
            if now - self._last_save >= self.interval:
                if self._process_and_save():
                    self._last_save = now

    def run_manual(self):
        cv2.namedWindow("BSD GT Collector", cv2.WINDOW_NORMAL)
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            data = self._get_synced()
            if data is not None:
                rgb, mask, obj_msg, ego_msg = data
                bboxes  = extract_bboxes(mask)
                labeled = assign_classes(obj_msg, bboxes,
                                         ego_msg.position, ego_msg.heading)
                preview = rgb.copy()
                for cls, (x1, y1, x2, y2) in labeled:
                    col = CLASS_COLORS[cls]
                    cv2.rectangle(preview, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(preview, BSD_CLASSES[cls],
                                (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, col, 1)
                cv2.imshow("BSD GT Collector", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                self._process_and_save()
            elif key in (ord('q'), 27):
                break
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", "-c", choices=["dusk", "night"], required=True)
    ap.add_argument("--output",    "-o", default="data/morai")
    ap.add_argument("--auto",      action="store_true")
    ap.add_argument("--interval",  "-i", type=float, default=0.5)
    args = ap.parse_args()

    rclpy.init()
    node = GTCollector(args.condition, args.output, args.auto, args.interval)
    try:
        if args.auto:
            node.run_auto()
        else:
            node.run_manual()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"총 저장: {node._saved}장")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
