#!/usr/bin/env python3
"""
MORAI 원시 이미지 수집기 (Linux 전용)
======================================
rclpy + cv2 만 사용. YOLOv8 불필요.

Camera-1 (RGB)  : /image_jpeg/rgb        → CompressedImage
Camera-2 (Mask) : /image_jpeg/compressed → CompressedImage

저장 결과 (raw_data/ 폴더):
  {timestamp}_rgb.jpg    ← RGB 이미지
  {timestamp}_mask.jpg   ← Instance 마스크

이후 Windows에서 generate_labels.py 로 YOLO 라벨 생성.

실행:
  python3 collect_raw.py
  python3 collect_raw.py --auto --interval 1.0
  python3 collect_raw.py --condition night --output raw_data
  python3 collect_raw.py --no-preview          # SSH 환경 (창 없음)
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
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge


class RawCollector(Node):
    """RGB + Instance 마스크를 파일로 저장하는 최소 ROS2 노드."""

    def __init__(self, args):
        super().__init__("morai_raw_collector")

        self.auto_save    = args.auto
        self.interval     = args.interval
        self.sync_tol     = 0.15
        self.show_preview = not args.no_preview

        # 저장 경로
        self.out_dir = Path(args.output) / args.condition
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()

        self._rgb_buf:  tuple | None = None   # (ts, bgr)
        self._mask_buf: tuple | None = None   # (ts, bgr)
        self._lock = threading.Lock()

        self.saved   = 0
        self.skipped = 0
        self._last_save = 0.0

        # QoS
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(
            CompressedImage, args.rgb_topic,
            self._cb_rgb, qos,
        )
        self.create_subscription(
            CompressedImage, args.mask_topic,
            self._cb_mask, qos,
        )

        self.get_logger().info(f"저장 경로: {self.out_dir}")
        self.get_logger().info("토픽 수신 대기 중...")

    # ── 콜백 ─────────────────────────────────────────────────────────────

    def _decode(self, msg) -> np.ndarray | None:
        """CompressedImage → BGR. 실패 시 None."""
        try:
            arr = np.frombuffer(msg.data, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _decode_raw(self, msg) -> np.ndarray | None:
        """sensor_msgs/Image → BGR. (토픽이 raw인 경우)"""
        try:
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return None

    def _cb_rgb(self, msg) -> None:
        img = self._decode(msg)
        if img is None:
            return
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._lock:
            self._rgb_buf = (ts, img)
        self._try_autosave()

    def _cb_mask(self, msg) -> None:
        img = self._decode(msg)
        if img is None:
            return
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._lock:
            self._mask_buf = (ts, img)
        self._try_autosave()

    def _try_autosave(self) -> None:
        if not self.auto_save:
            return
        now = time.time()
        if now - self._last_save < self.interval:
            return
        self._save_pair()
        self._last_save = now

    # ── 저장 ─────────────────────────────────────────────────────────────

    def _save_pair(self) -> bool:
        """동기화된 RGB + 마스크 쌍 저장. 성공 시 True."""
        with self._lock:
            if self._rgb_buf is None or self._mask_buf is None:
                return False
            ts_r, rgb  = self._rgb_buf
            ts_m, mask = self._mask_buf

        # 동기화 체크
        if abs(ts_r - ts_m) > self.sync_tol:
            self.skipped += 1
            self.get_logger().warn(
                f"프레임 비동기 ({abs(ts_r-ts_m):.3f}s > {self.sync_tol}s), 스킵"
            )
            return False

        ts_ms = int(time.time() * 1000)
        rgb_path  = self.out_dir / f"{ts_ms}_rgb.jpg"
        mask_path = self.out_dir / f"{ts_ms}_mask.jpg"

        cv2.imwrite(str(rgb_path),  rgb,  [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(mask_path), mask, [cv2.IMWRITE_JPEG_QUALITY, 95])

        self.saved += 1
        self.get_logger().info(f"[{self.saved:4d}] {rgb_path.name}")
        return True

    # ── 미리보기 ──────────────────────────────────────────────────────────

    def preview_and_input(self) -> bool:
        """메인 스레드: 미리보기 + 키 입력. False = 종료."""
        with self._lock:
            rgb_data  = self._rgb_buf
            mask_data = self._mask_buf

        # 미리보기
        if self.show_preview:
            if rgb_data is not None:
                vis = rgb_data[1].copy()
                mode = "AUTO" if self.auto_save else "SPACE=save"
                cv2.putText(
                    vis,
                    f"[{mode}] saved={self.saved} | {self.out_dir.parent.name}/{self.out_dir.name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2,
                )
                cv2.imshow("RGB (Camera-1)", vis)
            if mask_data is not None:
                cv2.imshow("Instance Mask (Camera-2)", mask_data[1])

        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):
            return False
        elif key == ord(' ') and not self.auto_save:
            self._save_pair()
        elif key == ord('a'):
            self.auto_save = not self.auto_save
            self.get_logger().info("모드: " + ("자동" if self.auto_save else "수동"))
        elif key == ord('+') or key == ord('='):
            self.interval = max(0.2, self.interval - 0.2)
            self.get_logger().info(f"간격 → {self.interval:.1f}초")
        elif key == ord('-'):
            self.interval += 0.2
            self.get_logger().info(f"간격 → {self.interval:.1f}초")

        return True


def parse_args():
    p = argparse.ArgumentParser(description="MORAI 원시 이미지 수집기 (Linux)")
    p.add_argument("--condition", "-c",
                   choices=["dusk", "night"], default="night")
    p.add_argument("--output", "-o", default="raw_data",
                   help="저장 루트 폴더 (기본: raw_data)")
    p.add_argument("--rgb-topic",  default="/image_jpeg/rgb")
    p.add_argument("--mask-topic", default="/image_jpeg/compressed")
    p.add_argument("--auto", action="store_true", help="자동 저장 모드")
    p.add_argument("--interval", type=float, default=1.0,
                   help="자동 저장 간격 (초)")
    p.add_argument("--no-preview", action="store_true",
                   help="미리보기 창 없음 (SSH 환경)")
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = RawCollector(args)

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    spin_thread.start()

    print(f"\n{'='*50}")
    print(f"  저장 경로  : {node.out_dir}")
    print(f"  RGB  토픽  : {args.rgb_topic}")
    print(f"  Mask 토픽  : {args.mask_topic}")
    print(f"  저장 모드  : {'자동 ' + str(args.interval) + '초' if args.auto else '수동 (Space)'}")
    print(f"  SPACE=저장  A=자동전환  +/-=간격  Q=종료")
    print(f"{'='*50}\n")

    try:
        while rclpy.ok():
            if not node.preview_and_input():
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        print(f"\n수집 완료: {node.saved}쌍 저장")
        print(f"저장 위치: {node.out_dir}")
        print(f"\n다음 단계 (Windows에서):")
        print(f"  python generate_labels.py --input raw_data/{args.condition} --condition {args.condition}")


if __name__ == "__main__":
    main()
