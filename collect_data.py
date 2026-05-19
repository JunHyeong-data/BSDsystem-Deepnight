#!/usr/bin/env python3
"""
MORAI 데이터 자동 수집기
========================
Camera-1 (RGB)   : /image_jpeg/rgb         → sensor_msgs/CompressedImage
Camera-2 (Mask)  : /image_jpeg/compressed  → sensor_msgs/CompressedImage

동작 흐름:
  1. RGB + Instance 마스크 시간 동기화 수신
  2. Instance 마스크 → 연결 컴포넌트 → 객체 bbox 추출 (배경 제외)
  3. YOLOv8(COCO) 로 RGB 전체 추론 → IoU 매칭으로 각 인스턴스 클래스 결정
  4. YOLO 형식 라벨 (.txt) + RGB 이미지 (.jpg) 저장

저장 구조:
  data/morai/{condition}/images/  ← RGB .jpg
  data/morai/{condition}/labels/  ← YOLO .txt

BSD 클래스: 0=car, 1=pedestrian, 2=truck

실행:
  # Linux 노트북 (MORAI 있는 곳)에서 실행
  python3 collect_data.py --condition night
  python3 collect_data.py --condition dusk  --auto --interval 1.0
"""

import argparse
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

from ultralytics import YOLO


# ── BSD 클래스 정의 ──────────────────────────────────────────────────────────

BSD_CLASSES = {0: "car", 1: "pedestrian", 2: "truck"}

# COCO 클래스 → BSD 클래스 매핑
COCO_TO_BSD = {
    0: (1, "pedestrian"),   # person
    2: (0, "car"),          # car
    5: (2, "truck"),        # bus → truck
    7: (2, "truck"),        # truck
}

# 시각화 색상 (BGR)
CLASS_COLORS = {
    0: (0, 200, 0),     # car       → 초록
    1: (255, 128, 0),   # pedestrian → 주황
    2: (0, 0, 255),     # truck     → 빨강
}


# ── 유틸리티 ─────────────────────────────────────────────────────────────────

def decode_compressed(msg) -> np.ndarray | None:
    """CompressedImage → BGR numpy array."""
    try:
        arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def decode_raw(msg, bridge: CvBridge) -> np.ndarray | None:
    """sensor_msgs/Image → BGR numpy array."""
    try:
        return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except Exception:
        return None


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def iou(b1, b2) -> float:
    """IoU 계산. b = (x, y, w, h) 픽셀 좌표."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2])
    y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0:
        return 0.0
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0


# ── 메인 수집 노드 ────────────────────────────────────────────────────────────

class MoraiDataCollector(Node):
    """
    MORAI RGB + Instance 마스크 동기화 수신 → YOLO 라벨 자동 생성.

    핵심 전략:
      ① Instance 마스크의 비(非)백색 연결 컴포넌트 → 객체 bbox (위치 정확)
      ② YOLOv8을 RGB 전체에 적용 → 클래스 + 신뢰도 (의미론 정확)
      ③ IoU 매칭: ①의 bbox ↔ ②의 bbox → 클래스 합치기
      ④ YOLO가 못 잡은 인스턴스 → 종횡비/크기 휴리스틱으로 폴백
    """

    def __init__(self, args):
        super().__init__("morai_data_collector")

        # ── 설정 ──────────────────────────────────────────────────
        self.condition    = args.condition
        self.auto_save    = args.auto
        self.interval     = args.interval        # 자동 저장 간격 (초)
        self.min_area     = args.min_area        # 최소 객체 픽셀 면적
        self.conf_thres   = args.conf            # YOLO 분류 신뢰도 임계값
        self.iou_match    = args.iou_match       # 인스턴스-YOLO 매칭 IoU 임계값
        self.sync_tol     = 0.15                 # 프레임 동기화 허용 오차 (초)
        self.show_preview = not args.no_preview

        # Fish-eye 유효 원형 ROI (어안렌즈 가장자리 검은 테두리 제거용)
        self.fisheye_cx = args.fisheye_cx        # 원 중심 x (px)
        self.fisheye_cy = args.fisheye_cy        # 원 중심 y (px)
        self.fisheye_r  = args.fisheye_r         # 유효 반지름 (px)

        # ── 저장 경로 ─────────────────────────────────────────────
        out = Path(args.output)
        self.img_dir = out / self.condition / "images"
        self.lbl_dir = out / self.condition / "labels"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)

        # ── YOLOv8 분류기 (COCO 사전학습) ─────────────────────────
        self.get_logger().info("YOLOv8 COCO 로드 중...")
        self.yolo = YOLO(args.weights)
        self.get_logger().info("YOLOv8 로드 완료")

        self.bridge = CvBridge()

        # ── 최신 프레임 버퍼 ─────────────────────────────────────
        self._rgb_buf:  tuple | None = None   # (timestamp_sec, bgr_array)
        self._mask_buf: tuple | None = None   # (timestamp_sec, bgr_array)
        self._lock = threading.Lock()

        # ── 통계 ────────────────────────────────────────────────
        self.saved   = 0
        self.skipped = 0
        self._last_save_time = 0.0

        # ── QoS ─────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Subscriber 자동 타입 감지 ─────────────────────────────
        self._sub_rgb  = self._make_subscriber(
            args.rgb_topic,  "rgb",  qos,
        )
        self._sub_mask = self._make_subscriber(
            args.mask_topic, "mask", qos,
        )

        self.get_logger().info(f"조도 조건: {self.condition}")
        self.get_logger().info(f"저장 경로: {self.img_dir.parent}")
        mode = f"자동 ({self.interval}초 간격)" if self.auto_save else "수동 (Space=저장)"
        self.get_logger().info(f"저장 모드: {mode}")
        self.get_logger().info("준비 완료. 토픽 수신 대기 중...")

    # ── Subscriber 생성 ────────────────────────────────────────────────────

    def _make_subscriber(self, topic: str, role: str, qos):
        """
        CompressedImage 먼저 시도.
        만약 decode 실패가 반복되면 _cb_rgb_raw 로 자동 전환.
        MORAI /image_jpeg/ 토픽은 CompressedImage (JPEG).
        """
        cb = self._cb_rgb if role == "rgb" else self._cb_mask
        sub = self.create_subscription(CompressedImage, topic, cb, qos)
        self.get_logger().info(f"[{role}] 구독: {topic} (CompressedImage)")
        return sub

    def _make_raw_subscriber(self, topic: str, role: str, qos):
        """sensor_msgs/Image (raw) 구독 버전 — 필요 시 수동 전환."""
        cb = self._cb_rgb_raw if role == "rgb" else self._cb_mask_raw
        sub = self.create_subscription(Image, topic, cb, qos)
        self.get_logger().info(f"[{role}] 구독: {topic} (Image/raw)")
        return sub

    def _cb_rgb_raw(self, msg) -> None:
        img = decode_raw(msg, self.bridge)
        if img is None:
            return
        ts = stamp_to_sec(msg.header.stamp)
        with self._lock:
            self._rgb_buf = (ts, img)
        self._try_process()

    def _cb_mask_raw(self, msg) -> None:
        img = decode_raw(msg, self.bridge)
        if img is None:
            return
        ts = stamp_to_sec(msg.header.stamp)
        with self._lock:
            self._mask_buf = (ts, img)
        self._try_process()

    # ── 콜백 ──────────────────────────────────────────────────────────────

    def _cb_rgb(self, msg) -> None:
        img = decode_compressed(msg)
        if img is None:
            # CompressedImage decode 실패 시 raw Image로 재시도 (방어)
            return
        ts = stamp_to_sec(msg.header.stamp)
        with self._lock:
            self._rgb_buf = (ts, img)
        self._try_process()

    def _cb_mask(self, msg) -> None:
        img = decode_compressed(msg)
        if img is None:
            return
        ts = stamp_to_sec(msg.header.stamp)
        with self._lock:
            self._mask_buf = (ts, img)
        self._try_process()

    def _try_process(self) -> None:
        """두 버퍼가 충분히 가까운 타임스탬프면 처리."""
        with self._lock:
            if self._rgb_buf is None or self._mask_buf is None:
                return
            ts_rgb,  rgb  = self._rgb_buf
            ts_mask, mask = self._mask_buf

        # 시간 동기화 체크
        if abs(ts_rgb - ts_mask) > self.sync_tol:
            return

        # 자동 저장 간격 체크
        if self.auto_save:
            now = time.time()
            if now - self._last_save_time < self.interval:
                return
            self._do_save(rgb, mask)
            self._last_save_time = now

    # ── 핵심 처리 파이프라인 ──────────────────────────────────────────────

    def _do_save(self, rgb: np.ndarray, mask: np.ndarray) -> None:
        """RGB + 마스크 → YOLO 라벨 생성 및 저장."""
        H, W = rgb.shape[:2]

        # ① Instance 마스크 → bbox 리스트
        instance_bboxes = self._extract_instance_bboxes(mask)
        if not instance_bboxes:
            self.skipped += 1
            self.get_logger().debug("객체 없음, 스킵")
            return

        # ② YOLOv8 전체 이미지 추론 → COCO 탐지
        yolo_dets = self._run_yolo(rgb)

        # ③ IoU 매칭: 인스턴스 bbox ↔ YOLO 탐지
        detections = self._match_and_classify(instance_bboxes, yolo_dets, W, H)
        if not detections:
            self.skipped += 1
            return

        # ④ 파일 저장
        ts = int(time.time() * 1000)
        img_path = self.img_dir / f"{ts}.jpg"
        lbl_path = self.lbl_dir / f"{ts}.txt"

        cv2.imwrite(str(img_path), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])

        lines = []
        for det in detections:
            cx, cy, w, h = det["yolo"]
            lines.append(f"{det['cls_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        self.saved += 1
        summary = ", ".join(f"{d['cls_name']}({d['conf']:.2f})" for d in detections)
        self.get_logger().info(
            f"[{self.saved:4d}] {img_path.name} | {len(detections)}개 객체: {summary}"
        )

    # ── ① Instance 마스크 → bbox ─────────────────────────────────────────

    def _extract_instance_bboxes(self, mask: np.ndarray) -> list[tuple]:
        """
        흰색 배경을 제외한 객체 픽셀을 연결 컴포넌트로 분리.

        흰색 기준: R, G, B 모두 240 이상.
        각 컴포넌트 → (x, y, w, h) 픽셀 bbox.

        Fish-eye ROI 필터:
          179° 어안렌즈의 유효 원형 영역 밖(검은 원형 테두리)은
          connectedComponents 이전에 마스킹하여 가짜 bbox 원천 차단.
          중심 (640, 360), 반지름 350 px (1280×720 기준).
        """
        H, W = mask.shape[:2]

        # ── ① 흰색 배경 제거 ─────────────────────────────────────
        bg_mask = np.all(mask >= 240, axis=2)   # True = 배경
        obj_bin = (~bg_mask).astype(np.uint8) * 255

        # ── ② Fish-eye 유효 원형 ROI 마스크 적용 ─────────────────
        # 어안렌즈 가장자리 검은 테두리를 객체로 잘못 인식하는 것을 방지
        roi_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(roi_mask,
                   center=(self.fisheye_cx, self.fisheye_cy),
                   radius=self.fisheye_r,
                   color=255,
                   thickness=-1)   # 속이 꽉 찬 원
        obj_bin = cv2.bitwise_and(obj_bin, roi_mask)

        # ── ③ 노이즈 제거 (모폴로지 오픈) ───────────────────────
        kernel = np.ones((3, 3), np.uint8)
        obj_bin = cv2.morphologyEx(obj_bin, cv2.MORPH_OPEN, kernel)

        # ── ④ 연결 컴포넌트 ──────────────────────────────────────
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            obj_bin, connectivity=8
        )

        H, W = mask.shape[:2]
        bboxes = []
        for i in range(1, num_labels):       # 0 = 배경 컴포넌트
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area:
                continue

            x = max(0, stats[i, cv2.CC_STAT_LEFT])
            y = max(0, stats[i, cv2.CC_STAT_TOP])
            w = min(stats[i, cv2.CC_STAT_WIDTH],  W - x)
            h = min(stats[i, cv2.CC_STAT_HEIGHT], H - y)

            # 비정상 bbox 필터 (너무 얇거나 작으면 제거)
            if w < 10 or h < 10:
                continue

            bboxes.append((x, y, w, h))

        return bboxes

    # ── ② YOLOv8 전체 이미지 추론 ────────────────────────────────────────

    def _run_yolo(self, rgb: np.ndarray) -> list[dict]:
        """
        COCO 사전학습 YOLOv8으로 전체 이미지 탐지.
        BSD 관련 클래스(car/person/truck/bus)만 반환.
        """
        results = self.yolo.predict(
            rgb,
            conf=self.conf_thres,
            verbose=False,
        )
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in COCO_TO_BSD:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bsd_id, bsd_name = COCO_TO_BSD[cls_id]
                dets.append({
                    "bbox":     (x1, y1, x2-x1, y2-y1),   # (x,y,w,h)
                    "cls_id":   bsd_id,
                    "cls_name": bsd_name,
                    "conf":     float(box.conf[0].item()),
                })
        return dets

    # ── ③ IoU 매칭: 인스턴스 ↔ YOLO ─────────────────────────────────────

    def _match_and_classify(
        self,
        inst_bboxes: list[tuple],
        yolo_dets:   list[dict],
        W: int,
        H: int,
    ) -> list[dict]:
        """
        Instance 마스크 bbox(위치 정확) + YOLO 탐지(클래스 정확) → 합치기.

        매칭 기준: IoU > iou_match 이면 YOLO 클래스 사용.
        매칭 실패 시: 종횡비·크기 기반 휴리스틱으로 클래스 추정.
        """
        matched = set()   # YOLO 탐지 중 이미 매칭된 인덱스
        detections = []

        for ibbox in inst_bboxes:
            best_iou  = 0.0
            best_det  = None
            best_idx  = -1

            for j, yd in enumerate(yolo_dets):
                if j in matched:
                    continue
                score = iou(ibbox, yd["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_det = yd
                    best_idx = j

            # YOLO와 매칭됨
            if best_iou >= self.iou_match and best_det is not None:
                cls_id   = best_det["cls_id"]
                cls_name = best_det["cls_name"]
                conf     = best_det["conf"]
                matched.add(best_idx)

            # 매칭 실패 → 크기 휴리스틱
            else:
                cls_id, cls_name = self._size_heuristic(*ibbox)
                conf = 0.40   # 휴리스틱 신뢰도는 낮게

            # YOLO 정규화 좌표 변환
            x, y, w, h = ibbox
            cx_n = (x + w / 2) / W
            cy_n = (y + h / 2) / H
            w_n  = w / W
            h_n  = h / H

            # 경계 클리핑 (0~1 사이)
            cx_n = max(0.0, min(1.0, cx_n))
            cy_n = max(0.0, min(1.0, cy_n))
            w_n  = max(0.0, min(1.0, w_n))
            h_n  = max(0.0, min(1.0, h_n))

            detections.append({
                "cls_id":   cls_id,
                "cls_name": cls_name,
                "conf":     conf,
                "bbox":     ibbox,
                "yolo":     (cx_n, cy_n, w_n, h_n),
            })

        return detections

    # ── 크기 휴리스틱 (YOLO 매칭 실패 시 폴백) ───────────────────────────

    def _size_heuristic(self, x, y, w, h) -> tuple[int, str]:
        """
        종횡비 + 면적으로 클래스 추정.

        기준 (BSD 카메라, 측면 뷰):
          - 종횡비 h/w > 1.5 → pedestrian  (키 큰 형태)
          - 면적 > 40000 px   → truck       (대형)
          - 그 외              → car
        """
        aspect = h / (w + 1e-6)
        area   = w * h

        if aspect > 1.5:
            return (1, "pedestrian")
        elif area > 40000:
            return (2, "truck")
        else:
            return (0, "car")

    # ── 미리보기 + 수동 저장 ──────────────────────────────────────────────

    def preview_and_input(self) -> bool:
        """
        미리보기 창 갱신 + 키 입력 처리.
        메인 스레드에서 호출.
        반환값: False = 종료 요청
        """
        with self._lock:
            if self._rgb_buf is None:
                return True
            _, rgb  = self._rgb_buf
            mask_data = self._mask_buf

        vis = rgb.copy()

        # Instance 마스크가 있으면 bbox 오버레이
        if mask_data is not None:
            _, mask = mask_data
            bboxes = self._extract_instance_bboxes(mask)
            for (x, y, w, h) in bboxes:
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)

            # 마스크 창
            if self.show_preview:
                cv2.imshow("Instance Mask", mask)

        # HUD
        mode_txt = "AUTO" if self.auto_save else "MANUAL"
        cv2.putText(vis, f"[{mode_txt}] Saved:{self.saved} Skip:{self.skipped} | {self.condition}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        if not self.auto_save:
            cv2.putText(vis, "SPACE=save  A=auto  Q=quit",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        if self.show_preview:
            cv2.imshow("MORAI BSD Collector", vis)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:   # q / ESC = 종료
            return False

        elif key == ord(' ') and not self.auto_save:
            # 수동 저장
            with self._lock:
                if self._rgb_buf and self._mask_buf:
                    _, rgb_  = self._rgb_buf
                    _, mask_ = self._mask_buf
                else:
                    rgb_ = mask_ = None
            if rgb_ is not None:
                self._do_save(rgb_.copy(), mask_.copy())

        elif key == ord('a'):
            self.auto_save = not self.auto_save
            mode = "자동" if self.auto_save else "수동"
            self.get_logger().info(f"저장 모드 전환 → {mode}")

        elif key == ord('+') or key == ord('='):
            self.interval = max(0.2, self.interval - 0.2)
            self.get_logger().info(f"저장 간격 → {self.interval:.1f}초")

        elif key == ord('-'):
            self.interval += 0.2
            self.get_logger().info(f"저장 간격 → {self.interval:.1f}초")

        return True


# ── 실행 엔트리포인트 ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MORAI BSD 데이터 수집기")

    p.add_argument("--condition", "-c",
                   choices=["dusk", "night"],
                   default="night",
                   help="조도 조건 (저장 폴더 분류용)")
    p.add_argument("--output", "-o",
                   default="data/morai",
                   help="데이터 저장 루트 경로 (기본: data/morai)")
    p.add_argument("--rgb-topic",
                   default="/image_jpeg/rgb",
                   help="Camera-1 RGB 토픽")
    p.add_argument("--mask-topic",
                   default="/image_jpeg/compressed",
                   help="Camera-2 Instance 마스크 토픽")
    p.add_argument("--weights",
                   default="yolov8m.pt",
                   help="YOLO 분류용 가중치 (기본: yolov8m.pt COCO)")
    p.add_argument("--auto", action="store_true",
                   help="자동 저장 모드 (기본: 수동)")
    p.add_argument("--interval", type=float, default=0.5,
                   help="자동 저장 간격 (초, 기본: 0.5)")
    p.add_argument("--min-area", type=int, default=800,
                   help="최소 객체 픽셀 면적 (기본: 800)")
    p.add_argument("--conf", type=float, default=0.20,
                   help="YOLO 분류 신뢰도 임계값 (기본: 0.20)")
    p.add_argument("--iou-match", type=float, default=0.30,
                   help="인스턴스-YOLO 매칭 IoU 임계값 (기본: 0.30)")
    p.add_argument("--no-preview", action="store_true",
                   help="미리보기 창 비활성화 (원격 SSH 환경)")

    # Fish-eye ROI (어안렌즈 가장자리 검은 테두리 제거)
    p.add_argument("--fisheye-cx", type=int, default=640,
                   help="Fish-eye 유효 원 중심 x (기본: 640)")
    p.add_argument("--fisheye-cy", type=int, default=360,
                   help="Fish-eye 유효 원 중심 y (기본: 360)")
    p.add_argument("--fisheye-r",  type=int, default=350,
                   help="Fish-eye 유효 반지름 px (기본: 350)")

    return p.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = MoraiDataCollector(args)

    # ROS spin → 별도 스레드
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("\n" + "="*55)
    print("  MORAI BSD 데이터 수집기 시작")
    print(f"  조도 조건  : {args.condition}")
    print(f"  RGB  토픽  : {args.rgb_topic}")
    print(f"  Mask 토픽  : {args.mask_topic}")
    print(f"  저장 경로  : {args.output}/{args.condition}/")
    if args.auto:
        print(f"  자동 저장  : {args.interval}초 간격")
    else:
        print("  수동 저장  : Space 키")
    print("  SPACE=저장  A=모드전환  +=빠르게  -=느리게  Q=종료")
    print("="*55 + "\n")

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
        print(f"\n수집 완료: {node.saved}장 저장, {node.skipped}장 스킵")
        print(f"저장 위치: {node.img_dir.parent}")


if __name__ == "__main__":
    main()
