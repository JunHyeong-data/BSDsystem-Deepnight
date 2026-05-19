"""
SGLDet 추론 전용 모듈 (Part B: 구동 단계)
------------------------------------------
학습 완료 후 AuxDecoder 없이 YOLOv8만 실행.

지원 모드:
  1. COCO 사전학습 (yolov8m.pt)
     → BSD 관련 클래스만 필터링: car, truck, person, motorcycle
  2. MORAI 파인튜닝 (best_model.pt)
     → 4개 클래스 그대로 사용
"""

import torch
import numpy as np
import cv2
from ultralytics import YOLO


# COCO 클래스 → BSD 관련 매핑
# COCO 클래스 인덱스: person=0, car=2, bus=5, truck=7
COCO_BSD_CLASSES = {
    0: "pedestrian",   # person
    2: "car",
    5: "truck",        # bus도 truck으로 묶음
    7: "truck",
}

# MORAI 학습된 모델 클래스 (3개)
MORAI_CLASSES = ["car", "pedestrian", "truck"]


class SGLDetInference:
    """
    SGLDet 경량 추론기 (보조 파이프라인 없음).

    Args:
        weights   : .pt 가중치 경로
        img_size  : 입력 크기
        conf_thres: 신뢰도 임계값
        iou_thres : NMS IoU 임계값
        device    : "cuda", "cpu", or None (자동)
        mode      : "auto" / "coco" / "morai"
                    auto: 가중치의 nc를 보고 자동 판단
    """

    def __init__(
        self,
        weights: str = "yolov8m.pt",
        img_size: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        device: str | None = None,
        mode: str = "auto",
    ):
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[SGLDetInference] 가중치 로드: {weights}")
        self.model = YOLO(weights)
        self.model.to(self.device)

        # 모드 판단
        nc = self._get_num_classes()
        if mode == "auto":
            self.mode = "coco" if nc >= 80 else "morai"
        else:
            self.mode = mode

        print(f"[SGLDetInference] device={self.device} | mode={self.mode} | nc={nc}")

    def _get_num_classes(self) -> int:
        """모델의 클래스 수 추출."""
        try:
            return int(self.model.model.nc)
        except Exception:
            try:
                return len(self.model.names)
            except Exception:
                return 80   # 기본값

    def detect(self, frame: np.ndarray) -> list:
        """
        단일 프레임 탐지.

        Args:
            frame: (H, W, 3) BGR 이미지
        Returns:
            list of detection dicts with keys:
              bbox, cls_id, cls_name, conf,
              cx_norm, cy_norm, w_norm, h_norm
        """
        results = self.model.predict(
            frame,
            imgsz=self.img_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )

        detections = []
        h, w = frame.shape[:2]

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0].item())

                # 모드별 클래스 필터링/매핑
                if self.mode == "coco":
                    if cls_id not in COCO_BSD_CLASSES:
                        continue
                    cls_name = COCO_BSD_CLASSES[cls_id]
                else:   # morai
                    if cls_id < len(MORAI_CLASSES):
                        cls_name = MORAI_CLASSES[cls_id]
                    else:
                        cls_name = str(cls_id)

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].item())

                detections.append({
                    "bbox":     [int(x1), int(y1), int(x2), int(y2)],
                    "cls_id":   cls_id,
                    "cls_name": cls_name,
                    "conf":     conf,
                    "cx_norm":  ((x1 + x2) / 2) / w,
                    "cy_norm":  ((y1 + y2) / 2) / h,
                    "w_norm":   (x2 - x1) / w,
                    "h_norm":   (y2 - y1) / h,
                })

        return detections

    def visualize(
        self,
        frame: np.ndarray,
        detections: list,
        bsd_indices: list | None = None,
        track_ids: list | None = None,
    ) -> np.ndarray:
        """
        탐지 결과 시각화.

        Args:
            frame      : 원본 BGR 프레임
            detections : detect() 결과
            bsd_indices: BSD 위험 인덱스 (빨간색)
            track_ids  : 각 detection의 SORT track id (선택)
        """
        vis = frame.copy()
        bsd_indices = bsd_indices or []

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            is_bsd = i in bsd_indices

            # 색상: 위험=빨강, 안전=초록
            color = (0, 0, 255) if is_bsd else (0, 200, 0)
            thick = 3 if is_bsd else 2

            label_parts = []
            if track_ids is not None and i < len(track_ids):
                label_parts.append(f"#{track_ids[i]}")
            label_parts.append(f"{det['cls_name']} {det['conf']:.2f}")
            if is_bsd:
                label_parts.append("BSD!")
            label = " ".join(label_parts)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)

            # 라벨 배경
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis
