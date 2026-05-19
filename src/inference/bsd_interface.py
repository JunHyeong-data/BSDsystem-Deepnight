"""
BSD 경고 인터페이스 (SORT 트래커 연동)
----------------------------------------
SGLDet 탐지 결과 → SORT 추적 → BSD 경고 영역 판단

두 카메라 지원:
  Camera-1 (camera_right) : 우측 BSD  — yaw=90°
  Camera-2 (camera_left)  : 좌측 BSD  — yaw=270°

전체 파이프라인:
  detector.py → [탐지 bbox] → bsd_interface.py → [경고 여부]
                                    ↑
                              sort_tracker.py
                              coord_transform.py (두 카메라 변환)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.preprocessing.coord_transform import CoordTransformer


@dataclass
class TrackedObject:
    """SORT 추적 결과 + BSD 판단 정보."""
    track_id:    int
    cls_id:      int
    cls_name:    str
    bbox:        list[int]        # [x1, y1, x2, y2]
    conf:        float
    X_fwd:       float = 0.0     # 전방 거리 (m)
    Y_lat:       float = 0.0     # 측방 거리 (m)
    is_bsd:      bool  = False   # BSD 위험 여부
    alert_level: str   = "SAFE"  # "SAFE" | "WARNING" | "DANGER"
    side:        str   = "right" # 카메라 측 ("right" | "left")


class BSDInterface:
    """
    좌·우 BSD 카메라의 CoordTransformer를 통합 관리.

    BSD 경고 레벨:
      SAFE    : 사각지대 외부
      WARNING : 사각지대 진입 전 (측방 bsd_lat_max ~ WARNING_LAT_MAX)
      DANGER  : 사각지대 내부 (측방 bsd_lat_min ~ bsd_lat_max, 후방 이내)

    Args:
        camera_config: camera_config.yaml 경로
    """

    WARNING_LAT_MAX = 4.0       # WARNING 구역 측방 최대 (m)

    def __init__(self, camera_config: str = "configs/camera_config.yaml"):
        self.transformer_right = CoordTransformer(camera_config, side="right")
        self.transformer_left  = CoordTransformer(camera_config, side="left")
        self._transformers = {
            "right": self.transformer_right,
            "left":  self.transformer_left,
        }

    # ── 메인 처리 ─────────────────────────────────────────────────────────

    def process(
        self,
        detections: list[dict],
        side: str = "right",
        tracked_ids: list[int] | None = None,
        img_w: int = 1280,
        img_h: int = 720,
    ) -> tuple[list[TrackedObject], bool]:
        """
        한 카메라의 탐지 결과를 처리하여 BSD 경고 여부 반환.

        Args:
            detections : SGLDetInference.detect() 반환값
            side       : "right" | "left"  (어느 카메라인지)
            tracked_ids: SORT 트래커가 할당한 track ID 리스트
                         (None이면 탐지 순서 번호 사용)
            img_w, img_h: 원본 이미지 크기
        Returns:
            (tracked_objects, any_danger)
        """
        if tracked_ids is None:
            tracked_ids = list(range(len(detections)))

        transformer = self._transformers.get(side, self.transformer_right)
        results = []
        any_danger = False

        for det, tid in zip(detections, tracked_ids):
            # 픽셀 → 차량 좌표계 지면점
            X_fwd, Y_lat = transformer.bbox_to_ground(
                det["cx_norm"], det["cy_norm"],
                det["w_norm"],  det["h_norm"],
                img_w, img_h,
            )

            alert    = self._get_alert_level(transformer, X_fwd, Y_lat)
            is_danger = (alert == "DANGER")
            if is_danger:
                any_danger = True

            results.append(TrackedObject(
                track_id    = tid,
                cls_id      = det["cls_id"],
                cls_name    = det["cls_name"],
                bbox        = det["bbox"],
                conf        = det["conf"],
                X_fwd       = X_fwd,
                Y_lat       = Y_lat,
                is_bsd      = is_danger,
                alert_level = alert,
                side        = side,
            ))

        return results, any_danger

    def process_both(
        self,
        detections_right: list[dict],
        detections_left:  list[dict],
        tracked_ids_right: list[int] | None = None,
        tracked_ids_left:  list[int] | None = None,
        img_w: int = 1280,
        img_h: int = 720,
    ) -> tuple[list[TrackedObject], bool]:
        """
        좌·우 두 카메라 탐지 결과를 한 번에 처리.

        Returns:
            (all_tracked_objects, any_danger)
        """
        right_objs, right_danger = self.process(
            detections_right, "right", tracked_ids_right, img_w, img_h,
        )
        left_objs, left_danger = self.process(
            detections_left, "left", tracked_ids_left, img_w, img_h,
        )
        return right_objs + left_objs, right_danger or left_danger

    # ── 경고 레벨 판단 ────────────────────────────────────────────────────

    def _get_alert_level(
        self,
        transformer: CoordTransformer,
        X_fwd: float,
        Y_lat: float,
    ) -> str:
        """BSD 경고 레벨 반환."""
        if X_fwd == float("inf") or Y_lat == float("inf"):
            return "SAFE"

        lat_abs = abs(Y_lat)
        in_long = -transformer.bsd_rear_max <= X_fwd <= transformer.bsd_fwd_max

        # DANGER: 사각지대 내부
        if transformer.bsd_lat_min <= lat_abs <= transformer.bsd_lat_max and in_long:
            return "DANGER"

        # WARNING: 접근 구역 (사각지대 바깥쪽 완충)
        if transformer.bsd_lat_max < lat_abs <= self.WARNING_LAT_MAX and in_long:
            return "WARNING"

        return "SAFE"

    # ── SORT 포맷 변환 헬퍼 ───────────────────────────────────────────────

    @staticmethod
    def format_sort_input(detections: list[dict]) -> np.ndarray:
        """
        SORT 트래커 입력 포맷으로 변환.

        Returns:
            (N, 5) float32 array: [x1, y1, x2, y2, conf]
        """
        if not detections:
            return np.empty((0, 5), dtype=np.float32)

        rows = [
            [*det["bbox"], det["conf"]]
            for det in detections
        ]
        return np.array(rows, dtype=np.float32)

    @staticmethod
    def parse_sort_output(
        sort_output: np.ndarray,
        detections: list[dict],
    ) -> tuple[list[dict], list[int]]:
        """
        SORT 출력 파싱 → track ID 리스트 반환.

        SORT 출력: (N, 5) [x1, y1, x2, y2, track_id]
        """
        if sort_output is None or len(sort_output) == 0:
            return detections, list(range(len(detections)))

        track_ids = [int(row[4]) for row in sort_output]
        return detections, track_ids
