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

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

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
    is_bsd:      bool  = False   # BSD 위험 여부 (alert_level == "DANGER")
    alert_level: str   = "SAFE"  # "SAFE" | "WARNING" | "DANGER"
    side:        str   = "right" # 카메라 측 ("right" | "left")
    approach_velocity: float = 0.0  # dX_fwd/dt (m/s, 양수=접근)


class BSDInterface:
    """
    좌·우 BSD 카메라의 CoordTransformer를 통합 관리.

    BSD 경고 레벨 (velocity-aware 3-stage):
      SAFE    : 사각지대 외부 (그리고 인접 WARNING 구역도 아님)
      WARNING : (a) 사각지대 인접 (측방 bsd_lat_max ~ WARNING_LAT_MAX), 또는
                (b) 사각지대 내부지만 접근 중 아님 / 히스토리 부족
      DANGER  : 사각지대 내부 + 접근 속도 dX_fwd/dt > approach_threshold

    timestamp_s 인자가 process() 에 들어오면 그 값(초)으로 dX/dt 계산.
    None 이면 time.perf_counter() — 실시간 카메라 / 즉시추론용. 저장 영상
    (오프라인) 처리 시 cap.get(CAP_PROP_POS_MSEC)/1000 등을 명시적으로 넘길 것.

    Args:
        camera_config:      camera_config.yaml 경로
        approach_threshold: dX_fwd/dt 임계값 (m/s). 이 값 초과면 DANGER.
        history_len:        track 별 (X, t) 히스토리 최대 길이.
        stale_after_s:      이 시간 이상 안 보인 track 히스토리 자동 삭제.
    """

    WARNING_LAT_MAX = 4.0       # WARNING 구역 측방 최대 (m)

    def __init__(
        self,
        camera_config: str = "configs/camera_config.yaml",
        approach_threshold: float = 0.3,
        history_len: int = 5,
        stale_after_s: float = 1.0,
    ):
        self.transformer_right = CoordTransformer(camera_config, side="right")
        self.transformer_left  = CoordTransformer(camera_config, side="left")
        self._transformers = {
            "right": self.transformer_right,
            "left":  self.transformer_left,
        }

        # ── velocity 기반 3-stage 상태 ──
        self.approach_threshold = approach_threshold
        self._history_len = history_len
        self._stale_after_s = stale_after_s
        # (side, track_id) → deque[(X_fwd, ts_s)]
        # SORT 트래커가 좌·우 별도이므로 side 도 키에 포함.
        self._track_history: dict[tuple[str, int], deque] = {}

    # ── 메인 처리 ─────────────────────────────────────────────────────────

    def process(
        self,
        detections: list[dict],
        side: str = "right",
        tracked_ids: list[int] | None = None,
        img_w: int = 1280,
        img_h: int = 720,
        timestamp_s: float | None = None,
    ) -> tuple[list[TrackedObject], bool]:
        """
        한 카메라의 탐지 결과를 처리하여 BSD 경고 여부 반환.

        Args:
            detections : SGLDetInference.detect() 반환값
            side       : "right" | "left"  (어느 카메라인지)
            tracked_ids: SORT 트래커가 할당한 track ID 리스트
                         (None이면 탐지 순서 번호 사용)
            img_w, img_h: 원본 이미지 크기
            timestamp_s : 프레임 타임스탬프(초). None 이면 time.perf_counter().
                          저장 영상 처리 시엔 CAP_PROP_POS_MSEC/1000 등 실시간
                          기준값을 명시적으로 넘기는 게 정확함.
        Returns:
            (tracked_objects, any_danger)
        """
        if tracked_ids is None:
            tracked_ids = list(range(len(detections)))
        if timestamp_s is None:
            timestamp_s = time.perf_counter()

        transformer = self._transformers.get(side, self.transformer_right)
        results = []
        any_danger = False
        seen_keys: set[tuple[str, int]] = set()

        for det, tid in zip(detections, tracked_ids):
            # 픽셀 → 차량 좌표계 지면점
            X_fwd, Y_lat = transformer.bbox_to_ground(
                det["cx_norm"], det["cy_norm"],
                det["w_norm"],  det["h_norm"],
                img_w, img_h,
            )

            # 히스토리 갱신 (zone 안팎 무관하게 누적 → 갓 진입 시에도 dt 확보)
            key = (side, tid)
            hist = self._track_history.setdefault(
                key, deque(maxlen=self._history_len)
            )
            hist.append((X_fwd, timestamp_s))
            seen_keys.add(key)

            zone_alert = self._get_zone_alert(transformer, X_fwd, Y_lat)
            alert, dX_dt = self._refine_alert(zone_alert, hist)

            is_danger = (alert == "DANGER")
            if is_danger:
                any_danger = True

            results.append(TrackedObject(
                track_id          = tid,
                cls_id            = det["cls_id"],
                cls_name          = det["cls_name"],
                bbox              = det["bbox"],
                conf              = det["conf"],
                X_fwd             = X_fwd,
                Y_lat             = Y_lat,
                is_bsd            = is_danger,
                alert_level       = alert,
                side              = side,
                approach_velocity = dX_dt,
            ))

        # 안 보이는 지 오래된 track 히스토리 청소 (장시간 실행 시 메모리 누수 방지)
        self._prune_stale(timestamp_s, seen_keys, side)

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

    # ── 경고 레벨 판단 (zone + velocity 2단 결합) ──────────────────────────

    def _get_zone_alert(
        self,
        transformer: CoordTransformer,
        X_fwd: float,
        Y_lat: float,
    ) -> str:
        """
        공간만 보고 zone-level 판정.
        반환값:
          "IN_ZONE"  사각지대 내부      (velocity 로 DANGER/WARNING 재판정)
          "WARNING"  사각지대 인접 완충 구역 (velocity 무관 WARNING 확정)
          "SAFE"     그 외
        """
        if X_fwd == float("inf") or Y_lat == float("inf"):
            return "SAFE"

        lat_abs = abs(Y_lat)
        in_long = -transformer.bsd_rear_max <= X_fwd <= transformer.bsd_fwd_max

        if transformer.bsd_lat_min <= lat_abs <= transformer.bsd_lat_max and in_long:
            return "IN_ZONE"
        if transformer.bsd_lat_max < lat_abs <= self.WARNING_LAT_MAX and in_long:
            return "WARNING"
        return "SAFE"

    def _refine_alert(
        self,
        zone_alert: str,
        hist: deque,
    ) -> tuple[str, float]:
        """
        zone 판정 + 히스토리 → 최종 alert + dX/dt.

        규칙:
          - SAFE / WARNING (인접) 은 velocity 무관하게 그대로 통과 (dX_dt = 0).
          - IN_ZONE 은:
              history < 2 또는 dt ≤ 0  → "WARNING" (보수적 기본값)
              dX/dt > approach_threshold → "DANGER"
              그 외                       → "WARNING"
        Returns:
            (alert_level, dX_dt)
        """
        if zone_alert != "IN_ZONE":
            return zone_alert, 0.0

        if len(hist) < 2:
            return "WARNING", 0.0

        X_old, t_old = hist[0]
        X_new, t_new = hist[-1]
        dt = t_new - t_old
        if dt <= 0:
            return "WARNING", 0.0

        dX_dt = (X_new - X_old) / dt
        alert = "DANGER" if dX_dt > self.approach_threshold else "WARNING"
        return alert, dX_dt

    def _prune_stale(
        self,
        now_s: float,
        seen_keys: set[tuple[str, int]],
        side: str,
    ) -> None:
        """이번 frame 에서 안 보였고 stale_after_s 초 이상 안 본 track 히스토리 제거."""
        to_drop = [
            k for k, h in self._track_history.items()
            if k[0] == side and k not in seen_keys
            and h and (now_s - h[-1][1]) > self._stale_after_s
        ]
        for k in to_drop:
            del self._track_history[k]

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
