"""
SORT 트래커 래퍼 (간이 IoU 기반 트래커 포함)
---------------------------------------------
SGLDet 탐지 결과를 입력받아 객체 ID를 유지하며 추적.

설치 (선택): pip install filterpy lap
            → 그러면 정식 SORT 라이브러리(scikit-image 기반) 사용 가능.
            → 설치 안 되어 있으면 IoU greedy 매칭 fallback 사용.
"""

import numpy as np


def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """두 bbox(xyxy)의 IoU 계산."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


class SORTTracker:
    """
    SORT 트래커 래퍼.

    사용:
      tracker = SORTTracker(max_age=3, min_hits=1, iou_threshold=0.3)
      tracked = tracker.update(detections_np)
      # detections_np: (N, 5) [x1, y1, x2, y2, conf]
      # 반환: (M, 5) [x1, y1, x2, y2, track_id]
    """

    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._next_id = 1
        self._tracks: list = []   # [{id, bbox, age, hits}]

        try:
            from sort import Sort
            self._sort = Sort(max_age=max_age, min_hits=min_hits,
                              iou_threshold=iou_threshold)
            self._use_lib = True
            print("[SORTTracker] 정식 SORT 라이브러리 사용")
        except ImportError:
            self._sort = None
            self._use_lib = False
            print("[SORTTracker] sort 라이브러리 미설치 → IoU greedy 매칭 사용")

    def update(self, dets: np.ndarray) -> np.ndarray:
        """
        Args:
            dets: (N, 5) [x1, y1, x2, y2, conf]
        Returns:
            (M, 5) [x1, y1, x2, y2, track_id]
        """
        if self._use_lib:
            return self._sort.update(dets)
        return self._iou_match_update(dets)

    # -----------------------------------------------------------------------
    # IoU greedy 매칭 fallback (SORT 라이브러리 없을 때)
    # -----------------------------------------------------------------------
    def _iou_match_update(self, dets: np.ndarray) -> np.ndarray:
        if len(dets) == 0:
            # 매칭 안 된 트랙은 age 증가
            self._tracks = [
                {**t, "age": t["age"] + 1}
                for t in self._tracks if t["age"] + 1 <= self.max_age
            ]
            return np.empty((0, 5))

        det_bboxes = dets[:, :4]
        unmatched_dets = list(range(len(dets)))
        unmatched_tracks = list(range(len(self._tracks)))
        matches = []

        # IoU 행렬 계산 후 greedy 매칭
        if self._tracks:
            iou_mat = np.zeros((len(self._tracks), len(dets)))
            for ti, tr in enumerate(self._tracks):
                for di in range(len(dets)):
                    iou_mat[ti, di] = _iou(tr["bbox"], det_bboxes[di])

            # 가장 큰 IoU부터 매칭
            while True:
                if not unmatched_tracks or not unmatched_dets:
                    break
                ti_best, di_best, best_iou = -1, -1, -1.0
                for ti in unmatched_tracks:
                    for di in unmatched_dets:
                        if iou_mat[ti, di] > best_iou:
                            best_iou = iou_mat[ti, di]
                            ti_best, di_best = ti, di
                if best_iou < self.iou_threshold:
                    break
                matches.append((ti_best, di_best))
                unmatched_tracks.remove(ti_best)
                unmatched_dets.remove(di_best)

        # 매칭된 트랙 업데이트
        for ti, di in matches:
            self._tracks[ti]["bbox"] = det_bboxes[di]
            self._tracks[ti]["age"] = 0
            self._tracks[ti]["hits"] += 1

        # 매칭 안 된 트랙은 age 증가
        for ti in unmatched_tracks:
            self._tracks[ti]["age"] += 1

        # 새로운 detection은 신규 트랙으로 등록
        for di in unmatched_dets:
            self._tracks.append({
                "id":   self._next_id,
                "bbox": det_bboxes[di].copy(),
                "age":  0,
                "hits": 1,
            })
            self._next_id += 1

        # 너무 오래된 트랙 제거
        self._tracks = [t for t in self._tracks if t["age"] <= self.max_age]

        # 출력: confirmed 트랙만 (min_hits 이상)
        result = []
        for t in self._tracks:
            if t["hits"] >= self.min_hits and t["age"] == 0:
                x1, y1, x2, y2 = t["bbox"]
                result.append([x1, y1, x2, y2, t["id"]])

        return np.array(result) if result else np.empty((0, 5))

    @staticmethod
    def get_track_ids(sort_output: np.ndarray) -> list:
        """SORT 출력에서 track ID 리스트 추출."""
        if len(sort_output) == 0:
            return []
        return [int(row[4]) for row in sort_output]
