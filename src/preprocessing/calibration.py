"""
카메라 캘리브레이션 모듈 (Part B: 구동 단계)
----------------------------------------------
MORAI 시뮬레이션 카메라 파라미터를 이용해
실제 입력 이미지의 왜곡을 보정.

실차 적용 시: OpenCV 체커보드 캘리브레이션으로 파라미터 교체.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path


class CameraCalibration:
    """
    카메라 내부 파라미터 기반 왜곡 보정.

    Args:
        config_path: camera_config.yaml 경로
    """

    def __init__(self, config_path: str = "configs/camera_config.yaml"):
        cfg = self._load_config(config_path)["camera"]

        self.fx = cfg["fx"]
        self.fy = cfg["fy"]
        self.cx = cfg["cx"]
        self.cy = cfg["cy"]
        self.img_w = cfg["image_width"]
        self.img_h = cfg["image_height"]

        # 카메라 행렬 K
        self.K = np.array([
            [self.fx,      0, self.cx],
            [     0,  self.fy, self.cy],
            [     0,       0,      1],
        ], dtype=np.float64)

        # 왜곡 계수
        self.dist = np.array(cfg["dist_coeffs"], dtype=np.float64)

        # 보정 맵 사전 계산 (매 프레임 반복 계산 방지)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, self.K,
            (self.img_w, self.img_h), cv2.CV_32FC1,
        )

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def undistort(self, img: np.ndarray) -> np.ndarray:
        """
        이미지 왜곡 보정.

        Args:
            img: (H, W, 3) BGR 이미지 (OpenCV 기본 포맷)
        Returns:
            보정된 이미지 (H, W, 3)
        """
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

    def project_to_image(self, points_3d: np.ndarray) -> np.ndarray:
        """
        3D 월드 좌표 → 이미지 픽셀 좌표 투영.

        Args:
            points_3d: (N, 3) [X, Y, Z] 차량 좌표계
        Returns:
            pixels: (N, 2) [u, v] 픽셀 좌표
        """
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        pixels, _ = cv2.projectPoints(
            points_3d.astype(np.float64),
            rvec, tvec, self.K, self.dist,
        )
        return pixels.reshape(-1, 2)


# ── 사용 예시 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import cv2

    calib = CameraCalibration("configs/camera_config.yaml")

    # 테스트 이미지 왜곡 보정
    dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
    corrected = calib.undistort(dummy)
    print("보정 완료:", corrected.shape)
