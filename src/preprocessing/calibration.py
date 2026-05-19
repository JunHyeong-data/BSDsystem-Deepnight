"""
카메라 캘리브레이션 모듈 (MORAI Fisheye Equidistant)
------------------------------------------------------
MORAI 시뮬레이션의 fisheye 카메라 파라미터를 이용해
- (선택) 입력 이미지의 어안 왜곡을 보정한 평면 영상 생성
- 3D 차량 좌표 → fisheye 이미지 픽셀 투영

카메라 모델 : OpenCV cv2.fisheye  (r = f · θ, k1..k4)
MORAI 가정 : distortion 0 의 이상적 equidistant.

학습/추론 입력을 fisheye 원본 그대로 쓸지 undistort 평면으로 쓸지는
별도 정책. 본 모듈은 둘 다 지원하되, 기본은 원본 fisheye 그대로 유지.
"""

import cv2
import numpy as np
import yaml


class CameraCalibration:
    """
    Fisheye 카메라 내부 파라미터 기반 처리 유틸.

    Args:
        config_path: camera_config.yaml 경로
        balance:     undistort 새 카메라 행렬 추정 시 사용 (0=잘림 최소, 1=FOV 보존)
                     실제 학습/추론에서 undistort를 안 쓰면 무시됨.
    """

    def __init__(
        self,
        config_path: str = "configs/camera_config.yaml",
        balance: float = 0.0,
    ):
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

        # Fisheye 왜곡 계수 [k1, k2, k3, k4] — shape (4, 1)
        self.D = np.array(cfg["dist_coeffs_fisheye"], dtype=np.float64).reshape(4, 1)

        # Undistort 출력용 새 카메라 행렬 (FOV 보존 정도 = balance)
        # MORAI는 D=0이라 사실상 K와 동일하지만, 실측 fisheye엔 유용.
        self.K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, (self.img_w, self.img_h),
            np.eye(3), balance=balance,
        )

        # Fisheye 보정 맵 사전 계산 (매 프레임 반복 계산 방지)
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.K_new,
            (self.img_w, self.img_h), cv2.CV_16SC2,
        )

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def undistort(self, img: np.ndarray) -> np.ndarray:
        """
        Fisheye → 평면(undistorted) 이미지 변환.

        Args:
            img: (H, W, 3) BGR fisheye 원본
        Returns:
            보정된 이미지 (H, W, 3) — K_new 기준 평면 영상
        """
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

    def project_to_image(self, points_3d: np.ndarray) -> np.ndarray:
        """
        3D 카메라 좌표 → fisheye 이미지 픽셀 투영 (Equidistant 모델).

        Args:
            points_3d: (N, 3) [X, Y, Z] 카메라 좌표계
                       (X=오른쪽, Y=아래, Z=광축 전방)
        Returns:
            pixels: (N, 2) [u, v] fisheye 이미지 픽셀 좌표
        """
        pts = points_3d.astype(np.float64).reshape(-1, 1, 3)
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        pixels, _ = cv2.fisheye.projectPoints(pts, rvec, tvec, self.K, self.D)
        return pixels.reshape(-1, 2)


# ── 사용 예시 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    calib = CameraCalibration("configs/camera_config.yaml")
    print(f"K =\n{calib.K}")
    print(f"D = {calib.D.ravel()}")
    print(f"K_new =\n{calib.K_new}")

    # 광축 위 점은 (cx, cy)에 떨어져야 함
    test_pts = np.array([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0]], dtype=np.float64)
    print(f"project {test_pts.tolist()} -> {calib.project_to_image(test_pts).tolist()}")
