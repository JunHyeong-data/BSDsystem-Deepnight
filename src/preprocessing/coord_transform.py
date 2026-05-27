"""
이미지 좌표 → 실제 공간 좌표 변환 (Part B: 구동 단계)
------------------------------------------------------
MORAI Fisheye 카메라 (Equidistant 모델) 의 외부·내부 파라미터를 이용해
탐지된 bbox 접지점을 차량 좌표계(m)로 변환하여
BSD 사각지대 침범 여부를 판단.

카메라 2대 구성:
  Camera-1 (camera_right) : 우측 BSD  — yaw=90°,  pitch=15°
  Camera-2 (camera_left)  : 좌측 BSD  — yaw=270°, pitch=15°

좌표계 약속 (MORAI 차량 프레임):
  X  : 전방(+)
  Y  : 우측(+)
  Z  : 상향(+)
  Yaw: 위에서 시계방향 (0=전방, 90=우측, 270=좌측)

수학적 배경 (Fisheye Equidistant):
  1. 픽셀 (u,v) → 카메라 광선 단위벡터 d_cam :
       du, dv  = u - cx, v - cy
       r       = sqrt(du² + dv²)
       θ       = r / fx          (등거리 역투영, k1..k4 = 0)
       d_cam   = (sin θ · du/r,  sin θ · dv/r,  cos θ)
       ※ 핀홀 정규화 좌표 (du/fx, dv/fy, 1)와 다름.
         fisheye에선 큰 θ에서 위 둘이 발산함.
  2. 카메라→차량 회전행렬 R_c2v 로 방향 변환: d_veh = R_c2v @ d_cam
  3. 카메라 장착 위치 p = (mount_x, mount_y, mount_z)
  4. 지면 교차 (Z_veh=0): t = -p[2] / d_veh[2]
  5. 교차점: (X_fwd, Y_lat) = (p[0]+t·d_veh[0], p[1]+t·d_veh[1])
"""

from __future__ import annotations

import numpy as np
import yaml


# ── 회전행렬 유틸 ────────────────────────────────────────────────────────────

def _Rz(yaw_deg: float) -> np.ndarray:
    """MORAI 시계방향 Yaw (Z축) 회전행렬."""
    θ = np.deg2rad(yaw_deg)
    return np.array([
        [ np.cos(θ), -np.sin(θ), 0],
        [ np.sin(θ),  np.cos(θ), 0],
        [ 0,          0,         1],
    ])


def _Ry(pitch_deg: float) -> np.ndarray:
    """Pitch (Y축) 회전행렬. 양수 = 앞쪽 하향."""
    θ = np.deg2rad(pitch_deg)
    return np.array([
        [ np.cos(θ), 0, np.sin(θ)],
        [ 0,         1, 0        ],
        [-np.sin(θ), 0, np.cos(θ)],
    ])


def _Rx(roll_deg: float) -> np.ndarray:
    """Roll (X축) 회전행렬."""
    θ = np.deg2rad(roll_deg)
    return np.array([
        [1, 0,          0         ],
        [0, np.cos(θ), -np.sin(θ)],
        [0, np.sin(θ),  np.cos(θ)],
    ])


# MORAI yaw=0, pitch=0, roll=0 일 때 카메라가 차량 전방을 향하는
# 카메라 프레임(Z=광축, X=이미지우, Y=이미지하) → 차량 프레임 기저 회전.
#   Z_cam → X_veh (전방)
#   X_cam → Y_veh (우측)
#   Y_cam → -Z_veh (하향)
#
# ⚠️ 주의: MORAI 차량 프레임 (X=전방, Y=우측, Z=상향) 은 left-handed.
#   따라서 이 행렬과 R_c2v 의 determinant 는 -1 (reflection 포함).
#   cv2.Rodrigues / cv2.solvePnP 등 SO(3) 전용 API 에 R_c2v 를 그대로 넣으면
#   round-trip 이 깨진다. 직접 수식(이 파일의 pixel_to_ground 등)을 쓸 것.
_R_CAM_TO_BODY_BASE = np.array([
    [0,  0, 1],
    [1,  0, 0],
    [0, -1, 0],
], dtype=float)


def build_R_cam_to_veh(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    MORAI 센서 외부 파라미터 → 카메라 프레임→차량 프레임 회전행렬.

    적용 순서 (MORAI intrinsic): Yaw → Pitch → Roll

    Args:
        yaw_deg  : MORAI yaw  (시계방향, 0=전방, 90=우, 270=좌)
        pitch_deg: MORAI pitch (양수=앞쪽 하향)
        roll_deg : MORAI roll

    Returns:
        R_c2v (3×3): camera frame → vehicle frame
    """
    R_body_to_veh = _Rz(yaw_deg) @ _Ry(pitch_deg) @ _Rx(roll_deg)
    return R_body_to_veh @ _R_CAM_TO_BODY_BASE


# ── 메인 클래스 ─────────────────────────────────────────────────────────────

class CoordTransformer:
    """
    단안 카메라 역투영: 픽셀 좌표 → 차량 좌표계 지면점.

    두 대의 BSD 카메라를 지원:
      - side="right" : Camera-1 (우측, yaw=90°)
      - side="left"  : Camera-2 (좌측, yaw=270°)

    Args:
        config_path: camera_config.yaml 경로
        side       : "right" | "left"
    """

    def __init__(
        self,
        config_path: str = "configs/camera_config.yaml",
        side: str = "right",
    ):
        if side not in ("right", "left"):
            raise ValueError(f"side must be 'right' or 'left', got: {side!r}")
        self.side = side

        with open(config_path, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        # ── 내부 파라미터 ─────────────────────────────────────
        intr = yaml_data["intrinsic"]
        self.fx     = float(intr["fx"])
        self.fy     = float(intr["fy"])
        self.cx     = float(intr["cx"])
        self.cy     = float(intr["cy"])
        self.img_w  = int(intr["image_width"])
        self.img_h  = int(intr["image_height"])

        # ── 외부 파라미터 ─────────────────────────────────────
        cam_key = "camera_right" if side == "right" else "camera_left"
        ext = yaml_data[cam_key]

        self.mount_x = float(ext["mount_x"])
        self.mount_y = float(ext["mount_y"])
        self.mount_z = float(ext["mount_z"])
        # 카메라 장착 위치 벡터 (차량 프레임)
        self.p_cam   = np.array([self.mount_x, self.mount_y, self.mount_z])

        # 카메라 → 차량 회전행렬
        self.R_c2v = build_R_cam_to_veh(
            yaw_deg   = float(ext["yaw_deg"]),
            pitch_deg = float(ext["pitch_deg"]),
            roll_deg  = float(ext["roll_deg"]),
        )

        # ── BSD 판단 기준 ──────────────────────────────────────
        bsd = yaml_data["bsd_zone"]
        self.bsd_lat_min   = float(bsd["lateral_min"])    # m
        self.bsd_lat_max   = float(bsd["lateral_max"])    # m
        self.bsd_fwd_max   = float(bsd["forward_max"])    # m  (전방 한계)
        self.bsd_rear_max  = float(bsd["rear_max"])       # m  (후방 한계)

        # ── 광축 방향 (디버그용) ───────────────────────────────
        # Z_cam = (0,0,1) → vehicle frame
        self._optical_axis_veh = self.R_c2v @ np.array([0., 0., 1.])

    # ── 핵심 변환 ─────────────────────────────────────────────────────────

    def pixel_to_ground(self, u: float, v: float) -> tuple[float, float]:
        """
        Fisheye 픽셀 좌표 (u, v) → 차량 좌표계 지면점 (X_fwd, Y_lat).

        Equidistant 역투영 (OpenCV cv2.fisheye, k1..k4 = 0 가정):
            r = sqrt((u-cx)² + (v-cy)²)
            θ = r / fx
            d_cam = (sin θ · (u-cx)/r,  sin θ · (v-cy)/r,  cos θ)
        지면: Z_veh = 0 평면.

        Args:
            u: 픽셀 x (좌→우)
            v: 픽셀 y (위→아래)

        Returns:
            (X_fwd, Y_lat): 전방 거리(m, 양수=전방), 측방 거리(m, 양수=우측)
            지면 교차 불가 / FOV 밖 / 광선 후방일 때 (inf, inf) 반환.
        """
        du = float(u) - self.cx
        dv = float(v) - self.cy
        r_pix = np.hypot(du, dv)

        if r_pix < 1e-9:
            # 광축 위의 픽셀 — 광선은 정확히 광축(Z_cam) 방향
            d_cam = np.array([0.0, 0.0, 1.0])
        else:
            # Equidistant 역투영 (k=0): θ = r / fx
            theta = r_pix / self.fx
            # θ ≥ 90° 면 카메라 측면/뒤편 — 지면에 닿을 일 거의 없음
            if theta >= np.pi / 2.0 - 1e-3:
                return float("inf"), float("inf")
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            inv_r = 1.0 / r_pix
            d_cam = np.array([sin_t * du * inv_r,
                              sin_t * dv * inv_r,
                              cos_t])

        # 차량 프레임 방향
        d_veh = self.R_c2v @ d_cam          # (3,)

        # 지면(Z_veh=0) 교차: p + t*d_veh 에서 Z=0
        if abs(d_veh[2]) < 1e-6:
            # 광선이 지면에 평행 → 교차 없음
            return float("inf"), float("inf")

        t = -self.p_cam[2] / d_veh[2]

        if t < 0:
            # 교차가 카메라 뒤쪽 → 유효하지 않음
            return float("inf"), float("inf")

        X_fwd = self.p_cam[0] + t * d_veh[0]
        Y_lat = self.p_cam[1] + t * d_veh[1]

        return float(X_fwd), float(Y_lat)

    def bbox_to_ground(
        self,
        cx_norm: float,
        cy_norm: float,
        w_norm:  float,
        h_norm:  float,
        img_w:   int | None = None,
        img_h:   int | None = None,
    ) -> tuple[float, float]:
        """
        정규화 YOLO bbox → 차량 좌표계 지면점.

        bbox 하단 중앙점을 객체 접지점으로 사용.

        Args:
            cx_norm, cy_norm, w_norm, h_norm: YOLO 정규화 좌표 (0~1)
            img_w, img_h: 이미지 해상도 (None이면 config 값 사용)

        Returns:
            (X_fwd, Y_lat) in m
        """
        W = img_w or self.img_w
        H = img_h or self.img_h
        u = cx_norm * W
        v = (cy_norm + h_norm / 2.0) * H   # 하단 중앙
        return self.pixel_to_ground(u, v)

    # ── BSD 판단 ──────────────────────────────────────────────────────────

    def is_in_bsd_zone(self, X_fwd: float, Y_lat: float) -> bool:
        """
        BSD 사각지대 침범 여부 판단.

        사각지대 정의 (MORAI 시뮬레이션 기준):
          - 측방 : |Y_lat| ∈ [lateral_min, lateral_max]
                   (차체 양 옆 0.5 m ~ 2.5 m)
          - 종방향: X_fwd ∈ (-rear_max, +forward_max)
                   (B-필러 기준 전후 6 m ~ 3 m)

        Args:
            X_fwd: 전방 거리 (m, 양수=전방)
            Y_lat: 측방 거리 (m, 양수=우측)

        Returns:
            True if 위험 영역
        """
        if X_fwd == float("inf") or Y_lat == float("inf"):
            return False

        in_lateral = self.bsd_lat_min <= abs(Y_lat) <= self.bsd_lat_max
        in_long    = -self.bsd_rear_max <= X_fwd <= self.bsd_fwd_max

        return in_lateral and in_long

    def get_risk_level(self, X_fwd: float, Y_lat: float) -> str:
        """
        거리 기반 위험 단계 반환.

        Returns:
            "SAFE" | "WARNING" | "DANGER"
        """
        if not self.is_in_bsd_zone(X_fwd, Y_lat):
            return "SAFE"

        # 측방 거리 기반 단계 (0.5 m 이내 = 매우 위험)
        lat_margin = abs(Y_lat) - self.bsd_lat_min
        if lat_margin < 0.5:
            return "DANGER"
        return "WARNING"

    # ── 디버그 유틸 ────────────────────────────────────────────────────────

    def print_camera_info(self) -> None:
        """카메라 파라미터 요약 출력."""
        print(f"\n{'=' * 50}")
        print(f"  CoordTransformer — {self.side.upper()} camera")
        print(f"{'=' * 50}")
        print(f"  내부 파라미터: fx={self.fx}, fy={self.fy}, "
              f"cx={self.cx}, cy={self.cy}")
        print(f"  해상도: {self.img_w} × {self.img_h}")
        print(f"  장착 위치: x={self.mount_x}m, y={self.mount_y}m, "
              f"z={self.mount_z}m")
        print(f"  광축 (차량 프레임): {self.R_c2v @ [0, 0, 1]}")
        print(f"  BSD 판단 기준: "
              f"측방 [{self.bsd_lat_min}, {self.bsd_lat_max}]m, "
              f"종방향 [{-self.bsd_rear_max}, {self.bsd_fwd_max}]m")
        print(f"{'=' * 50}\n")


# ── CoordBSDInterface: 두 카메라 통합 인터페이스 (coord_transform 내부 유틸) ──

class CoordBSDInterface:
    """
    좌·우 두 BSD 카메라의 CoordTransformer를 통합 관리.

    ROS2 노드 등에서 단일 인터페이스로 양쪽 카메라 결과를 처리.

    Args:
        config_path: camera_config.yaml 경로
    """

    def __init__(self, config_path: str = "configs/camera_config.yaml"):
        self.right = CoordTransformer(config_path, side="right")
        self.left  = CoordTransformer(config_path, side="left")

    def check_detections(
        self,
        detections: list[dict],
        side: str,
        img_w: int | None = None,
        img_h: int | None = None,
    ) -> tuple[list[tuple[float, float]], list[str]]:
        """
        탐지 결과 배치 처리.

        Args:
            detections: detector.py detect() 반환값
                        각 dict에 cx_norm, cy_norm, w_norm, h_norm 포함
            side: "right" | "left"
            img_w, img_h: 이미지 해상도

        Returns:
            coords   : [(X_fwd, Y_lat), ...] 차량 좌표계 지면점
            levels   : ["SAFE"/"WARNING"/"DANGER", ...] 위험 단계
        """
        transformer = self.right if side == "right" else self.left
        coords, levels = [], []

        for det in detections:
            xy = transformer.bbox_to_ground(
                det["cx_norm"], det["cy_norm"],
                det["w_norm"],  det["h_norm"],
                img_w, img_h,
            )
            coords.append(xy)
            levels.append(transformer.get_risk_level(*xy))

        return coords, levels

    def overall_level(self, levels: list[str]) -> str:
        """여러 탐지 결과 중 가장 높은 위험 단계 반환."""
        if "DANGER" in levels:
            return "DANGER"
        if "WARNING" in levels:
            return "WARNING"
        return "SAFE"


# ── 사용 예시 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== 카메라 파라미터 검증 ===")

    right_cam = CoordTransformer("configs/camera_config.yaml", side="right")
    left_cam  = CoordTransformer("configs/camera_config.yaml", side="left")

    right_cam.print_camera_info()
    left_cam.print_camera_info()

    # ── 우측 카메라: 이미지 중앙 하단 픽셀 ──────────────────
    print("[RIGHT] 이미지 중앙 하단 (640, 600) →", end=" ")
    X, Y = right_cam.pixel_to_ground(640, 600)
    print(f"전방={X:.2f}m, 측방={Y:.2f}m, "
          f"BSD={right_cam.is_in_bsd_zone(X, Y)}, "
          f"위험={right_cam.get_risk_level(X, Y)}")

    print("[RIGHT] 이미지 중앙 (640, 360) →", end=" ")
    X, Y = right_cam.pixel_to_ground(640, 360)
    print(f"전방={X:.2f}m, 측방={Y:.2f}m")

    # ── 좌측 카메라 ──────────────────────────────────────────
    print("[LEFT]  이미지 중앙 하단 (640, 600) →", end=" ")
    X, Y = left_cam.pixel_to_ground(640, 600)
    print(f"전방={X:.2f}m, 측방={Y:.2f}m, "
          f"BSD={left_cam.is_in_bsd_zone(X, Y)}, "
          f"위험={left_cam.get_risk_level(X, Y)}")

    # ── BSDInterface 통합 테스트 ──────────────────────────────
    print("\n[BSDInterface 통합 테스트]")
    bsd = CoordBSDInterface("configs/camera_config.yaml")
    fake_dets = [
        {"cx_norm": 0.5, "cy_norm": 0.5, "w_norm": 0.1, "h_norm": 0.15},
        {"cx_norm": 0.3, "cy_norm": 0.7, "w_norm": 0.08, "h_norm": 0.12},
    ]
    coords, levels = bsd.check_detections(fake_dets, side="right")
    for i, (xy, lv) in enumerate(zip(coords, levels)):
        print(f"  det[{i}]: {xy[0]:.2f}m 전방, {xy[1]:.2f}m 측방 → {lv}")
    print(f"  전체 위험 단계: {bsd.overall_level(levels)}")
