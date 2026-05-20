# BSDsystem — 야간 사각지대 감지 (SGLDet + Fisheye)

MORAI 시뮬레이터 기반 야간/황혼 BSD(Blind Spot Detection) 시스템.
SGLDet(Self-Guided Low-Light Detection, ICLR 2026) 프레임워크와 YOLOv8m 백본을 결합해
저조도 환경에서 차량·보행자를 검출한다.

## 카메라 사양 (MORAI Fisheye)

- 모델: **Equidistant Fisheye** (OpenCV `cv2.fisheye`, r = f·θ)
- 해상도: 1280 × 720
- 수평 FOV: **179°**
- Intrinsic: fx = fy = **409.73**, cx = 640, cy = 360
- 왜곡 계수: [0, 0, 0, 0]  (MORAI는 이상적 모델)
- 장착 (Camera-1, 우측 BSD): x=2.15m, y=0.9m, z=0.55m, yaw=90°, pitch=15°
- 장착 (Camera-2, 좌측 BSD): x=2.15m, y=-0.9m, z=0.55m, yaw=270°, pitch=15°

## 클래스 (2개)

| ID | 클래스 | MORAI 출처 |
|----|--------|------------|
| 0 | vehicle | `npc_list` (NPC 차량 전체 — 승용/트럭/버스 모두) |
| 1 | pedestrian | `pedestrian_list` |

## 구조

```
BSDsystem/
├── configs/                # 카메라 캘리브레이션 + 학습 설정
│   ├── camera_config.yaml  # Fisheye intrinsic/extrinsic
│   └── sgldet_config.yaml  # 모델/학습 하이퍼파라미터 (num_classes=2)
├── src/
│   ├── datasets/           # MORAI YOLO 데이터셋 로더
│   ├── inference/          # SGLDet 추론 + BSD 판단 로직
│   └── preprocessing/      # Fisheye 보정 + 좌표 변환 (cv2.fisheye)
├── models/                 # SGLDetYOLO, SORT Tracker
├── ros2_ws/                # MORAI ROS2 연동 패키지 (추론용, 추후 빌드)
├── collect_data.py         # MORAI GT 기반 데이터 수집 (Linux/ROS2)
├── augment_data.py         # 야간 특화 데이터 증강
├── train.py                # SGLDet 학습
└── main.py                 # 학습 / 실시간 추론 진입점
```

## 환경 설정

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux
source .venv/bin/activate

pip install ultralytics opencv-python torch pyyaml numpy
# ROS2 코드 사용 시: rclpy, cv_bridge, morai_msgs 별도 (시스템 ROS2 의존)
```

## 실행 순서

### 1. 데이터 수집 (Linux + MORAI)

`/Object_topic` (GT) + `/Ego_topic` + Instance Mask 토픽을 구독해
YOLO 라벨을 직접 생성한다. YOLOv8 자동라벨링 사용 안 함 (야간 오탐 회피).

```bash
python3 collect_data.py --condition night
python3 collect_data.py --condition dusk --auto --interval 0.5
```

### 2. 데이터 증강 (Windows)

야간 특화 증강 10종 (감마/노이즈/헤드라이트/색온도 등).
좌우 반전은 BSD 측면 카메라 특성상 기본 OFF.

```bash
python augment_data.py --condition dusk  --n-aug 5 --yes
python augment_data.py --condition night --n-aug 5 --yes
```

### 3. 학습

```bash
python main.py --mode train
python main.py --mode train --epochs 200 --batch 16
python main.py --mode train --pretrain        # SCI/SDAP 사전학습 포함
```

### 4. 추론

```bash
python main.py --mode run --source 0          # 카메라
python main.py --mode run --source video.mp4  # 영상
```

## 데이터 구조 (YOLO 형식)

```
data/morai/
├── dusk/
│   ├── images/  *.jpg
│   └── labels/  *.txt   (class cx cy w h, 정규화 좌표)
└── night/
    ├── images/
    └── labels/
```

`data/` 폴더는 `.gitignore` 처리 (용량).

## 좌표계 약속 (MORAI 차량 프레임)

- X : 전방(+) / 후방(-)
- Y : 우측(+) / 좌측(-)
- Z : 상향(+) / 하향(-)
- Yaw : 위에서 시계방향 (0=전방, 90=우측, 270=좌측)

⚠️ `(X=fwd, Y=right, Z=up)` 은 **left-handed** 좌표계 → R_c2v 의 det = -1.
`cv2.Rodrigues` / `cv2.solvePnP` 등 SO(3) 전용 API 에 직접 넣지 말 것.
이 프로젝트의 변환은 모두 직접 수식으로 처리한다
(`collect_data.world_to_pixel`, `coord_transform.pixel_to_ground`).
