# BSDsystem — 야간 사각지대 감지 (SGLDet)

MORAI 시뮬레이터 기반 야간/황혼 BSD(Blind Spot Detection) 시스템.  
SGLDet(Self-Guided Low-Light Detection) 프레임워크와 YOLOv8을 결합해 저조도 환경에서 차량·보행자·트럭을 검출합니다.

## 구조

```
BSDsystem/
├── configs/              # 카메라 캘리브레이션 + 학습 설정
├── src/
│   ├── datasets/         # MORAI YOLO 데이터셋 로더
│   ├── inference/        # SGLDet 추론 + BSD 판단 로직
│   └── preprocessing/    # 왜곡 보정 + 좌표 변환
├── models/               # SGLDetYOLO, SORT Tracker
├── ros2_ws/              # MORAI ROS2 연동 패키지
├── collect_data.py       # MORAI 데이터 수집 (Linux/ROS2)
├── collect_raw.py        # 원시 이미지 수집 (Linux/ROS2)
├── generate_labels.py    # YOLO 라벨 생성 (Windows)
├── augment_data.py       # 야간 특화 데이터 증강
├── train.py              # SGLDet 학습
└── main.py               # 학습 / 실시간 추론 진입점
```

## 클래스

| ID | 클래스 |
|----|--------|
| 0 | car |
| 1 | pedestrian |
| 2 | truck |

## 환경 설정

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux
source .venv/bin/activate

pip install ultralytics opencv-python torch pyyaml numpy
# ROS2 코드 사용 시: rclpy, cv_bridge 별도 설치
```

## 실행 순서

### 1. 데이터 수집 (Linux + MORAI)
```bash
python collect_data.py --condition night --auto --interval 0.5
```

### 2. 데이터 증강 (Windows)
```bash
python augment_data.py --condition dusk  --n-aug 5 --yes
python augment_data.py --condition night --n-aug 5 --yes
```

### 3. 학습
```bash
python main.py --mode train
python main.py --mode train --epochs 200 --batch 16
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
│   └── labels/  *.txt  (class cx cy w h, 정규화)
└── night/
    ├── images/
    └── labels/
```

`data/` 폴더는 .gitignore에 포함되어 있습니다 (용량 문제).

## 카메라 사양 (MORAI)

- 해상도: 1280 × 720
- 위치: 우측 사이드 미러 부근 (x=2.15m, y=0.9m, z=0.55m)
- Pitch: 15° (지면 방향)
- Yaw: 90° (우측)
