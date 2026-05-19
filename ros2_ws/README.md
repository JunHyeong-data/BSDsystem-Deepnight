# BSD DeepNight ROS2 패키지

MORAI 시뮬레이션 → SGLDet 추론 → BSD 경고를 ROS2로 연동.

## 환경 요구사항

- Ubuntu 22.04 + ROS2 Humble (또는 Foxy on Ubuntu 20.04)
- Python 3.8+
- CUDA GPU 권장

## 설치

### 1) ROS2 의존성 설치
```bash
sudo apt install ros-${ROS_DISTRO}-cv-bridge \
                 ros-${ROS_DISTRO}-vision-msgs \
                 ros-${ROS_DISTRO}-sensor-msgs

pip3 install torch ultralytics opencv-python pyyaml numpy
```

### 2) 워크스페이스 빌드
```bash
# 리눅스 노트북 (MORAI 있는 곳)
cd ~/PycharmProjects/BSDsystem/ros2_ws

# Python 모듈 import 경로 보장 위해 BSDsystem/ 자체를 PYTHONPATH에 추가
export PYTHONPATH=$PYTHONPATH:$(realpath ..)

# colcon 빌드
colcon build --packages-select bsd_deepnight --symlink-install

# 환경 source
source install/setup.bash
```

## 실행

### A) 사전학습 가중치로 바로 추론 (학습 없이)
```bash
# Terminal 1: MORAI 시뮬레이터 실행

# Terminal 2: ROS2 detector (두 BSD 카메라)
ros2 launch bsd_deepnight bsd_detector.launch.py \
    weights:=$(realpath ../yolov8m.pt) \
    topic_right:=/morai/camera_right/image_raw \
    topic_left:=/morai/camera_left/image_raw

# Terminal 3: 시각화 확인
rqt_image_view /bsd/visualization_right
rqt_image_view /bsd/visualization_left

# Terminal 4: BSD 경고 확인
ros2 topic echo /bsd/warning
```

### B) MORAI 데이터로 학습한 가중치 사용
```bash
ros2 launch bsd_deepnight bsd_detector.launch.py \
    weights:=$(realpath ../checkpoints/best_model.pt)
```

### MORAI 카메라 토픽 확인 방법
```bash
# MORAI에서 실제 발행하는 카메라 토픽명 확인
ros2 topic list | grep -i camera

# 토픽명이 다른 경우 launch 인자로 지정
ros2 launch bsd_deepnight bsd_detector.launch.py \
    topic_right:=/실제_우측_토픽명 \
    topic_left:=/실제_좌측_토픽명
```

## ROS2 토픽 구조

```
[MORAI Camera-1 (우측 BSD)]           [MORAI Camera-2 (좌측 BSD)]
  /morai/camera_right/image_raw          /morai/camera_left/image_raw
        ↓                                        ↓
              [bsd_detector_node]
                      ↓
  /bsd/detections          (vision_msgs/Detection2DArray)
  /bsd/warning             (std_msgs/String: "SAFE"/"WARNING"/"DANGER")
  /bsd/visualization_right (sensor_msgs/Image)
  /bsd/visualization_left  (sensor_msgs/Image)
```

## 카메라 파라미터 (MORAI JSON 기준)

| 항목 | Camera-1 (우측) | Camera-2 (좌측) |
|------|----------------|----------------|
| 장착 위치 | x=2.15m, y=+0.9m, z=0.55m | x=2.15m, y=-0.9m, z=0.55m |
| Yaw | 90° (우측 방향) | 270° (좌측 방향) |
| Pitch | 15° (앞쪽 하향) | 15° (앞쪽 하향) |
| 해상도 | 1280×720 | 1280×720 |
| fx/fy | 320 / 320 | 320 / 320 |

## launch 파라미터

| 이름 | 기본값 | 설명 |
|------|--------|------|
| weights | yolov8m.pt | 가중치 파일 경로 |
| camera_config | configs/camera_config.yaml | 카메라 캘리브레이션 |
| topic_right | /morai/camera_right/image_raw | 우측 BSD 카메라 토픽 |
| topic_left | /morai/camera_left/image_raw | 좌측 BSD 카메라 토픽 |
| img_size | 640 | YOLO 입력 크기 |
| conf_threshold | 0.25 | 탐지 신뢰도 임계값 |
| iou_threshold | 0.45 | NMS IoU 임계값 |
| device | cuda | cuda/cpu |
| publish_vis | true | 시각화 영상 발행 여부 |
| undistort | false | 왜곡 보정 (MORAI 시뮬은 false) |

## 디버깅

```bash
# 토픽 목록
ros2 topic list

# 카메라 입력 들어오는지 확인
ros2 topic hz /morai/camera/image_raw

# 추론 결과 확인
ros2 topic echo /bsd/detections --once

# 경고 모니터링
ros2 topic echo /bsd/warning
```

## 트러블슈팅

**1. `import models` 실패**
→ `PYTHONPATH`에 BSDsystem 루트 추가:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/BSDsystem
```

**2. CUDA 메모리 부족**
→ launch 시 `device:=cpu`

**3. cv_bridge encoding 에러**
→ MORAI 카메라가 BGR/RGB/RGBA 어느 포맷인지 확인:
```bash
ros2 topic echo /morai/camera/image_raw --once | grep encoding
```
