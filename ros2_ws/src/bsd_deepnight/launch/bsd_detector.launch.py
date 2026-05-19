"""
BSD DeepNight launch 파일 (두 카메라 지원)
==========================================
MORAI Camera-1(우측) + Camera-2(좌측) BSD 시스템 실행.

사용:
  ros2 launch bsd_deepnight bsd_detector.launch.py
  ros2 launch bsd_deepnight bsd_detector.launch.py weights:=/abs/path/best_model.pt

MORAI 토픽 확인:
  ros2 topic list | grep camera
  ros2 topic echo /morai/camera_right/image_raw --once | grep encoding
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── 인자 선언 ──────────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument(
            "weights",
            default_value="yolov8m.pt",
            description="YOLOv8 가중치 경로 (yolov8m.pt 또는 best_model.pt)",
        ),
        DeclareLaunchArgument(
            "camera_config",
            default_value="configs/camera_config.yaml",
            description="카메라 캘리브레이션 YAML 경로",
        ),
        # 우측 BSD 카메라 토픽 (MORAI 실제 토픽명으로 변경 필요)
        DeclareLaunchArgument(
            "topic_right",
            default_value="/morai/camera_right/image_raw",
            description="우측 BSD 카메라 토픽 (Camera-1, yaw=90°)",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="cuda",
            description="추론 디바이스 (cuda / cpu)",
        ),
        DeclareLaunchArgument(
            "img_size",
            default_value="640",
            description="YOLO 입력 이미지 크기",
        ),
        DeclareLaunchArgument(
            "conf_threshold",
            default_value="0.25",
        ),
        DeclareLaunchArgument(
            "iou_threshold",
            default_value="0.45",
        ),
        DeclareLaunchArgument(
            "publish_vis",
            default_value="true",
            description="시각화 토픽 발행 여부",
        ),
        DeclareLaunchArgument(
            "undistort",
            default_value="false",
            description="카메라 왜곡 보정 (MORAI 시뮬은 false)",
        ),
    ]

    # ── 노드 ───────────────────────────────────────────────────────────────
    detector_node = Node(
        package="bsd_deepnight",
        executable="detector_node",
        name="bsd_detector",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "weights":         LaunchConfiguration("weights"),
            "camera_config":   LaunchConfiguration("camera_config"),
            "topic_right":     LaunchConfiguration("topic_right"),
            "device":          LaunchConfiguration("device"),
            "img_size":        LaunchConfiguration("img_size"),
            "conf_threshold":  LaunchConfiguration("conf_threshold"),
            "iou_threshold":   LaunchConfiguration("iou_threshold"),
            "publish_vis":     LaunchConfiguration("publish_vis"),
            "undistort":       LaunchConfiguration("undistort"),
        }],
    )

    return LaunchDescription(args + [detector_node])
