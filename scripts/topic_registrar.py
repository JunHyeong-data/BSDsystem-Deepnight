#!/usr/bin/env python3
"""
MORAI rosbridge 우회용 타입 등록기.

MORAI Sim(rosbridge 클라이언트)이 'advertise' 단계 없이 바로 publish해서
rosbridge_server가 토픽 타입을 모르고 메시지를 drop함.

이 노드는 같은 토픽에 대해 ROS2 publisher를 미리 만들어 타입을 ROS graph에
등록시켜둠. 그러면 MORAI가 보내는 메시지를 rosbridge_server가 기존 타입으로
infer해서 정상 라우팅함.

실행: source ROS2 + colcon_ws 후 python3 scripts/topic_registrar.py
종료: Ctrl+C
"""
import rclpy
from rclpy.node import Node
from morai_msgs.msg import EgoVehicleStatus, ObjectStatusList
from sensor_msgs.msg import CompressedImage


class TopicRegistrar(Node):
    def __init__(self):
        super().__init__("topic_registrar")
        # Publisher 단순 등록 (실제 publish는 안 함)
        self.pub_ego = self.create_publisher(EgoVehicleStatus, "/Ego_topic", 10)
        self.pub_obj = self.create_publisher(ObjectStatusList, "/Object_topic", 10)
        self.pub_img = self.create_publisher(CompressedImage, "/image_jpeg/rgb", 10)
        self.get_logger().info(
            "Registered topic types: /Ego_topic, /Object_topic, /image_jpeg/rgb. "
            "Keeping node alive — Ctrl+C to stop."
        )


def main():
    rclpy.init()
    node = TopicRegistrar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
