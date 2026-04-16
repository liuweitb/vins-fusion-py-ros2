"""ROS2 node for loop-closure detection and pose graph (mirrors pose_graph_node.cpp)."""
import sys
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from .keyframe import KeyFrame
from .pose_graph import PoseGraph


class LoopFusionNode(Node):
    def __init__(self, config_path: str):
        super().__init__("loop_fusion")

        self.pose_graph = PoseGraph()
        self.bridge = CvBridge()
        self._pending_img: dict = {}   # timestamp -> image
        self._keyframe_index = 0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )

        # Subscribe to odometry from VINS estimator
        self.create_subscription(Odometry, "/vins_estimator/odometry",
                                  self._odom_callback, sensor_qos)
        # Subscribe to image for keyframe building (reuse cam0 topic)
        self.create_subscription(Image, "/cam0/image_raw",
                                  self._image_callback, sensor_qos)

        # Publish loop-corrected path
        self._pub_path = self.create_publisher(Path, "/loop_fusion/path", 10)
        self._path_msg = Path()
        self._path_msg.header.frame_id = "world"

        self.get_logger().info("Loop fusion node started.")

    def _image_callback(self, msg: Image):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        img = self.bridge.imgmsg_to_cv2(msg, "mono8")
        self._pending_img[t] = img
        # Keep buffer small
        if len(self._pending_img) > 20:
            oldest = min(self._pending_img)
            del self._pending_img[oldest]

    def _odom_callback(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        P = np.array([msg.pose.pose.position.x,
                       msg.pose.pose.position.y,
                       msg.pose.pose.position.z])
        Q = np.array([msg.pose.pose.orientation.w,
                       msg.pose.pose.orientation.x,
                       msg.pose.pose.orientation.y,
                       msg.pose.pose.orientation.z])

        from scipy.spatial.transform import Rotation
        R = Rotation.from_quat([Q[1], Q[2], Q[3], Q[0]]).as_matrix()

        # Find closest image
        if not self._pending_img:
            return
        closest_t = min(self._pending_img, key=lambda k: abs(k - t))
        if abs(closest_t - t) > 0.05:
            return
        img = self._pending_img.pop(closest_t)

        kf = KeyFrame(
            t=t,
            index=self._keyframe_index,
            P=P, R=R,
            img=img,
            pts_3d=np.zeros((0, 3)),
            pts_2d=np.zeros((0, 2)),
        )
        self._keyframe_index += 1

        loop_idx = self.pose_graph.add_keyframe(kf)
        if loop_idx is not None:
            self.get_logger().info(
                f"Loop detected: frame {kf.index} <-> frame {loop_idx}")

        self._publish_path()

    def _publish_path(self):
        now = self.get_clock().now().to_msg()
        self._path_msg.header.stamp = now
        self._path_msg.poses.clear()

        for i, kf in enumerate(self.pose_graph.keyframes):
            from scipy.spatial.transform import Rotation
            q_xyzw = Rotation.from_matrix(kf.R_w_i).as_quat()
            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = "world"
            ps.pose.position.x = float(kf.T_w_i[0])
            ps.pose.position.y = float(kf.T_w_i[1])
            ps.pose.position.z = float(kf.T_w_i[2])
            ps.pose.orientation.x = float(q_xyzw[0])
            ps.pose.orientation.y = float(q_xyzw[1])
            ps.pose.orientation.z = float(q_xyzw[2])
            ps.pose.orientation.w = float(q_xyzw[3])
            self._path_msg.poses.append(ps)

        self._pub_path.publish(self._path_msg)


def main(args=None):
    rclpy.init(args=args)
    if len(sys.argv) < 2:
        print("Usage: loop_fusion_node <config.yaml>")
        sys.exit(1)
    node = LoopFusionNode(sys.argv[1])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
