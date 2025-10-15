#!/usr/bin/env python3
"""
Aruco Pose Publisher Node
-------------------------
This ROS2 node detects ArUco markers using an Intel RealSense camera,
estimates their poses, filters positions using a Kalman filter, and publishes
poses to the 'aruco_pose' topic with TRANSIENT_LOCAL QoS.

Author: Marwan Mahmoud
"""

import os
import yaml
import joblib
import numpy as np
import cv2
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from ament_index_python.packages import get_package_share_directory

# Local modules
from realsense_aruco_tracking.ArUcoDetector import ArUcoDetector
from realsense_aruco_tracking.Camera import Camera
from realsense_aruco_tracking.TrajectoryTracker import TrajectoryTracker
from realsense_aruco_tracking.ArUcoGenerator import readConfig


# ==================== Kalman Filter ====================
class Kalman1D:
    """Simple 1D Kalman filter for smoothing positional data."""
    def __init__(self, process_var=1e-4, meas_var=1e-2):
        self.x = None
        self.P = 1.0
        self.Q = process_var
        self.R = meas_var

    def filter(self, z):
        if self.x is None:
            self.x = z
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (z - self.x)
        self.P *= (1 - K)
        return self.x


# ==================== Main Node ====================
class ArucoPosePublisher(Node):
    def __init__(self):
        super().__init__("aruco_pose_publisher")
        self._init_publishers()
        self._init_timer()

        self.get_logger().info("Aruco Pose Publisher Node started.")

        # Initialization pipeline
        self._load_config()
        self._load_calibration()
        self._load_model()
        self._init_camera()
        self._init_filters()
        self._init_detector_and_tracker()

        self.marker_size = 0.08
        self.pos_filters = {}

    # -----------------------------------------------------
    # ------------------- Initialization ------------------
    # -----------------------------------------------------
    def _init_publishers(self):
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        self.pose_pub = self.create_publisher(PoseStamped, "aruco_pose", qos)
        self.id_pub = self.create_publisher(Int32, "aruco_id", 10)

    def _init_timer(self):
        self.timer = self.create_timer(0.05, self.timer_callback)

    def _init_camera(self):
        self.camera = Camera()
        self.camera.startStreaming()

    def _init_filters(self):
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

    def _init_detector_and_tracker(self):
        self.aruco_detector = ArUcoDetector(self.dict_to_use)
        self.tracker = TrajectoryTracker()

    def _load_config(self):
        config_path = ('/home/marwan/camera_tracking/src/realsense_aruco_tracking/realsense_aruco_tracking/config.json')
        self.dict_to_use, self.visualize, self.grey_color, _ = readConfig(config_path)

    def _load_calibration(self):
        calib_path = ('/home/marwan/camera_tracking/src/realsense_aruco_tracking/realsense_aruco_tracking/realsense_calib.yaml')
        with open(calib_path, "r") as f:
            calib = yaml.safe_load(f)

        self.K = np.array(calib["camera_matrix"]["data"]).reshape(3, 3)
        self.D = np.array(calib["dist_coeffs"]["data"])
        self.fx, self.fy = self.K[0, 0], self.K[1, 1]
        self.cx, self.cy = self.K[0, 2], self.K[1, 2]
        self.depth_scale = calib.get("depth_scale", 0.001)
        dc = calib.get("depth_correction", {})
        self.a, self.b = dc.get("a", 1.0), dc.get("b", 1.0)

    def _load_model(self):
        try:
            model_path = "/home/marwan/camera_tracking/src/realsense_aruco_tracking/realsense_aruco_tracking/height_model_out/polynomial_model.joblib"
            model_bundle = joblib.load(model_path)
            self.model = model_bundle["model"]
            self.u_mean, self.u_std = model_bundle["u_mean"], model_bundle["u_std"]
            self.v_mean, self.v_std = model_bundle["v_mean"], model_bundle["v_std"]
            self.z_mean, self.z_std = model_bundle["z_mean"], model_bundle["z_std"]
            self.use_z_measured = model_bundle["use_z_measured"]
            self.get_logger().info("Loaded trained height model.")
        except Exception as e:
            self.model = None
            self.get_logger().warn(f"Could not load model: {e}")

    # -----------------------------------------------------
    # ------------------- Helper Methods ------------------
    # -----------------------------------------------------
    def _apply_realsense_filters(self, depth_frame):
        """Apply spatial, temporal, and hole-filling filters to depth frame."""
        for f in [self.spatial, self.temporal, self.hole_filling]:
            depth_frame = f.process(depth_frame)
        return depth_frame

    def _extract_marker_depth(self, depth_img, u, v):
        """Get depth value at pixel and convert to meters."""
        if v < 0 or u < 0 or v >= depth_img.shape[0] or u >= depth_img.shape[1]:
            return None
        depth_raw = depth_img[v, u]
        depth_m = depth_raw * self.depth_scale
        return depth_m if depth_m > 0 else None

    def _depth_to_camera_coords(self, u, v, z):
        """Convert depth pixel to camera coordinates."""
        X = (u - self.cx) * z / self.fx
        Y = (v - self.cy) * z / self.fy
        return np.array([X, Y, z])

    def _cam_to_base_coords(self, t_cam):
        """Transform camera coordinates to base frame."""
        if t_cam is None:
            return None
        x_c, y_c, z_c = t_cam
        x_b = x_c - 0.0546
        y_b = y_c + 0.674
        z_b = 0.769 - z_c
        return np.array([y_b, -x_b, z_b])

    def _apply_kalman(self, marker_id, pos):
        """Filter XYZ positions per marker."""
        if marker_id not in self.pos_filters:
            self.pos_filters[marker_id] = {k: Kalman1D(1e-5, 1e-3) for k in ['x', 'y', 'z']}
        return np.array([
            self.pos_filters[marker_id]['x'].filter(pos[0]),
            self.pos_filters[marker_id]['y'].filter(pos[1]),
            self.pos_filters[marker_id]['z'].filter(pos[2])
        ])

    def _predict_z_model(self, u, v, z_measured):
        """Predict corrected Z using trained regression model."""
        if self.model is None:
            return None
        try:
            if self.use_z_measured:
                X_in = np.array([[
                    (u - self.u_mean) / self.u_std,
                    (v - self.v_mean) / self.v_std,
                    (z_measured - self.z_mean) / self.z_std
                ]])
            else:
                X_in = np.array([[
                    (u - self.u_mean) / self.u_std,
                    (v - self.v_mean) / self.v_std
                ]])
            return float(self.model.predict(X_in)[0])
        except Exception as e:
            self.get_logger().warn(f"Z prediction failed: {e}")
            return None

    def _publish_pose(self, marker_id, pos, z_pred):
        """Publish pose and marker ID."""
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "base_link"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = pos[0], pos[1], z_pred
        pose.pose.orientation.w = 1.0

        self.pose_pub.publish(pose)
        self.id_pub.publish(Int32(data=int(marker_id)))

    # -----------------------------------------------------
    # ------------------- Main Callback -------------------
    # -----------------------------------------------------
    def timer_callback(self):
        frame = self.camera.getNextFrame()
        depth_frame = self._apply_realsense_filters(frame.get_depth_frame())
        color_frame = frame.get_color_frame()

        depth_img = np.asanyarray(depth_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())

        corners, ids, rvecs, tvecs = self.aruco_detector.detect_with_pose_estimation(
            color_img, self.K, None, self.marker_size
        )
        if ids is None:
            return

        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        for corner in corners:
            cv2.cornerSubPix(
                gray, corner, (3, 3), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            )

        for i, marker_id in enumerate(ids.flatten()):
            u, v = map(int, corners[i][0].mean(axis=0))
            depth_val = self._extract_marker_depth(depth_img, u, v)
            tvec_depth = self._depth_to_camera_coords(u, v, depth_val) if depth_val else None
            tvec_pnp = tvecs[i].flatten()

            base_pnp = self._cam_to_base_coords(tvec_pnp)
            base_depth = self._cam_to_base_coords(tvec_depth)

            fused = base_pnp.copy()
            if base_depth is not None:
                fused[2] = base_depth[2]

            filtered = self._apply_kalman(marker_id, fused)
            z_pred = self._predict_z_model(u, v, filtered[2]) or filtered[2]

            self._publish_pose(marker_id, filtered, z_pred)

            self.get_logger().info(
                f"[ID {marker_id}] PnP=({base_pnp[0]:.3f}, {base_pnp[1]:.3f}, {base_pnp[2]:.3f}) "
                f"| Zfused={filtered[2]:.3f} | Zpred={z_pred:.3f}"
            )

    # -----------------------------------------------------
    # ------------------- Cleanup -------------------------
    # -----------------------------------------------------
    def destroy_node(self):
        self.camera.stopStreaming()
        super().destroy_node()


# ==================== Entry Point ====================
def main(args=None):
    rclpy.init(args=args)
    node = ArucoPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
