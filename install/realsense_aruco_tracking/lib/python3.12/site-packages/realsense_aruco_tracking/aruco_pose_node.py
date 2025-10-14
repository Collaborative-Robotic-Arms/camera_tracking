#!/usr/bin/env python3
import os
import time
import yaml
import numpy as np
import cv2
import pyrealsense2 as rs
import pandas as pd
import logging
import joblib
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from realsense_aruco_tracking.ArUcoDetector import ArUcoDetector
from realsense_aruco_tracking.Camera import Camera
from realsense_aruco_tracking.TrajectoryTracker import TrajectoryTracker
from realsense_aruco_tracking.ArUcoGenerator import readConfig
from ament_index_python.packages import get_package_share_directory


# ---------------- Kalman Filter ----------------
class Kalman1D:
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2):
        self.x = None
        self.P = 1.0
        self.Q = process_variance
        self.R = measurement_variance

    def filter(self, z):
        if self.x is None:
            self.x = z
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x


# ---------------- ROS2 Node ----------------
class ArucoPosePublisher(Node):
    def __init__(self):
        super().__init__("aruco_pose_publisher")

        self.publisher_ = self.create_publisher(PoseStamped, "aruco_pose", 10)
        self.id_pub_ = self.create_publisher(Int32, "aruco_id", 10)
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info("Aruco Pose Publisher Node started.")

        # ---------- Load configuration ----------
        config_path = os.path.join(
            get_package_share_directory("realsense_aruco_tracking"),
            "config.json"
        )
        dict_to_use, visualize, grey_color, _ = readConfig(config_path)

        # ---------- Load calibration ----------
        pkg_path = get_package_share_directory('realsense_aruco_tracking')
        calib_path = os.path.join(pkg_path, 'config', 'realsense_calib.yaml')

        with open(calib_path, "r") as f:
            calib = yaml.safe_load(f)

        K = np.array(calib["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        D = np.array(calib["dist_coeffs"]["data"], dtype=np.float64)
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.depth_scale = float(calib.get("depth_scale", 0.001))
        self.a = float(calib.get("depth_correction", {}).get("a", 1.0))
        self.b = float(calib.get("depth_correction", {}).get("b", 1.0))
        self.K = K
        self.D = D

        # ---------- Load trained model ----------
        try:
            model_path = "/home/marwan/camera_tracking/src/realsense_aruco_tracking/realsense_aruco_tracking/height_model_out/polynomial_model.joblib"
            model_bundle = joblib.load(model_path)

            self.model = model_bundle["model"]
            self.u_mean = model_bundle["u_mean"]
            self.u_std = model_bundle["u_std"]
            self.v_mean = model_bundle["v_mean"]
            self.v_std = model_bundle["v_std"]
            self.z_mean = model_bundle["z_mean"]
            self.z_std = model_bundle["z_std"]
            self.use_z_measured = model_bundle["use_z_measured"]
            self.get_logger().info("Loaded trained height model.")
        except Exception as e:
            self.model = None
            self.get_logger().warn(f"Could not load model: {e}")

        # ---------- Initialize pipeline ----------
        self.arucoDetector = ArUcoDetector(dict_to_use)
        self.tracker = TrajectoryTracker()
        self.camera = Camera()
        self.camera.startStreaming()

        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

        self.marker_size = 0.08
        self.pos_filters = {}

    # -------- Camera to Base Frame transform --------
    def cam_to_base_pos(self, t_cam):
        x_c, y_c, z_c = t_cam
        x_b = x_c - 0.05
        y_b = y_c + 0.67
        z_b = 0.769 - z_c
        return np.array([y_b, -x_b, z_b], dtype=np.float64)

    # -------- Main periodic callback --------
    def timer_callback(self):
        frame = self.camera.getNextFrame()
        depth_frame = frame.get_depth_frame()
        color_frame = frame.get_color_frame()

        # Apply RealSense filters
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)

        # Convert to numpy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        h, w = color_image.shape[:2]

        corners, ids, rvecs, tvecs = self.arucoDetector.detect_with_pose_estimation(
            color_image, self.K, None, self.marker_size
        )

        if ids is not None:
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            for corner in corners:
                cv2.cornerSubPix(
                    gray,
                    corner,
                    winSize=(3, 3),
                    zeroZone=(-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                )

            for i, markerID in enumerate(ids.flatten()):
                tvec_pnp = tvecs[i].flatten()
                u_center = int(corners[i][0].mean(axis=0)[0])
                v_center = int(corners[i][0].mean(axis=0)[1])
                depth_value = depth_image[v_center, u_center] * self.depth_scale  # same as working code

                if depth_value > 0:
                    X = (u_center - self.cx) * depth_value / self.fx
                    Y = (v_center - self.cy) * depth_value / self.fy
                    Z = depth_value
                    tvec_depth = np.array([X, Y, Z], dtype=np.float64)
                else:
                    tvec_depth = None

                base_pos_pnp = self.cam_to_base_pos(tvec_pnp)
                base_pos_depth = self.cam_to_base_pos(tvec_depth) if tvec_depth is not None else None

                fused_pos = base_pos_pnp.copy()
                if base_pos_depth is not None:
                    fused_pos[2] = base_pos_depth[2]

                if markerID not in self.pos_filters:
                    self.pos_filters[markerID] = {
                        "x": Kalman1D(1e-5, 1e-3),
                        "y": Kalman1D(1e-5, 1e-3),
                        "z": Kalman1D(1e-5, 1e-3)
                    }

                fused_pos[0] = self.pos_filters[markerID]["x"].filter(fused_pos[0])
                fused_pos[1] = self.pos_filters[markerID]["y"].filter(fused_pos[1])
                z_fused = self.pos_filters[markerID]["z"].filter(fused_pos[2])

                # ---- Use model to predict true Z ----
                z_pred = None
                if self.model is not None:
                    try:
                        if self.use_z_measured:
                            X_in = np.array([[
                                (u_center - self.u_mean) / self.u_std,
                                (v_center - self.v_mean) / self.v_std,
                                (z_fused - self.z_mean) / self.z_std
                            ]])
                        else:
                            X_in = np.array([[
                                (u_center - self.u_mean) / self.u_std,
                                (v_center - self.v_mean) / self.v_std
                            ]])
                        z_pred = float(self.model.predict(X_in)[0])
                    except Exception as e:
                        self.get_logger().warn(f"[ID {markerID}] Prediction failed: {e}")

                # -------- Publish pose --------
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = "base_link"
                pose_msg.pose.position.x = float(fused_pos[0])
                pose_msg.pose.position.y = float(fused_pos[1])
                pose_msg.pose.position.z = float(z_pred if z_pred is not None else z_fused)
                pose_msg.pose.orientation.x = 0.0
                pose_msg.pose.orientation.y = 0.0
                pose_msg.pose.orientation.z = 0.0
                pose_msg.pose.orientation.w = 1.0

                self.publisher_.publish(pose_msg)
                self.id_pub_.publish(Int32(data=int(markerID)))

                self.get_logger().info(
                    f"[ID {markerID}] PnP: ({base_pos_pnp[0]:.3f}, {base_pos_pnp[1]:.3f}, {base_pos_pnp[2]:.3f}) | "
                    f"Zfused={z_fused:.3f} | Zpred={z_pred if z_pred else 'N/A'}"
                )

    def destroy_node(self):
        self.camera.stopStreaming()
        super().destroy_node()


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
