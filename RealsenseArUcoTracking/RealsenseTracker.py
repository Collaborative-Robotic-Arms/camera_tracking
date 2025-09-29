import time
import yaml
import numpy as np
import cv2
import pyrealsense2 as rs
import pandas as pd
import logging
from ArUcoDetector import ArUcoDetector
from Camera import Camera
from TrajectoryTracker import TrajectoryTracker
from ArUcoGenerator import readConfig
import math
import os


# ---------------- Kalman Filter for smoothing ----------------
class Kalman1D:
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2):
        """
        Simple 1D Kalman filter.
        process_variance: how much we trust the motion model (lower = smoother)
        measurement_variance: expected sensor noise variance
        """
        self.x = None  # state estimate
        self.P = 1.0   # covariance
        self.Q = process_variance
        self.R = measurement_variance

    def filter(self, z):
        if self.x is None:
            self.x = z  # initialize with first measurement
        # Prediction
        self.P = self.P + self.Q
        # Kalman gain
        K = self.P / (self.P + self.R)
        # Update
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x
# ---------------- Rotation correction ----------------
def rotation_matrix(rx, ry, rz):
    """Builds a 3D rotation matrix from Euler angles (radians)."""
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(rx), -math.sin(rx)],
                   [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)],
                   [0, 1, 0],
                   [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                   [math.sin(rz), math.cos(rz), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def compute_y_tilt(z_left, z_right, table_width):
    """Compute Y-axis tilt angle in radians."""
    delta_z = z_right - z_left
    slope = delta_z / table_width
    return math.atan(slope)
def compute_x_tilt(z_up, z_down, table_length):
    """Compute X-axis tilt angle in radians."""
    delta_z = z_up - z_down
    slope = delta_z / table_length
    return math.atan(slope)

def get_correction_matrix(z_left, z_right, table_width ):
    """Return correction rotation matrix based on table measurements."""
    ry = compute_y_tilt(z_left, z_right, table_width)
    rx = compute_x_tilt(z_up, z_down, table_length)
    return rotation_matrix(rx, ry, 0)

# ---------------- Example usage ----------------
z_up = 770
z_down = 768
z_left = 790   # mm
z_right = 768  # mm
table_width = 740  # mm (370+370)
table_length = 740  # mm (370+370)

R_align = get_correction_matrix(z_left, z_right, table_width)


def main():
    # ---------------- Setup logger ----------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    dict_to_use, visualize, grey_color, _ = readConfig(config_path)

    # ---------------- Load calibration from YAML ----------------
    with open("realsense_calib.yaml", "r") as f:
        calib = yaml.safe_load(f)

    # Intrinsics
    K = np.array(calib["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    D = np.array(calib["dist_coeffs"]["data"], dtype=np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Depth scale & correction
    depth_scale = float(calib.get("depth_scale", 0.001))
    a = float(calib.get("depth_correction", {}).get("a", 1.0))
    b = float(calib.get("depth_correction", {}).get("b", 1.0))

    logging.info("Calibration loaded")
    logging.info(f"K:\n{K}")
    logging.info(f"D: {D}")
    logging.info(f"depth_scale={depth_scale}, depth_correction: a={a}, b={b}")

    # ---------------- ABB base frame relative to camera ----------------
    def cam_to_base_pos(t_cam):
        """
        t_cam: (3,) position in camera frame (meters) from ArUco pose or depth
        Returns (3,) position in ABB base frame (meters)
        """
        t_cam_rot = R_align @ t_cam

        x_c, y_c, z_c = t_cam_rot
        x_b = x_c - 0.05
        y_b = y_c + 0.67
        z_b = 0.773 - z_c  # reversed Z with offset at table plane
        return np.array([y_b, -x_b, z_b], dtype=np.float64)

    # ---------------- Your pipeline / classes ----------------
    arucoDetector = ArUcoDetector(dict_to_use)
    tracker = TrajectoryTracker()
    camera = Camera()
    camera.startStreaming()

    # --- RealSense filters ---
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    marker_size = 0.08  # meters (adjust to your printed marker)
    corner_data = []

    # Kalman filters per marker for X, Y, Z
    pos_filters = {}

    try:
        while True:
            frame = camera.getNextFrame()
            depth_image, color_image = camera.extractImagesFromFrame(frame)

            h, w = color_image.shape[:2]
            # No extra undistort since RealSense already outputs rectified frames

            # Detect ArUco markers
            corners, ids, rvecs, tvecs = arucoDetector.detect_with_pose_estimation(
                color_image, K, None, marker_size
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
                    tvec_pnp = tvecs[i].flatten()  # camera frame (PnP)
                    rvec_marker = rvecs[i].flatten()

                    # --- Depth from RealSense at marker center ---
                    u_center = int(corners[i][0].mean(axis=0)[0])
                    v_center = int(corners[i][0].mean(axis=0)[1])
                    depth_value = depth_image[v_center, u_center] * 0.001

                    if depth_value >= 0:  # valid depth
                        X = (u_center - cx) * depth_value / fx
                        Y = (v_center - cy) * depth_value / fy
                        Z = depth_value
                        tvec_depth = np.array([X, Y, Z], dtype=np.float64)
                    else:
                        tvec_depth = None

                    # --- Base-frame positions ---
                    base_pos_pnp = cam_to_base_pos(tvec_pnp)
                    base_pos_depth = cam_to_base_pos(tvec_depth) if tvec_depth is not None else None

                    # --- Fusion: take XY from PnP, Z from depth if available ---
                    fused_pos = base_pos_pnp.copy()
                    if base_pos_depth is not None:
                        fused_pos[2] = base_pos_depth[2]

                    # --- Kalman filtering ---
                    if markerID not in pos_filters:
                        pos_filters[markerID] = {
                            "x": Kalman1D(process_variance=1e-5, measurement_variance=1e-3),
                            "y": Kalman1D(process_variance=1e-5, measurement_variance=1e-3),
                            "z": Kalman1D(process_variance=1e-5, measurement_variance=1e-3)
                        }

                    fused_pos[0] = pos_filters[markerID]["x"].filter(fused_pos[0])
                    fused_pos[1] = pos_filters[markerID]["y"].filter(fused_pos[1])
                    fused_pos[2] = pos_filters[markerID]["z"].filter(fused_pos[2])

                    # --- Logging ---
                    logging.info(
                        f"[ID {markerID}] "
                        f"PnP: X={base_pos_pnp[0]:.3f}, Y={base_pos_pnp[1]:.3f}, Z={base_pos_pnp[2]:.3f} | "
                        f"Depth: {base_pos_depth if base_pos_depth is not None else 'N/A'} | "
                        f"Fused: X={fused_pos[0]:.3f}, Y={fused_pos[1]:.3f}, Z={fused_pos[2]:.3f}"
                    )

                    # --- Save data ---
                    corner_data.append({
                        "marker_id": int(markerID),
                        "cam_X_pnp": float(tvec_pnp[0]),
                        "cam_Y_pnp": float(tvec_pnp[1]),
                        "cam_Z_pnp": float(tvec_pnp[2]),
                        "base_X_pnp": float(base_pos_pnp[0]),
                        "base_Y_pnp": float(base_pos_pnp[1]),
                        "base_Z_pnp": float(base_pos_pnp[2]),
                        "cam_X_depth": float(tvec_depth[0]) if tvec_depth is not None else None,
                        "cam_Y_depth": float(tvec_depth[1]) if tvec_depth is not None else None,
                        "cam_Z_depth": float(tvec_depth[2]) if tvec_depth is not None else None,
                        "base_X_depth": float(base_pos_depth[0]) if base_pos_depth is not None else None,
                        "base_Y_depth": float(base_pos_depth[1]) if base_pos_depth is not None else None,
                        "base_Z_depth": float(base_pos_depth[2]) if base_pos_depth is not None else None,
                        "base_X_fused": float(fused_pos[0]),
                        "base_Y_fused": float(fused_pos[1]),
                        "base_Z_fused": float(fused_pos[2]),
                        "u_center": u_center,
                        "v_center": v_center
                    })

                    # Overlay text
                    topLeft = tuple(np.intp(corners[i][0][0]))
                    text = (f"ID:{int(markerID)} "
                            f"X:{fused_pos[0]:.3f} Y:{fused_pos[1]:.3f} Z:{fused_pos[2]:.3f} m")
                    cv2.putText(
                        color_image, text, (topLeft[0], topLeft[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )

            # Draw poses
            result = (corners, ids, rvecs, tvecs)
            color_image_with_pose = ArUcoDetector.getImageWithPose(
                color_image, corners, ids, rvecs, tvecs, K, None
            )

            tracker.updateTrajectory(frame, result)

            cv2.namedWindow('World Frame ArUco Tracking', cv2.WINDOW_NORMAL)
            cv2.imshow('World Frame ArUco Tracking', color_image_with_pose)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.stopStreaming()
        cv2.destroyAllWindows()
        if corner_data:
            df = pd.DataFrame(corner_data)
            df.to_csv("aruco_base_frame_coordinates.csv", index=False)
            logging.info("[INFO] Saved base-frame coordinates to aruco_base_frame_coordinates.csv")
        else:
            logging.info("[INFO] No marker data captured.")


if __name__ == "__main__":
    main()