import time
import yaml
import numpy as np
import cv2
import pyrealsense2 as rs
import pandas as pd
from ArUcoDetector import ArUcoDetector
from Camera import Camera
from TrajectoryTracker import TrajectoryTracker
from ArUcoGenerator import readConfig


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


def main():
    dict_to_use, visualize, grey_color, _ = readConfig(
        "/home/kareem-saleh/camera_tracking/RealsenseArUcoTracking/config.json")

    # ---------------- Load calibration from YAML ----------------
    with open("realsense_calib.yaml", "r") as f:
        calib = yaml.safe_load(f)

    # Intrinsics
    K = np.array(calib["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    D = np.array(calib["dist_coeffs"]["data"], dtype=np.float64)

    # Depth scale & correction
    depth_scale = float(calib.get("depth_scale", 0.001))
    a = float(calib.get("depth_correction", {}).get("a", 1.0))
    b = float(calib.get("depth_correction", {}).get("b", 1.0))

    print("[INFO] Calibration loaded")
    print("K:\n", K)
    print("D:", D)
    print(f"depth_scale={depth_scale}, depth_correction: a={a}, b={b}")

    # ---------------- ABB base frame relative to camera ----------------
    def cam_to_base_pos(t_cam):
        """
        t_cam: (3,) position in camera frame (meters) from ArUco pose
        Returns (3,) position in ABB base frame (meters)
        """
        x_c, y_c, z_c = t_cam
        x_b = x_c - 0.05
        y_b = y_c + 0.67
        z_b = 0.769 - z_c  # reversed Z with offset at table plane
        return np.array([y_b, x_b, z_b], dtype=np.float64)

    # ---------------- Your pipeline / classes ----------------
    arucoDetector = ArUcoDetector(dict_to_use)
    tracker = TrajectoryTracker()
    camera = Camera()
    camera.startStreaming()

    profile = camera.pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    _depth_intrinsics = depth_profile.get_intrinsics()

    marker_size = 0.08  # meters (adjust to your printed marker)
    corner_data = []

    # Kalman filters per marker for X, Y, Z
    pos_filters = {}

    # --- Setup RealSense filters ---
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()


    try:
        while True:
            frameset = camera.getNextFrame()

            # --- Get raw frames ---
            depth_frame = frameset.get_depth_frame()
            color_frame = frameset.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # --- Apply filters on depth frame ---
            filtered_depth = spatial.process(depth_frame)
            filtered_depth = temporal.process(filtered_depth)
            filtered_depth = hole_filling.process(filtered_depth)

            # --- Convert to numpy ---
            depth_image = np.asanyarray(filtered_depth.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # Undistort image
            h, w = color_image.shape[:2]
            _new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1)
            undistorted = cv2.undistort(color_image, K, None, None, None)

            # Detect ArUco markers
            corners, ids, rvecs, tvecs = arucoDetector.detect_with_pose_estimation(
                undistorted, K, None, marker_size
            )

            # ---- Subpixel corner refinement ----
            if corners is not None:
                gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
                for corner in corners:
                    cv2.cornerSubPix(
                        gray,
                        corner,
                        winSize=(3, 3),
                        zeroZone=(-1, -1),
                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                    )

            if ids is not None:
                for i, markerID in enumerate(ids.flatten()):
                    tvec_marker = tvecs[i].flatten()  # camera frame (meters)
                    rvec_marker = rvecs[i].flatten()

                    # --- Base-frame position from ArUco pose (raw) ---
                    base_pos = cam_to_base_pos(tvec_marker)
                    raw_base_pos = base_pos.copy()

                    # --- Create Kalman filters if first time seeing this marker ---
                    if markerID not in pos_filters:
                        pos_filters[markerID] = {
                            "x": Kalman1D(process_variance=1e-5, measurement_variance=1e-3),
                            "y": Kalman1D(process_variance=1e-5, measurement_variance=1e-3),
                            "z": Kalman1D(process_variance=1e-5, measurement_variance=1e-3)
                        }

                    # --- Apply Kalman filters to X, Y, Z ---
                    base_pos[0] = pos_filters[markerID]["x"].filter(base_pos[0])
                    base_pos[1] = pos_filters[markerID]["y"].filter(base_pos[1])
                    base_pos[2] = pos_filters[markerID]["z"].filter(base_pos[2])

                    # Print & log
                    print(f"[ID {markerID}] Base frame (m): "
                          f"X(filt)={base_pos[0]:.3f}, "
                          f"Y(filt)={base_pos[1]:.3f}, "
                          f"Z(filt)={base_pos[2]:.3f}")

                    corner_data.append({
                        "marker_id": int(markerID),
                        "cam_X": float(tvec_marker[0]),
                        "cam_Y": float(tvec_marker[1]),
                        "cam_Z": float(tvec_marker[2]),
                        "base_X_raw": float(raw_base_pos[0]),
                        "base_Y_raw": float(raw_base_pos[1]),
                        "base_Z_raw": float(raw_base_pos[2]),
                        "base_X_filtered": float(base_pos[0]),
                        "base_Y_filtered": float(base_pos[1]),
                        "base_Z_filtered": float(base_pos[2]),
                        "u_center": int(corners[i][0].mean(axis=0)[0]),
                        "v_center": int(corners[i][0].mean(axis=0)[1])
                    })

                    # Overlay text on image
                    topLeft = tuple(np.intp(corners[i][0][0]))
                    text = (f"ID:{int(markerID)} "
                            f"Bx:{base_pos[0]:.4f} By:{base_pos[1]:.4f} "
                            f"Bz:{base_pos[2]:.4f} m")
                    cv2.putText(
                        undistorted, text, (topLeft[0], topLeft[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )

            # Draw poses
            result = (corners, ids, rvecs, tvecs)
            color_image_with_pose = ArUcoDetector.getImageWithPose(
                undistorted, corners, ids, rvecs, tvecs, K, None
            )

            tracker.updateTrajectory(frameset, result)

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
            print("[INFO] Saved base-frame coordinates to aruco_base_frame_coordinates.csv")
        else:
            print("[INFO] No marker data captured.")


if __name__ == "__main__":
    main()