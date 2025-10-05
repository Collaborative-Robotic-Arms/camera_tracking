import os
import time
import yaml
import numpy as np
import cv2
import pyrealsense2 as rs
import pandas as pd
import logging
import joblib
from ArUcoDetector import ArUcoDetector
from Camera import Camera
from TrajectoryTracker import TrajectoryTracker
from ArUcoGenerator import readConfig


# ---------------- Kalman Filter for smoothing ----------------
class Kalman1D:
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2):
        self.x = None  # state estimate
        self.P = 1.0   # covariance
        self.Q = process_variance
        self.R = measurement_variance

    def filter(self, z):
        if self.x is None:
            self.x = z  # initialize with first measurement
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x


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

    K = np.array(calib["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    D = np.array(calib["dist_coeffs"]["data"], dtype=np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    depth_scale = float(calib.get("depth_scale", 0.001))
    a = float(calib.get("depth_correction", {}).get("a", 1.0))
    b = float(calib.get("depth_correction", {}).get("b", 1.0))

    logging.info("Calibration loaded")
    logging.info(f"K:\n{K}")
    logging.info(f"D: {D}")
    logging.info(f"depth_scale={depth_scale}, depth_correction: a={a}, b={b}")

    # ---------------- Load trained model ----------------
    try:
        model_bundle = joblib.load("height_model_out/polynomial_model.joblib")
        model = model_bundle["model"]
        u_mean, u_std = model_bundle["u_mean"], model_bundle["u_std"]
        v_mean, v_std = model_bundle["v_mean"], model_bundle["v_std"]
        z_mean, z_std = model_bundle["z_mean"], model_bundle["z_std"]
        use_z_measured = model_bundle["use_z_measured"]
        logging.info("[INFO] Loaded trained RF model for height prediction")
    except Exception as e:
        logging.error(f"[ERROR] Could not load model: {e}")
        model = None

    # ---------------- ABB base frame relative to camera ----------------
    def cam_to_base_pos(t_cam):
        x_c, y_c, z_c = t_cam
        x_b = x_c - 0.05
        y_b = y_c + 0.67
        z_b = 0.769 - z_c
        return np.array([y_b, -x_b, z_b], dtype=np.float64)

    # ---------------- Your pipeline / classes ----------------
    arucoDetector = ArUcoDetector(dict_to_use)
    tracker = TrajectoryTracker()
    camera = Camera()
    camera.startStreaming()

    # Add filters
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()


    marker_size = 0.08  # meters
    corner_data = []
    pos_filters = {}

    # --- Throttle parameters ---
    save_interval = 0.5  # seconds between saving data points
    last_save_time = time.time()

    try:
        while True:
            frame = camera.getNextFrame()

            # Extract individual frames
            depth_frame = frame.get_depth_frame()
            color_frame = frame.get_color_frame()

            # Apply filters
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            # Convert filtered frames to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            h, w = color_image.shape[:2]

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
                    tvec_pnp = tvecs[i].flatten()
                    rvec_marker = rvecs[i].flatten()

                    u_center = int(corners[i][0].mean(axis=0)[0])
                    v_center = int(corners[i][0].mean(axis=0)[1])
                    depth_value = depth_image[v_center, u_center] * 0.001

                    if depth_value >= 0:
                        X = (u_center - cx) * depth_value / fx
                        Y = (v_center - cy) * depth_value / fy
                        Z = depth_value
                        tvec_depth = np.array([X, Y, Z], dtype=np.float64)
                    else:
                        tvec_depth = None

                    base_pos_pnp = cam_to_base_pos(tvec_pnp)
                    base_pos_depth = cam_to_base_pos(tvec_depth) if tvec_depth is not None else None

                    fused_pos = base_pos_pnp.copy()
                    if base_pos_depth is not None:
                        fused_pos[2] = base_pos_depth[2]

                    if markerID not in pos_filters:
                        pos_filters[markerID] = {
                            "x": Kalman1D(process_variance=1e-5, measurement_variance=1e-3),
                            "y": Kalman1D(process_variance=1e-5, measurement_variance=1e-3),
                            "z": Kalman1D(process_variance=1e-5, measurement_variance=1e-3)
                        }

                    fused_pos[0] = pos_filters[markerID]["x"].filter(fused_pos[0])
                    fused_pos[1] = pos_filters[markerID]["y"].filter(fused_pos[1])
                    z_fused = pos_filters[markerID]["z"].filter(fused_pos[2])

                    # ---- Use model to predict true Z ----
                    z_pred = None
                    if model is not None:
                        try:
                            if use_z_measured:
                                # ✅ use z_fused as z_measured input
                                X_in = np.array([[
                                    (u_center - u_mean) / u_std,
                                    (v_center - v_mean) / v_std,
                                    (z_fused - z_mean) / z_std
                                ]])
                            else:
                                X_in = np.array([[
                                    (u_center - u_mean) / u_std,
                                    (v_center - v_mean) / v_std
                                ]])
                            z_pred = float(model.predict(X_in)[0])
                        except Exception as e:
                            logging.warning(f"[ID {markerID}] Prediction failed: {e}")

                    logging.info(
                        f"[ID {markerID}] "
                        f"PnP: X={base_pos_pnp[0]:.3f}, Y={base_pos_pnp[1]:.3f}, Z={base_pos_pnp[2]:.3f} | "
                        f"Fused Z={z_fused:.3f} | "
                        f"Z_pred={z_pred if z_pred is not None else 'N/A':.3f} | "
                        f"Final: X={base_pos_pnp[0]:.3f}, Y={base_pos_pnp[1]:.3f}, Z={z_pred if z_pred is not None else 'N/A':.3f}"
                    )

                    # --- Throttled data saving ---
                    current_time = time.time()
                    if current_time - last_save_time >= save_interval:
                        corner_data.append({
                            "base_Z_fused": float(z_fused),
                            "u_center": u_center,
                            "v_center": v_center,
                            "Z_pred": z_pred if z_pred is not None else np.nan
                        })
                        last_save_time = current_time

                    # Overlay text
                    topLeft = tuple(np.intp(corners[i][0][0]))
                    text = (f"ID:{int(markerID)} "
                            f"Zfused:{z_fused:.3f} Zpred:{z_pred:.3f}" if z_pred is not None else f"ID:{int(markerID)} Zfused:{z_fused:.3f}")
                    cv2.putText(
                        color_image, text, (topLeft[0], topLeft[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )

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