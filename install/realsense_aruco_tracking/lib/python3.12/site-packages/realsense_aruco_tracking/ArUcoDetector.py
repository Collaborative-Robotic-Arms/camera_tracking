## Detector for ArUco Markers with Intel RealSense Camera
## Author: zptang (UMass Amherst)

import cv2
import numpy as np
import pyrealsense2 as rs
from realsense_aruco_tracking.Camera import Camera

class ArUcoDetector:

    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }

    def __init__(self, dict_to_use):
        self.dict_to_use = dict_to_use
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ArUcoDetector.ARUCO_DICT[dict_to_use])
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def detect(self, image):
        return self.aruco_detector.detectMarkers(image)

    @staticmethod
    def getImageWithMarkers(input_image, detect_res):
        image = input_image.copy()
        corners, ids, rejected = detect_res
        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
        return image
    
    def detect_with_pose_estimation(self, image, camera_matrix, dist_coeffs, marker_size):
        corners, ids, rejected = self.aruco_detector.detectMarkers(image)
        rvecs, tvecs = [], []

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_size, camera_matrix, dist_coeffs
            )
        return corners, ids, rvecs, tvecs

    @staticmethod
    def getImageWithPose(input_image, corners, ids, rvecs, tvecs, camera_matrix, dist_coeffs):
        image = input_image.copy()
        if ids is not None:
            for i in range(len(ids)):
                cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.04)
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
        return image

if __name__ == '__main__':
    import time
    
    dict_to_use = 'DICT_5X5_50'
    arucoDetector = ArUcoDetector(dict_to_use)
    camera = Camera()
    camera.startStreaming()
    
    profile = camera.pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()

    camera_matrix = np.array([
        [607.6493463464219, 0.0, 330.2045740645484],
        [0.0, 605.19606629627, 246.36866587909964],
        [0, 0, 1]
    ])
    dist_coeffs = np.array(depth_intrinsics.coeffs)
    marker_size = 0.08

    try:
        while True:
            frame = camera.getNextFrame()
            depth_image, color_image = camera.extractImagesFromFrame(frame)

            grey_color = 153
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
            masked_color_image = np.where(depth_image_3d <= 0, grey_color, color_image)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            corners, ids, rvecs, tvecs = arucoDetector.detect_with_pose_estimation(
                color_image, camera_matrix, dist_coeffs, marker_size
            )

            color_image_with_pose = ArUcoDetector.getImageWithPose(
                color_image, corners, ids, rvecs, tvecs, camera_matrix, dist_coeffs
            )

            cv2.namedWindow('RealSense with Pose', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense with Pose', color_image_with_pose)
            cv2.waitKey(1)

    finally:
        camera.stopStreaming()