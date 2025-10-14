import sys
if sys.prefix == '/home/marwan/camera_tracking/venv':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/marwan/camera_tracking/install/realsense_aruco_tracking'
