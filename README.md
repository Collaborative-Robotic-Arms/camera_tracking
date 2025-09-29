# camera_tracking
# RealSense ArUco Tracking

This project generates ArUco markers, detects them in real-time, and tracks their position using an **Intel RealSense D400 series camera** (tested on D435i).  
It uses **PnP pose estimation** combined with depth information and applies a **Kalman filter** for smoother tracking.  
The system outputs camera-frame, base-frame, and fused coordinates, and saves them into a CSV file for later analysis.

---

## Features

- ArUco marker generation and detection  
- Pose estimation using OpenCV's ArUco module  
- Depth integration from Intel RealSense D400 series cameras  
- Fusion of PnP pose and depth measurements  
- Kalman filter smoothing for stable tracking  
- Data logging to CSV (marker positions, pixel coordinates, etc.)  
- Visualization of marker tracking and pose overlay  

---

## Requirements

- Python 3.13.5  
- NumPy 2.2.6  
- OpenCV (with contrib modules) 4.12.0  
- Intel RealSense D400 series (tested on D435i)  

---

## Usage

1. Connect an **Intel RealSense D400 series camera** (e.g., D435i, D415, D455).  
2. Run the tracker script:  

   ```bash
   python RealsenseTracker.py
3. While running:

- Press **q** to save the tracked coordinates into a CSV file.  
- Press **CTRL + C** to exit.  


