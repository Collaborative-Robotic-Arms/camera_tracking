## Detector for ArUco Markers with Intel RealSense Camera
## Author: zptang (UMass Amherst)

import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt

class TrajectoryTracker:

    def __init__(self):
        self.trajectory = dict()

    def clear(self):
        self.trajectory = dict()

    def updateTrajectory(self, aligned_frame, detectorResult):
        corners, ids, rvecs, tvecs = detectorResult
        timestamp = aligned_frame.get_timestamp()

        if ids is not None:
            for i, markerID in enumerate(ids.flatten()):
                x, y, z = tvecs[i].flatten()
                rvec = rvecs[i].flatten()
                
                self._add(timestamp, markerID, (x, y, z), rvec)

    def _add(self, timestamp, id, coord, rvec):
        if id not in self.trajectory.keys():
            self.trajectory[id] = list()
        
        x, y, z = coord
        self.trajectory[id].append((timestamp, x, y, z, rvec))

    def plotTrajectory(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        for id in self.trajectory.keys():
            x_line = [x for t, x, y, z, rvec in self.trajectory[id]]
            y_line = [y for t, x, y, z, rvec in self.trajectory[id]]
            z_line = [z for t, x, y, z, rvec in self.trajectory[id]]
            ax.scatter3D(x_line, y_line, z_line)

        plt.show()