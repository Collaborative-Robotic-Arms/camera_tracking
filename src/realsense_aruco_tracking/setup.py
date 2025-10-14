from setuptools import setup
import os
from glob import glob

package_name = 'realsense_aruco_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Copy config.json into the install/share directory
        (os.path.join('share', package_name), ['realsense_aruco_tracking/config.json']),
        ('share/ament_index/resource_index/packages', ['resource/realsense_aruco_tracking']),
        ('share/realsense_aruco_tracking', ['package.xml']),
        ('share/realsense_aruco_tracking/config', ['realsense_aruco_tracking/realsense_calib.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marwan',
    maintainer_email='your_email@example.com',
    description='Aruco marker pose tracking using RealSense camera',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracker = realsense_aruco_tracking.aruco_pose_node:main',
        ],
    },
)
