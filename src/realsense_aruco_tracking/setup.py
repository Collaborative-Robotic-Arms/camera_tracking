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

        # Include config and model files here
        (os.path.join('share', package_name, 'config'), glob('config/*.json')),
        (os.path.join('share', package_name, 'calibration'), glob('*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('height_model_out/*.joblib')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marwan',
    maintainer_email='2100771@eng.asu.edu.eg',
    description='RealSense ArUco tracking and pose publishing node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracker = realsense_aruco_tracking.aruco_pose_node:main',
        ],
    },
)
