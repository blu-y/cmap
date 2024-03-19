import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
import numpy as np
import open3d as o3d
import os
import pandas as pd
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

''' TF Tree
map  
└── odom  
    ├── base_footprint  
    └── base_link  
        ├── imu_link  
        └── shell_link  
            ├── rplidar_link  
            └── oakd_link  
                └── oakd_rgb_camera_frame  
                    └── oakd_rgb_camera_optical_frame 
'''
'''
/odom nav_msgs/msg/Odometry 1765
/scan sensor_msgs/msg/LaserScan 819
/imu sensor_msgs/msg/Imu 3612
/pose geometry_msgs/msg/PoseWithCovarianceStamped 253
/map nav_msgs/msg/OccupancyGrid 225
/oakd/rgb/preview/image_raw/compressed sensor_msgs/msg/CompressedImage 2928
/tf_static tf2_msgs/msg/TFMessage 4
/tf tf2_msgs/msg/TFMessage 6460
'''
class DataGenNode(Node):
    def __init__(self):
        super().__init__('DataGenNode')

        self.path = './dataset/tb4/pcd'
        if not os.path.exists(self.path): os.makedirs(self.path)
        self.path = './dataset/tb4/image'
        if not os.path.exists(self.path): os.makedirs(self.path)
        self.path = './dataset/tb4'

        self.laser_scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_scan_callback,
            100)
        self.laser_scan_sub  # prevent unused variable warning
        self.angles = None

        self.camera_sub = self.create_subscription(
            CompressedImage,
            '/oakd/rgb/preview/image_raw/compressed',
            self.camera_callback,
            100)
        self.camera_sub  # prevent unused variable warning
        self.bridge = CvBridge()
        self.image = None
        
    def camera_callback(self, image_data):
        fn = str(image_data.header.stamp.sec * 1000000000 + image_data.header.stamp.nanosec) + str('.png')
        fn = os.path.join(self.path, 'image', fn)
        self.image = self.bridge.compressed_imgmsg_to_cv2(image_data, 'bgr8')
        # with open(fn, 'wb') as f:
        #     f.write(image_data.data)
        # self.get_logger().info('Saved '+ fn)   


    def laser_scan_callback(self, scan_data):
        fn = str(scan_data.header.stamp.sec * 1000000000 + scan_data.header.stamp.nanosec) + str('.pcd')
        fn = os.path.join(self.path, 'pcd', fn)
        if self.angles is None:
            self.angles = np.arange(scan_data.angle_min, scan_data.angle_max + scan_data.angle_increment, scan_data.angle_increment)
        ranges = np.array(scan_data.ranges)
        x_ = ranges * np.cos(self.angles)
        y_ = ranges * np.sin(self.angles)
        z_ = np.zeros(len(ranges))  # Assuming flat surface
        cloud_points = np.vstack((x_, y_, z_)).transpose()
        
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_points)
        # o3d.io.write_point_cloud(fn, pcd)
        # self.get_logger().info('Saved '+ fn)

def main(args=None):
    rclpy.init(args=args)
    node = DataGenNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
