import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, CompressedImage
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
import numpy as np
import open3d as o3d
import os
import pandas as pd
from cv_bridge import CvBridge
import cv2
from rclpy.qos import qos_profile_sensor_data

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
/imu sensor_msgs/msg/Imu 3612
/pose geometry_msgs/msg/PoseWithCovarianceStamped 253
/map nav_msgs/msg/OccupancyGrid 225
/tf_static tf2_msgs/msg/TFMessage 4
/tf tf2_msgs/msg/TFMessage 6460
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
        self.fn_db = './dataset/tb4/db.csv'
        with open(self.fn_db, 'w') as f:
            f.write('timestamp, sensor, filename, position_x, position_y, position_z, orientation_x, orientation_y, orientation_z, orientation_w\n')

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

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile_sensor_data)
        self.odom_sub  # prevent unused variable warning
        self.odom = None

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            qos_profile_sensor_data)
        self.imu_sub  # prevent unused variable warning
        self.imu = None

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/pose',
            self.pose_callback,
            100)
        self.pose_sub  # prevent unused variable warning
        self.pose = None
    
        self.tf_sub = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            100)
        self.tf_sub  # prevent unused variable warning
        self.tf = None

        self.tf_static_sub = self.create_subscription(
            TFMessage,
            '/tf_static',
            self.tf_static_callback,
            100)
        self.tf_static_sub  # prevent unused variable warning
        self.tf_static = None

    def savedb(self, timestamp, sensor, filename):
        if self.pose is None: pose_x, pose_y, pose_z, ori_x, ori_y, ori_z, ori_w = None, None, None, None, None, None, None
        else: pose_x, pose_y, pose_z, ori_x, ori_y, ori_z, ori_w = self.pose[0:7]
        with open(self.fn_db, 'a') as f:
            f.write(str(timestamp) + ',' + 
                    sensor + ',' + 
                    filename + ',' + 
                    str(pose_x) + ',' + 
                    str(pose_y) + ',' + 
                    str(pose_z) + ',' + 
                    str(ori_x) + ',' + 
                    str(ori_y) + ',' + 
                    str(ori_z) + ',' + 
                    str(ori_w) + '\n')
        

    def tf_static_callback(self, tf_data):
        timestamp = tf_data.transforms[0].header.stamp.sec * 1000000000 + tf_data.transforms[0].header.stamp.nanosec
        if self.tf_static is None:
            self.tf_static_fn = 'tf_static.csv'
            self.tf_static_fn = os.path.join(self.path, self.tf_static_fn)
            with open(self.tf_static_fn, 'w') as f:
                f.write('timestamp,' +
                        'frame_id,' +
                        'child_frame_id,' +
                        'transform_translation_x,transform_translation_y,transform_translation_z,' + 
                        'transform_rotation_x,transform_rotation_y,transform_rotation_z,transform_rotation_w\n')
            self.tf_static = {}
        for transform in tf_data.transforms:
            tf = [timestamp, transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z, 
                  transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
            if transform.header.frame_id not in self.tf_static:
                self.tf_static[transform.header.frame_id] = {}
            self.tf_static[transform.header.frame_id][transform.child_frame_id] = tf
            with open(self.tf_static_fn, 'a') as f:
                f.write(str(timestamp) + ',' + 
                        transform.header.frame_id + ',' +
                        transform.child_frame_id + ',' + 
                        str(transform.transform.translation.x) + ',' + 
                        str(transform.transform.translation.y) + ',' + 
                        str(transform.transform.translation.z) + ',' + 
                        str(transform.transform.rotation.x) + ',' + 
                        str(transform.transform.rotation.y) + ',' + 
                        str(transform.transform.rotation.z) + ',' + 
                        str(transform.transform.rotation.w) + '\n')
    
    def tf_callback(self, tf_data):
        timestamp = tf_data.transforms[0].header.stamp.sec * 1000000000 + tf_data.transforms[0].header.stamp.nanosec
        if self.tf is None:
            self.tf_fn = 'tf.csv'
            self.tf_fn = os.path.join(self.path, self.tf_fn)
            with open(self.tf_fn, 'w') as f:
                f.write('timestamp,' +
                        'frame_id,' +
                        'child_frame_id,' +
                        'transform_translation_x,transform_translation_y,transform_translation_z,' + 
                        'transform_rotation_x,transform_rotation_y,transform_rotation_z,transform_rotation_w\n')
            self.tf = {}
        for transform in tf_data.transforms:
            tf = [timestamp, transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z, 
                  transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
            if transform.header.frame_id not in self.tf:
                self.tf[transform.header.frame_id] = {}
            self.tf[transform.header.frame_id][transform.child_frame_id] = tf
            with open(self.tf_fn, 'a') as f:
                f.write(str(timestamp) + ',' + 
                        transform.header.frame_id + ',' +
                        transform.child_frame_id + ',' + 
                        str(transform.transform.translation.x) + ',' + 
                        str(transform.transform.translation.y) + ',' + 
                        str(transform.transform.translation.z) + ',' + 
                        str(transform.transform.rotation.x) + ',' + 
                        str(transform.transform.rotation.y) + ',' + 
                        str(transform.transform.rotation.z) + ',' + 
                        str(transform.transform.rotation.w) + '\n')
    
    def pose_callback(self, pose_data):
        timestamp = pose_data.header.stamp.sec * 1000000000 + pose_data.header.stamp.nanosec
        if self.pose is None:
            self.pose_fn = 'pose.csv'
            self.pose_fn = os.path.join(self.path, self.pose_fn)
            with open(self.pose_fn, 'w') as f:
                f.write('timestamp,' +
                        'position_x,position_y,position_z,' + 
                        'orientation_x,orientation_y,orientation_z,orientation_w,' +
                        'covariance_0,covariance_1,covariance_6,covariance_7,covariance_35\n')
        pose = [timestamp, 
                pose_data.pose.pose.position.x, pose_data.pose.pose.position.y, pose_data.pose.pose.position.z, 
                pose_data.pose.pose.orientation.x, pose_data.pose.pose.orientation.y, pose_data.pose.pose.orientation.z, pose_data.pose.pose.orientation.w]
        self.pose = [pose_data.pose.pose.position.x, pose_data.pose.pose.position.y, pose_data.pose.pose.position.z,
                     pose_data.pose.pose.orientation.x, pose_data.pose.pose.orientation.y, pose_data.pose.pose.orientation.z, pose_data.pose.pose.orientation.w,
                     pose_data.pose.covariance[0], pose_data.pose.covariance[1], 
                     pose_data.pose.covariance[6], pose_data.pose.covariance[7], pose_data.pose.covariance[35]]
        with open(self.pose_fn, 'a') as f:
            f.write(str(timestamp) + ',' + 
                    str(pose_data.pose.pose.position.x) + ',' + 
                    str(pose_data.pose.pose.position.y) + ',' + 
                    str(pose_data.pose.pose.position.z) + ',' + 
                    str(pose_data.pose.pose.orientation.x) + ',' + 
                    str(pose_data.pose.pose.orientation.y) + ',' + 
                    str(pose_data.pose.pose.orientation.z) + ',' + 
                    str(pose_data.pose.pose.orientation.w) + ',' +
                    str(pose_data.pose.covariance[0]) + ',' +
                    str(pose_data.pose.covariance[1]) + ',' +
                    str(pose_data.pose.covariance[6]) + ',' +
                    str(pose_data.pose.covariance[7]) + ',' +
                    str(pose_data.pose.covariance[35]) + '\n')
        
    def imu_callback(self, imu_data):
        timestamp = imu_data.header.stamp.sec * 1000000000 + imu_data.header.stamp.nanosec
        if self.imu is None:
            self.imu_fn = 'imu.csv'
            self.imu_fn = os.path.join(self.path, self.imu_fn)
            with open(self.imu_fn, 'w') as f:
                f.write('timestamp,' +
                        'orientation_x,orientation_y,orientation_z,orientation_w,' +
                        'angular_x,angular_y,angular_z,' +
                        'linear_x,linear_y,linear_z\n')
            self.imu = []
        imu = [timestamp, 
                imu_data.orientation.x, imu_data.orientation.y, imu_data.orientation.z, imu_data.orientation.w, 
                imu_data.angular_velocity.x, imu_data.angular_velocity.y, imu_data.angular_velocity.z, 
                imu_data.linear_acceleration.x, imu_data.linear_acceleration.y, imu_data.linear_acceleration.z,
                imu_data.orientation_covariance, imu_data.angular_velocity_covariance, imu_data.linear_acceleration_covariance]
        if sum(imu_data.orientation_covariance) != 0 or sum(imu_data.angular_velocity_covariance) != 0 or sum(imu_data.linear_acceleration_covariance) != 0:
            print(timestamp, sum(imu_data.orientation_covariance), sum(imu_data.angular_velocity_covariance), sum(imu_data.linear_acceleration_covariance))
        self.imu.append(imu)
        with open(self.imu_fn, 'a') as f:
            f.write(str(timestamp) + ',' + 
                    str(imu_data.orientation.x) + ',' + 
                    str(imu_data.orientation.y) + ',' + 
                    str(imu_data.orientation.z) + ',' + 
                    str(imu_data.orientation.w) + ',' + 
                    str(imu_data.angular_velocity.x) + ',' + 
                    str(imu_data.angular_velocity.y) + ',' + 
                    str(imu_data.angular_velocity.z) + ',' + 
                    str(imu_data.linear_acceleration.x) + ',' + 
                    str(imu_data.linear_acceleration.y) + ',' + 
                    str(imu_data.linear_acceleration.z) + '\n')

    def odom_callback(self, odom_data):
        timestamp = odom_data.header.stamp.sec * 1000000000 + odom_data.header.stamp.nanosec
        if self.odom is None:
            self.odom_fn = 'odometry.csv'
            self.odom_fn = os.path.join(self.path, self.odom_fn)
            with open(self.odom_fn, 'w') as f:
                f.write('timestamp,' +
                        'position_x,position_y,position_z,' + 
                        'orientation_x,orientation_y,orientation_z,orientation_w,' +
                        'linear_x,linear_y,linear_z,' +
                        'angular_x,angular_y,angular_z\n')
            self.odom = []
        odom = [timestamp, 
                odom_data.pose.pose.position.x, odom_data.pose.pose.position.y, odom_data.pose.pose.position.z,  
                odom_data.pose.pose.orientation.x, odom_data.pose.pose.orientation.y, odom_data.pose.pose.orientation.z, 
                odom_data.pose.pose.orientation.w, odom_data.twist.twist.linear.x, odom_data.twist.twist.linear.y, 
                odom_data.twist.twist.linear.z, odom_data.twist.twist.angular.x, odom_data.twist.twist.angular.y, odom_data.twist.twist.angular.z]
        self.odom.append(odom)
        with open(self.odom_fn, 'a') as f:
            f.write(str(timestamp) + ',' + 
                    str(odom_data.pose.pose.position.x) + ',' + 
                    str(odom_data.pose.pose.position.y) + ',' + 
                    str(odom_data.pose.pose.position.z) + ',' + 
                    str(odom_data.pose.pose.orientation.x) + ',' + 
                    str(odom_data.pose.pose.orientation.y) + ',' + 
                    str(odom_data.pose.pose.orientation.z) + ',' + 
                    str(odom_data.pose.pose.orientation.w) + ',' + 
                    str(odom_data.twist.twist.linear.x) + ',' + 
                    str(odom_data.twist.twist.linear.y) + ',' + 
                    str(odom_data.twist.twist.linear.z) + ',' + 
                    str(odom_data.twist.twist.angular.x) + ',' + 
                    str(odom_data.twist.twist.angular.y) + ',' + 
                    str(odom_data.twist.twist.angular.z) + '\n')
        
    def camera_callback(self, image_data):
        timespamp = image_data.header.stamp.sec * 1000000000 + image_data.header.stamp.nanosec
        fn = str(timespamp) + str('.png')
        fn = os.path.join(self.path, 'image', fn)
        self.image = self.bridge.compressed_imgmsg_to_cv2(image_data, 'bgr8')
        with open(fn, 'wb') as f:
            f.write(image_data.data)
        self.savedb(timespamp, 'camera', fn)
        # self.get_logger().info('Saved '+ fn)   

    def laser_scan_callback(self, scan_data):
        timestamp = scan_data.header.stamp.sec * 1000000000 + scan_data.header.stamp.nanosec
        fn = str(timestamp) + str('.pcd')
        fn = os.path.join(self.path, 'pcd', fn)
        if self.angles is None:
            self.angles = np.arange(scan_data.angle_min, scan_data.angle_max + scan_data.angle_increment, scan_data.angle_increment)
        ranges = np.array(scan_data.ranges)
        x_ = ranges * np.cos(self.angles)
        y_ = ranges * np.sin(self.angles)
        z_ = np.zeros(len(ranges))  # Assuming flat surface
        cloud_points = np.vstack((x_, y_, z_)).transpose()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_points)
        o3d.io.write_point_cloud(fn, pcd)
        self.savedb(timestamp, 'laser_scan', fn)
        # self.get_logger().info('Saved '+ fn)

def main(args=None):
    rclpy.init(args=args)
    node = DataGenNode()
    node.get_logger().info('DataGenNode is running')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
