import os
import sys
sys.path.append(os.getcwd())
import glob
import rclpy
import rclpy.logging
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, PointField, Image
from tf2_ros.buffer import Buffer
from tf2_geometry_msgs import do_transform_point # to transform
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from math import cos, sin, inf
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
import cv2
from utils import *
import time
import PIL.Image

class CMAPNode(Node):
    def __init__(self, camera=0, model='ViT-B-16-SigLIP', leaf_size=0.25):
        '''
        camera: int (if usb camera, ex) 0, 1, 2, ...)
                str (if topic, ex) '/camera/image_raw')
        '''
        super().__init__('cmap')
        self.set_camera(camera)
        self.get_logger().info(f'Camera set to {self.camera}, {camera}')
        self.get_logger().info(f'CLIP model initializing to {model}')
        self.clip = CLIP(model)
        self.get_logger().info(f'CLIP model initialized')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.scan = LaserScan()
        self.leaf_size = leaf_size
        # HFOV 67.983 deg
        self.scan_from = -103
        self.scan_to = 101
        self.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]
        self.last_scan_time = time.time()
        self.last_frame_time = 0.0
        self.scan_subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 1)
        self.scan_pub_L = self.create_publisher(PointCloud2, '/scan_L', 1)
        self.scan_pub_M = self.create_publisher(PointCloud2, '/scan_M', 1)
        self.scan_pub_R = self.create_publisher(PointCloud2, '/scan_R', 1)
        self.scan_subscription
    
    def set_camera(self, camera):
        if isinstance(camera, int):
            self.get_logger().info(f'Using USB camera {camera}')
            self.camera = 'usb'
            self.cap = Camera(camera)
        elif isinstance(camera, str):
            if os.path.exists(os.path.expanduser(camera)):
                self.get_logger().info(f'Using local images {camera}')
                self.camera = 'local'
                self.image_list = glob.glob(os.path.join(camera, '*.png'))
                self.image_list.sort()
                self.get_logger().info(f'Found {len(self.image_list)} images')
                self.frame = PIL.Image.open(self.image_list[0])
                self.last_utc = make_utc(self.image_list[0])
            else:
                self.get_logger().info(f'Using ROS camera topic {camera}')
                self.camera = 'ros'
                self.bridge = CvBridge() 
                self.camera_subscription = self.create_subscription(
                    Image,
                    camera,
                    self.camera_callback,
                    1
                )
                self.camera_subscription

    def get_frame(self, stamp=time.time()):
        if self.camera == 'usb':
            self.frame = self.cap.getFrame()
        elif self.camera == 'local':
            stamp = stamp.sec + stamp.nanosec * 1e-9
            while True:
                curr_ = make_utc(self.image_list[0])
                next_ = make_utc(self.image_list[1])
                if next_ > stamp: break
                self.image_list.pop(0)
            self.frame = PIL.Image.open(self.image_list[0])
            stamp = curr_
        return self.frame, stamp

    def scan_callback(self, msg):
        self.scan = msg
        self.transform_scan()

    def camera_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def divide(self, points):
        size = len(points)//3
        return points[:size], points[size:size*2], points[size*2:]

    def voxelize(self, points):
        array = np.array(points, dtype=np.float32)
        quantized = np.round(array / self.leaf_size).astype(int)
        unique_voxels = np.unique(quantized, axis=0)
        data = unique_voxels * self.leaf_size
        return data

    def transform_scan(self):
        if not self.clip.available: return
        try:
            scan = self.scan
            frame, stamp = self.get_frame(scan.header.stamp)
            if stamp == self.last_frame_time: return
            self.last_frame_time = stamp
            if self.scan_from < 0:
                scan.ranges = scan.ranges[self.scan_from:] + scan.ranges[:self.scan_to]
            else: scan.ranges = scan.ranges[self.scan_from:self.scan_to]
            size = len(scan.ranges)//3
            # transform = self.tf_buffer.lookup_transform('map', 'base_link', scan.header.stamp)
            sr, sm, sl = self.divide(scan.ranges)
            p = []
            for scans, j in zip([sr, sm, sl],[0, size, size*2]):
                points = []
                for i in range(len(scans)):
                    if scans[i] == inf or scans[i] == 'nan' or scans[i] < 0.2:
                        continue
                    angle = scan.angle_min + (i+j+self.scan_from) * scan.angle_increment
                    x = scans[i] * cos(angle)
                    y = scans[i] * sin(angle)
                    points_ = PointStamped()
                    points_.header = scan.header
                    points_.point.x = x
                    points_.point.y = y
                    points_ = self.tf_buffer.transform(points_, 'map')
                    points.append([points_.point.x, points_.point.y, 0])
                    # 여기서 카메라 각도 안에 있는 것들도 필터해야함
                p.append(points)
            header = scan.header
            header.frame_id = 'map'
            vr = self.voxelize(p[0])
            vm = self.voxelize(p[1])
            vl = self.voxelize(p[2])
            self.scan_pub_R.publish(pc2.create_cloud(header, self.fields, vr))
            self.scan_pub_M.publish(pc2.create_cloud(header, self.fields, vm))
            self.scan_pub_L.publish(pc2.create_cloud(header, self.fields, vl))
            self.get_logger().debug(f"Published {vr.shape[0]}, {vm.shape[0]}, {vl.shape[0]} points in map frame, {1/(time.time()-self.last_scan_time):.3f} fps")
            self.last_scan_time = time.time()
        except Exception as e:
            self.get_logger().warn(f'{e}')

def main(args=None):
    rclpy.init(args=args)
    rclpy.logging.set_logger_level('cmap', rclpy.logging.LoggingSeverity.DEBUG)
    node = CMAPNode(camera='./athirdmapper/images/images')
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()