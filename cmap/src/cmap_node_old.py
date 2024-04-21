#!/usr/bin/env python
from PIL import Image as PIL
from datetime import datetime
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from cv_bridge import CvBridge
from utils import PCA, CLIP
import os
import csv

class CMAP(Node):
    def __init__(self, profile='default_pca_profile.pkl'):
        super().__init__('cmap_node')
        self.bridge = CvBridge() 
        self.image_sub = self.create_subscription(
            Image, '/oakd/rgb/preview/image_raw', self.image_cb, qos_profile_sensor_data)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/pose', self.pose_cb, 1)
        self.cmap_pub = self.create_publisher(
            MarkerArray, '/cmap/markers', 10)
        self.cmap_goal_sub = self.create_subscription(
            String, '/cmap/goal', self.goal_cb, 1)
        self.goal_pub = self.create_publisher(
            PoseStamped, '/goal_pose', 1)
        self.markers = MarkerArray()
        self.image = []
        self.pose = PoseWithCovarianceStamped()
        self.folder = os.path.join(os.getcwd(), 'cmap')
        fn = os.path.join(self.folder, 'profiles')
        if not os.path.exists(fn): os.makedirs(fn)
        if type(profile) == str: 
            profile = os.path.join(fn, profile)
        self.pca = PCA(profile=profile)
        self.clip = CLIP()
        self.features = []
        self.k = 0

    def goal_cb(self, msg):
        text = msg.data
        text_encodings = self.encode_text(text)
        image_encodings = np.array(self.features)[:, 8:]
        idx = np.argmax(self.clip.similarity(image_encodings, text_encodings))
        t = int(datetime.fromtimestamp(self.features[idx][0]).strftime('%M%S%f'))/100
        self.get_logger().info('Goal: ' + msg.data + ' ID: ' + str(int(t)))
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = self.features[idx][1:4]
        goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w = self.features[idx][4:8]
        self.goal_pub.publish(goal)

    def encode_image(self, image):
        return self.clip.encode_image(image)
    
    def encode_text(self, label):
        return self.clip.encode_text([label])

    def pose_cb(self, msg):
        self.pose = msg

    def keyframe_selection(self, image):
        # Keyframe selection logic required
        self.k += 1
        if self.k % 90 == 0: return True
        return False
    
    def header_to_time(self, header, to_str=True, to_int=False):
        t = header.stamp.sec + header.stamp.nanosec * 1e-9
        if to_str: t = datetime.fromtimestamp(t).strftime('%Y%m%d_%H%M%S_%f')
        if to_int:
            t = int(datetime.fromtimestamp(t).strftime('%M%S%f'))/100
            t = int(t)
            self.get_logger().info('ID: ' + str(t))
        return t

    def get_rgb(self):
        # PCA reduction required
        f = self.features[-1][8:]
        return self.pca.pca_color(f)

    def create_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.id = self.header_to_time(self.header, to_str=False, to_int=True)
        marker.pose = self.pose.pose.pose
        marker.scale.x = 0.3
        marker.scale.y = 0.03
        marker.scale.z = 0.01
        [marker.color.r, marker.color.g, marker.color.b] = self.get_rgb()
        marker.color.a = 1.0
        self.markers.markers.append(marker)
        self.cmap_pub.publish(self.markers)

    def cmap(self):
        if self.keyframe_selection(self.image):
            t = self.header_to_time(self.header, to_str=False)
            p, o = self.get_pose()
            f = self.encode_image(PIL.fromarray(self.image))
            self.features.append([t] + p + o + f)
            self.create_marker()

    def get_pose(self):
        p = self.pose.pose.pose.position
        o = self.pose.pose.pose.orientation
        return [p.x, p.y, p.z], [o.x, o.y, o.z, o.w]

    def save_features(self):
        folder = os.path.join(self.folder, 'results')
        if not os.path.exists(folder): os.makedirs(folder)
        fn = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_features.csv'
        fn = os.path.join(folder, fn)
        with open(fn, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['time', 'px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow'] + ['f'+str(i) for i in range(self.clip.dim)])
            for row in self.features:
                writer.writerow(row)
        self.get_logger().info('Feature Saved ' + fn)

    def save_image(self):
        folder = os.path.join(self.folder, 'images', 'rgb')
        if not os.path.exists(folder): os.makedirs(folder)
        fn = str(self.header.stamp.sec + self.header.stamp.nanosec * 1e-9)  + ".png"
        fn = os.path.join(folder, fn)
        cv2.imwrite(fn, self.image)

    def image_cb(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.header = msg.header
        self.cmap()
        # self.save_image()
        cv2.imshow('img', self.image)
        key = cv2.waitKey(1)    
        if key == 13:
            # filename = self.header_to_time(msg.header) + ".png"
            # filename = os.path.join(self.folder, filename)
            # cv2.imwrite(filename, self.image)
            self.save_features()

def main(args=None) :
    rclpy.init(args=args)
    # profile = '2024-04-15_20-42-04_features_pca_profile.pkl'
    # profile = ['livingroom', 'kitchen', 'bathroom']
    node = CMAP(profile='default_pca_profile.pkl')
    node.get_logger().info('CMAP Node Running')
    node.get_logger().info('Press Enter to save features')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__' :
  main()