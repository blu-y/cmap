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
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
from utils import PCA, CLIP, KFS
import os
import csv

class CMAP(Node):
    def __init__(self, profile='default_pca_profile.pkl', radius=10, resolution=0.05, angle=60):
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
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_cb, 1)
        self.markers = MarkerArray()
        self.image = []
        self.pose = None
        self.pose_kf = None
        self.map = None
        self.map_kf = None
        self.folder = os.path.join(os.getcwd(), 'cmap')
        fn = os.path.join(self.folder, 'profiles')
        if not os.path.exists(fn): os.makedirs(fn)
        if type(profile) == str: 
            profile = os.path.join(fn, profile)
        self.pca = PCA(profile=profile)
        self.clip = CLIP()
        self.radius = 5
        self.angle = 40
        self.kfs = KFS(radius, resolution, angle)
        self.k = 500
        self.k_ = 0
        self.features = []
        self.k_t = []

    def map_cb(self, msg):
        self.map = msg

    def pose_cb(self, msg):
        self.pose = msg

    def goal_cb(self, msg):
        text = msg.data
        text_encodings = self.encode_text(text)
        image_encodings = np.array(self.features)[:, 8:]
        idx = np.argmax(self.clip.similarity(image_encodings, text_encodings))
        t = int(datetime.fromtimestamp(self.features[idx][0]).strftime('%M%S%f'))/100
        self.get_logger().info('Goal: ' + msg.data + ' ID: ' + str(int(t)))
        ### debug ###
        # self.get_logger().info('Goal: ' + msg.data + ' ID: ' + str(self.features[idx][0]))
        # fn = os.path.join('/home/iram/images', 'rgb_04082250', str(self.features[idx][0]) + '.png')
        # im = cv2.imread(fn)
        # fd = '/home/iram/images/exp49/'
        # if not os.path.exists(fd): os.makedirs(fd)
        # cv2.imwrite(fd+text+'.png', im)
        ### debug ###
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

    def keyframe_selection(self):
        # Keyframe selection logic required
        if self.map is None: return False
        if self.pose is None: return False
        if self.map_kf==None or self.pose_kf== None:
            self.map_kf = self.map
            self.pose_kf = self.pose
            print('Initial Keyframe')
            return True
        # self.k_ += 1
        # if self.k_ % 60 == 0: 
        #     self.k_ = 0
        #     self.pose_kf = self.pose
        #     self.map_kf = self.map
        #     return True
        if abs(self.header.stamp.sec - self.pose.header.stamp.sec) > 3:
            print(self.header.stamp.sec, self.pose.header.stamp.sec)
            self.pose_kf = self.pose
            self.map_kf = self.map
            self.pose.header.stamp = self.header.stamp
            print('Pose Time Mismatch')
            return True
        if self.pose == self.pose_kf: return False
        k_t = self.kfs.weighted_sum(self.pose, self.pose_kf, self.map, self.map_kf)
        print('K value:', k_t)
        if abs(k_t) > self.k:
            self.k_ = 0
            self.k_t.append(k_t)
            # cv2.imshow('mask', mask)
            # cv2.waitKey(1)
            self.pose_kf = self.pose
            self.map_kf = self.map
            return True
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
        if self.keyframe_selection():
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

    def save_k_t(self):
        folder = os.path.join(self.folder, 'results')
        if not os.path.exists(folder): os.makedirs(folder)
        fn = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_k_t.csv'
        fn = os.path.join(folder, fn)
        with open(fn, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['k_t'])
            for row in self.k_t:
                writer.writerow([row])
        self.get_logger().info('k_t Saved ' + fn)

def main(args=None) :
    rclpy.init(args=args)
    profile = '2024-04-15_20-42-04_features_pca_profile.pkl'
    # profile = 'default_pca_profile.pkl'
    # profile = ['livingroom', 'kitchen', 'bathroom']
    try:
        node = CMAP(profile=profile)
        node.get_logger().info('CMAP Node Running')
        node.get_logger().info('Press Enter to save features')
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        node.save_features()
        node.save_k_t()
        node.destroy_node()
        rclpy.shutdown()
    except Exception as e:
        print(e)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__' :
  main()