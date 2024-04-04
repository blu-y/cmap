#!/usr/bin/env python
from time import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import csv
import torch
from PIL import Image as PIL
from datetime import datetime
from open_clip import create_model_from_pretrained, get_tokenizer
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

class PCA:
    def __init__(self, n=3, profile='default_pca_profile.pkl'):
        self.n = n
        self.columns = ['pca'+str(i) for i in range(self.n)]
        try:
            with open(profile, "rb") as f:
                _profile = pickle.load(f)
            # [self.mean, self.std, self.top_evec] = _profile
            [self.mean, self.std, self.top_evec, self.explained_variance_ratio_] = _profile
        except: print("No profile data, use .fit to fit data")

    def pca_color(self, data_pca):
        rgb = data_pca/30+0.5
        rgb = np.clip(rgb, 0, 1)
        return rgb

    def fit(self, data, fn=None):
        data_pca = None
        vec = self.preprocess(data)
        vec = vec.to_numpy()
        self.mean = np.mean(vec, axis=0)
        self.std = np.std(vec, axis=0)
        vec_std = (vec - self.mean) / self.std
        self.cov = np.cov(vec_std, rowvar=False)
        self.eival, self.eivec = np.linalg.eig(self.cov)
        self.ind = np.argsort(self.eival)[::-1]
        sorted_eigenvectors = self.eivec[:, self.ind]
        self.top_evec = sorted_eigenvectors[:, :self.n]
        self.data_pca = np.dot(vec_std, self.top_evec)
        # Calculate explained variance ratio
        total_variance = sum(self.eival)
        self.explained_variance_ratio_ = [eigval / total_variance for eigval in self.eival[self.ind][:self.n]]
        
        if fn is None:
            if not os.path.exists('./result'): os.makedirs('./result')
            t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fn='./result/pca_profile_'+str(t)+'.pkl'
            with open(fn,"wb") as f:
                pickle.dump([self.mean, self.std, self.top_evec, self.explained_variance_ratio_], f)
                # pickle.dump([self.mean, self.std, self.top_evec], f)
            print('Profile saved at :', fn)
        else: 
            with open(fn,"wb") as f:
                pickle.dump([self.mean, self.std, self.top_evec, self.explained_variance_ratio_], f)
                # pickle.dump([self.mean, self.std, self.top_evec], f)
        self.data_pca = pd.DataFrame(self.data_pca, columns=self.columns)
        print('min :', self.data_pca.min(axis=0))
        print('max :', self.data_pca.max(axis=0))
        print('Explained variance ratio:', self.explained_variance_ratio_)
        return pd.concat([data, self.data_pca], axis=1)
    
    def transform(self, data):
        mean = np.mean(data)
        std = np.std(data)
        data_std = [(x - mean) / std for x in data]
        data_pca = np.dot(data_std, self.top_evec)
        return data_pca

class CLIP:
    def __init__(self, model='ViT-B-32'):
        pt = './'+model+'/open_clip_pytorch_model.bin'
        self.model, self.preprocess = create_model_from_pretrained(model, pretrained=pt)
        self.tokenizer = get_tokenizer(model)
        try: self.dim = self.model.positional_embedding.size()[1]
        except Exception as e:
            print(e)
            self.dim = 1

    def encode_image(self, image):
        with torch.no_grad(), torch.cuda.amp.autocast():
            try: 
                image = self.preprocess(image).unsqueeze(0)
                image_features = self.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features[0].tolist()
            except Exception as e:
                print(e)
                image_features = [0.0] * self.dim
                pass
        return image_features

    def encode_text(self, label_list):
        text = self.tokenizer(label_list, context_length=self.model.context_length)
        with torch.no_grad(), torch.cuda.amp.autocast(): text_features = self.model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def similarity(self, image_features, text_features):
        return image_features @ text_features.numpy().T

class CMAP(Node) :
    def __init__(self) :
        super().__init__('cmap_node')
        self.bridge = CvBridge() 
        self.image_sub = self.create_subscription(
            Image, '/oakd/rgb/preview/image_raw', self.image_cb, qos_profile_sensor_data)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/pose', self.pose_cb, 1)
        self.cmap_pub = self.create_publisher(
            MarkerArray, '/cmap/marker', 10)
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
        fn = os.path.join(fn, 'default_pca_profile.pkl')
        self.clip = CLIP()
        self.pca = PCA(profile=fn)
        self.features = []
        self.k = 0

    def goal_cb(self, msg):
        self.get_logger().info('Goal: ' + msg.data)
        text = msg.data
        text_encodings = self.encode_text(text)
        image_encodings = np.array(self.features)[:, 8:]
        idx = np.argmax(self.clip.similarity(image_encodings, text_encodings))
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
        if self.k % 50 == 0: return True
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
        return self.pca.pca_color(self.pca.transform(f))

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

    def image_cb(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.header = msg.header
        self.cmap()
        cv2.imshow('img', self.image)
        key = cv2.waitKey(1)
        if key == 13:
            # filename = self.header_to_time(msg.header) + ".png"
            # filename = os.path.join(self.folder, filename)
            # cv2.imwrite(filename, self.image)
            self.save_features()

def main(args=None) :
  rclpy.init(args=args)
  node = CMAP()
  node.get_logger().info('CMAP Node Running')
  node.get_logger().info('Press Enter to save features')
  rclpy.spin(node)
  node.destroy_node()
  rclpy.shutdown()

if __name__ == '__main__' :
  main()