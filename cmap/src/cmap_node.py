#!/usr/bin/env python
from time import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
from PIL import Image as PIL
from datetime import datetime
from open_clip import create_model_from_pretrained, get_tokenizer
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge

class PCA:
    def __init__(self, n=3, profile='./cmap/src/default_pca_profile.pkl'):
        self.n = n
        self.columns = ['pca'+str(i) for i in range(self.n)]
        try:
            with open(profile, "rb") as f:
                _profile = pickle.load(f)
            # [self.mean, self.std, self.top_evec] = _profile
            [self.mean, self.std, self.top_evec, self.explained_variance_ratio_] = _profile
        except: print("No profile data, use .fit to fit data")

    def pca_color(self, df_p):
        rgba = df_p[['pca0', 'pca1', 'pca2']]/40+0.5
        rgba.columns = ['r','g', 'b']
        rgba['a'] = 1
        if rgba.max(axis=None) > 1 or rgba.min(axis=None) < 0:
            print('rgba max or min overflowed')
        df = pd.concat([df_p, rgba], axis=1)
        return df[0], df[1], df[2]

    def import_vector(self, df, fn):
        with open(fn,"rb") as f:
            features = pickle.load(f)
        features = pd.DataFrame(features)
        df_f = pd.concat([df, features], axis=1)
        return df_f

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

    def preprocess(self, df_f):
        # get vectors from df_f
        df_out = pd.DataFrame({col: df_f[col] for col in df_f.columns if isinstance(col, int)})
        return df_out

    def standardize(self, data):
        data = self.preprocess(data)
        return (data - self.mean) / self.std
    
    def transform(self, data):
        data_std = self.standardize(data)
        self.n_data_pca = np.dot(data_std, self.top_evec)
        self.n_data_pca = pd.DataFrame(self.n_data_pca, columns=['pca0', 'pca1', 'pca2'])
        print('min :', self.n_data_pca.min(axis=0))
        print('max :', self.n_data_pca.max(axis=0))
        return pd.concat([data, self.n_data_pca], axis=1)

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
    ### 슬램 노드가 pose를 퍼블리쉬하는데 그걸 받으면 그대로 저장
    ### 그걸 이제 이미지를 받으면 clip 인코딩하고
    def __init__(self) :
        super().__init__('image_subscriber')
        self.bridge = CvBridge() 
        self.image_sub = self.create_subscription(
            Image, '/oakd/rgb/preview/image_raw', self.image_cb, qos_profile_sensor_data)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/pose', self.pose_cb, 1)
        self.cmap_pub = self.create_publisher(
            MarkerArray, '/cmap_marker', 10)
        self.markers = MarkerArray()
        self.image = []
        self.pose = PoseWithCovarianceStamped()
        self.clip = CLIP()
        # self.pca = PCA()
        self.features = []
        self.k = 0

    def encode_image(self, image):
        return self.clip.encode_image(image)
    
    def encode_text(self, label):
        return self.clip.encode_text([label])

    def pose_cb(self, msg):
        self.pose = msg

    def keyframe_selection(self, image):
        # Keyframe selection logic required
        self.k += 1
        if self.k % 20 == 0: return True
        return False
    
    def header_to_time(self, header, str=True, to_int=False):
        t = header.stamp.sec + header.stamp.nanosec * 1e-9
        if str: t = datetime.fromtimestamp(t).strftime('%Y%m%d_%H%M%S_%f')
        if to_int: t = int(t)
        return t
    
    def get_rgb(self):
        # PCA reduction required
        # return self.pca.pca_color(self.pca.transform(self.image))
        return 1.0, 0.0, 0.0

    def create_marker(self, header):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.id = self.header_to_time(header, str=False, to_int=True)
        marker.pose = self.pose.pose.pose
        marker.scale.x = 0.1
        marker.scale.y = 0.2
        marker.scale.z = 0.01
        marker.color.r, marker.color.g, marker.color.b = self.get_rgb()
        marker.color.a = 1.0
        self.markers.markers.append(marker)
        self.cmap_pub.publish(self.markers)

    def cmap(self, header):
        if self.keyframe_selection(self.image):
            t = self.header_to_time(header, str=False)
            p, o = self.get_pose()
            f = self.encode_image(PIL.fromarray(self.image))
            self.features.append([t] + p + o + f)
            self.create_marker(header)

    def get_pose(self):
        p = self.pose.pose.pose.position
        o = self.pose.pose.pose.orientation
        return [p.x, p.y, p.z], [o.x, o.y, o.z, o.w]

    def image_cb(self, msg) :
        self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.cmap(msg.header)
        cv2.imshow('img', self.image)
        key = cv2.waitKey(1)
        if key == 13:
            filename = self.header_to_time(msg.header) + ".png"
            filename = os.path.join(os.getcwd(), 'cmap', 'images', filename)
            cv2.imwrite(filename, self.image)
            print('Image Saved')

def main(args=None) :
  rclpy.init(args=args)
  node = CMAP()
  rclpy.spin(node)
  node.destroy_node()
  rclpy.shutdown()

if __name__ == '__main__' :
  main()