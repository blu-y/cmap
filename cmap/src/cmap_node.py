#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge
from time import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
from PIL import Image as PIL
from datetime import datetime
from open_clip import create_model_from_pretrained, get_tokenizer
from geometry_msgs.msg import PoseWithCovarianceStamped

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

class ImageSubscriber(Node) :
    def __init__(self) :
        super().__init__('image_subscriber')
        self.bridge = CvBridge() 
        self.image_sub = self.create_subscription(
        Image, '/oakd/rgb/preview/image_raw', self.image_cb, qos_profile_sensor_data)
        self.pose_sub = self.create_subscription(
        PoseWithCovarianceStamped, '/pose', self.pose_cb, 1)
        
        self.image = []
        self.pose = PoseWithCovarianceStamped()
        self.clip = CLIP()
        self.features = []

    def encode_image(self, image):
        return self.clip.encode_image(image)
    
    def encode_text(self, label):
        return self.clip.encode_text([label])

    def pose_cb(self, msg):
        self.pose = msg

    def keyframe_selection(self, image):
        # Keyframe selection logic required
        return True
    
    def header_to_time(self, header, str=True):
        t = header.stamp.sec + header.stamp.nanosec * 1e-9
        if str: t = datetime.fromtimestamp(t).strftime('%Y%m%d_%H%M%S_%f')
        return t

    def get_pose(self):
        p = self.pose.pose.pose.position
        o = self.pose.pose.pose.orientation
        return [p.x, p.y, p.z], [o.x, o.y, o.z, o.w]

    def image_cb(self, msg) :
        self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv2.imshow('img', self.image)
        key = cv2.waitKey(1)
        if self.keyframe_selection(self.image):
            t = self.header_to_time(msg.header, str=False)
            p, o = self.get_pose()
            f = self.encode_image(PIL.fromarray(self.image))
            self.features.append(t + p + o + f)
        if key == 13:
            filename = self.header_to_time(msg.header) + ".png"
            filename = os.path.join(os.getcwd(), 'cmap', 'images', filename)
            cv2.imwrite(filename, self.image)
            print('Image Saved')

def main(args=None) :
  rclpy.init(args=args)
  node = ImageSubscriber()
  rclpy.spin(node)
  node.destroy_node()
  rclpy.shutdown()

if __name__ == '__main__' :
  main()