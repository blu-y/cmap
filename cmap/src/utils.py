import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import torch
from open_clip import create_model_from_pretrained, get_tokenizer
import csv
import cv2
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from tf_transformations import euler_from_quaternion

class KFS:
    def __init__(self, radius, resolution, angle, n=0.5, d=2):
        self.make_default_mask(radius, resolution, angle)
        self.make_weight_matrix(n, d)

    def make_weight_matrix(self, n, d):
        # Create a coordinate grid
        m = n/d
        k = n-n*np.log(d)
        Y, X = np.ogrid[:self.size*2, :self.size*2]
        # Compute the distance squared from the origin
        distance_squared = (X - self.size)**2 + (Y - self.size*2)**2
        distance = np.sqrt(distance_squared)
        # Calculate weights based on the distance
        weights = (self.resolution * distance)**n * np.exp(-m * self.resolution * distance + k)
        self.weights = weights

    def weighted_sum(self, 
                     pose1:PoseWithCovarianceStamped, 
                     pose2:PoseWithCovarianceStamped,
                     map1:OccupancyGrid, 
                     map2:OccupancyGrid):
        pose1 = pose1.pose.pose
        pose2 = pose2.pose.pose
        rpy1 = euler_from_quaternion([pose1.orientation.x, 
                                      pose1.orientation.y, 
                                      pose1.orientation.z, 
                                      pose1.orientation.w])
        rpy2 = euler_from_quaternion([pose2.orientation.x, 
                                      pose2.orientation.y, 
                                      pose2.orientation.z, 
                                      pose2.orientation.w])
        diff = ((pose2.position.x - pose1.position.x)/self.resolution, 
                (pose2.position.y - pose1.position.y)/self.resolution)
        mask1, mask3 = self.get_intersection(diff, rpy1[2], rpy2[2])
        # cv2.imshow('mask', mask1*255)
        # cv2.waitKey(1)
        # cv2.imshow('mask_inter', mask3*255)
        # cv2.waitKey(1)
        w1 = self.weights[self.mask_r:self.mask_r*3, 
                          self.mask_r:self.mask_r*3]
        w3 = self.weights[self.mask_r-round(diff[0]):self.mask_r*3-round(diff[1]), 
                          self.mask_r-round(diff[0]):self.mask_r*3-round(diff[1])]
        ## add map data
        map = np.array(map1.data).reshape(map1.info.height, map1.info.width)
        map = np.pad(map, self.mask_r, 'constant', constant_values=0)
        x = int((pose1.position.x - map1.info.origin.position.x) / map1.info.resolution)
        y = int((pose1.position.y - map1.info.origin.position.y) / map1.info.resolution)
        # print(map.shape)
        # print(y, x, y+self.mask_r, x+self.mask_r)
        map = map[y:y+2*self.mask_r, x:x+2*self.mask_r]
        # map = np.flip(map, 0)
        # cv2.imshow('localmap', map)
        # cv2.waitKey(1)
        map[map != -1] = 1
        map[map == -1] = 0
        if w3.shape != mask3.shape: 
            w3 = np.zeros(mask3.shape)
        return np.sum(mask1*w1*map)-np.sum(mask3*w3*map)

    def make_default_mask(self, radius, resolution, angle):
        self.radius = radius
        self.resolution = resolution
        self.angle = angle
        self.mask_r= int(self.radius / self.resolution)
        self.size = self.mask_r * 2
        self.center = (self.mask_r, self.mask_r)
        self.mask = self.circular_sector_mask()
                                         
    def circular_sector_mask(self):
        # Initialize an empty mask
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        # Define the sector
        angle = self.angle // 2
        sector_points = [self.center]
        for angle in range(-angle, angle+1):
            x = int(self.center[0] + self.mask_r * np.cos(np.deg2rad(angle)))
            y = int(self.center[1] + self.mask_r * np.sin(np.deg2rad(angle)))
            sector_points.append((x, y))
        sector_points = np.array([sector_points], dtype=np.int32)
        # Fill the sector
        cv2.fillPoly(mask, [sector_points], 1)
        return mask

    def transform_mask(self, new_center, rotation_angle):
        rotation_angle = int(np.rad2deg(rotation_angle))
        # Calculate transformation matrix for rotation and translation
        rows, cols = self.mask.shape
        M = cv2.getRotationMatrix2D((cols // 2, rows // 2), rotation_angle, 1)
        rotated_mask = cv2.warpAffine(self.mask, M, (cols, rows))
        # Translate mask to new center
        translation_matrix = np.float32([[1, 0, new_center[0] - cols // 2], [0, 1, new_center[1] - rows // 2]])
        translated_mask = cv2.warpAffine(rotated_mask, translation_matrix, (cols, rows))
        return translated_mask

    def get_intersection(self, diff, yaw1, yaw2):
        mask1 = self.transform_mask(self.center, yaw1)
        mask2 = self.transform_mask(self.center+diff, yaw2)
        return mask1, mask1 & mask2

class PCA:
    def __init__(self, n=3, profile='default_pca_profile.pkl', model='ViT-B-32'):
        self.n = n
        self.columns = ['pca'+str(i) for i in range(self.n)]
        self.profile_ = []
        if type(profile) == str: self.load_profile(profile)
        elif type(profile) == list: self.profile_text(profile, model)

    def load_profile(self, profile):
        try:
            with open(profile, "rb") as f:
                _profile = pickle.load(f)
            # [self.mean, self.std, self.top_evec] = _profile
            [self.mean, self.std, self.top_evec, self.evr_] = _profile
        except: print("No profile data, use .fit to fit data")
    
    def profile_text(self, profile, model):
        self.clip = CLIP(model=model)
        for p in profile:
            self.profile_.append(self.clip.encode_text([p]))

    def pca_color(self, data):
        if len(self.profile_) > 0:
            r = self.clip.similarity(data, self.profile_[0])[0]
            g = self.clip.similarity(data, self.profile_[1])[0]
            b = self.clip.similarity(data, self.profile_[2])[0]
            rgb = [r, g, b]
            print(rgb)
        else: 
            data_pca = self.transform(data)
            rgb = data_pca/15+0.5
        rgb = np.clip(rgb, 0, 1)
        return rgb

    def fit(self, fn):
        with open(fn, 'r') as file:
            reader = csv.reader(file)
            data_list = list(reader)
        data_list.pop(0)
        data = np.array(data_list).astype(float)
        vec = data[:,8:]
        self.mean = np.mean(vec, axis=0)
        self.std = np.std(vec, axis=0)
        vec_std = (vec - self.mean) / self.std
        self.cov = np.cov(vec_std, rowvar=False)
        self.eival, self.eivec = np.linalg.eig(self.cov)
        self.eival = np.real(self.eival)
        self.eivec = np.real(self.eivec)
        self.ind = np.argsort(self.eival)[::-1]
        sorted_eigenvectors = self.eivec[:, self.ind]
        self.top_evec = sorted_eigenvectors[:, :self.n]
        self.data_pca = np.dot(vec_std, self.top_evec)
        # Calculate explained variance ratio
        total_variance = sum(self.eival)
        self.evr_ = [eigval / total_variance for eigval in self.eival[self.ind][:self.n]]
        profile = [self.mean, self.std, self.top_evec, self.evr_]
        print('min :', np.min(self.data_pca, axis=0))
        print('max :', np.max(self.data_pca, axis=0))
        print('Explained variance ratio: %.3f, %.3f, %.3f'
              %(self.evr_[0], self.evr_[1], self.evr_[2]))
        print('Total: %.3f' %np.sum(self.evr_))
        fn = os.path.split(fn)[-1]
        out_fn = './cmap/profiles/' + fn[:-4]+'_pca_profile.pkl'
        with open(out_fn,"wb") as f:
            pickle.dump(profile, f)
            print('Profile saved at :', out_fn)
        self.data_pca = pd.DataFrame(self.data_pca, columns=self.columns)
        return profile

    def preprocess(self, array):
        # get vectors from df_f
        array_out = array
        return array_out

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
