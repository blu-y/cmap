import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import torch
from open_clip import create_model_from_pretrained, get_tokenizer
import csv

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
