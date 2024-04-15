import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import torch
from open_clip import create_model_from_pretrained, get_tokenizer

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
