import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from semi_exp_clip import load_dataset, CLIP
from semi_exp_viz import *
import datetime

os.chdir(os.path.expanduser("~")+'/cmap')

def pca_color(df_p):
    rgba = df_p[['pca0', 'pca1', 'pca2']]/40+0.5
    rgba.columns = ['r','g', 'b']
    rgba['a'] = 1
    if rgba.max(axis=None) > 1 or rgba.min(axis=None) < 0:
        print('rgba max or min overflowed')
    return pd.concat([df_p, rgba], axis=1)

def import_vector(df, fn):
    with open(fn,"rb") as f:
        features = pickle.load(f)
    features = pd.DataFrame(features)
    df_f = pd.concat([df, features], axis=1)
    return df_f

class PCA:
    def __init__(self, profile=None, n=3):
        try:
            with open(profile, "rb") as f:
                _profile = pickle.load(f)
            # [self.mean, self.std, self.top_evec] = _profile
            [self.mean, self.std, self.top_evec, self.explained_variance_ratio_] = _profile
        except: print("No profile data, use .fit to fit data")

    def fit(self, data, fn=None, n=3):
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
        self.top_evec = sorted_eigenvectors[:, :n]
        self.data_pca = np.dot(vec_std, self.top_evec)
        # Calculate explained variance ratio
        total_variance = sum(self.eival)
        self.explained_variance_ratio_ = [eigval / total_variance for eigval in self.eival[self.ind][:n]]
        
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
        self.data_pca = pd.DataFrame(self.data_pca, columns=['pca0', 'pca1', 'pca2'])
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

if __name__ == '__main__':
    ### from scratch
    # Load data & CLIP
    rh, df, ids = load_dataset('RGBD_1', scale=None)
    clip = CLIP(model='ViT-B-32', overwrite=False)
    df_s, df_f = clip.encode_rh(rh, df, ids)
    # PCA
    pca = PCA()
    df_p = pca.fit(df_f)

    ### from existing pca profile
    # Load and select data & CLIP
    # rh, df, ids = load_dataset('RGBD_1', scale=None)
    # dfs = separate_session(df)
    # df = dfs[0]
    # df = df.sort_values("timestamp")
    # df = df.reset_index(drop=True)
    # ids = get_ids(df)
    # clip = CLIP(model='ViT-B-32', fn='s1')
    # df_s, df_f = clip.encode_rh(rh, df, ids)
    # # PCA
    # pca = PCA(profile='./result/pca_profile.pkl')
    # df_p = pca.transform(df_f)
    # df_p = pca_color(df_p)
    # plot_viewpoint(df_p, 5, color=True)

    # from PIL import Image
    # import cv2
    # cv2.namedWindow('img')
    # for id in ids:
    #     [img_f, _] = rh.get_RGBD_files(id)
    #     image = Image.open(img_f).convert('RGB')
    #     image = np.array(image)
    #     cv2.imshow('img', image) 
    #     key = cv2.waitKey(0)
    # cv2.destroyAllWindows()