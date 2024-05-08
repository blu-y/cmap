import pandas as pd
import matplotlib.pyplot as plt
from time import time
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
import os
from robotathome import RobotAtHome, logger, log

os.chdir(os.path.expanduser("~")+'/cmap')

def separate_session(df):
    hsi = df.home_session_id.unique()
    hsi.sort()
    dfs = []
    for i in hsi:
        _df = df.loc[df.home_session_id==i]
        _df = _df.loc[df.sensor_name=="RGBD_1"]
        _df = _df.sort_values(by='room_id')
        _df = _df.reset_index(drop=True)
        counts = _df.room_id.value_counts()
        dfs.append(_df)
    return dfs

def load_rh():
    log.set_log_level('INFO')  # SUCCESS is the default
    level_no, level_name = log.get_current_log_level()
    print(f'Current log level name: {level_name}')
    my_rh_path = '~/cmap/dataset'
    my_rgbd_path = os.path.join(my_rh_path, 'files/rgbd')
    my_scene_path = os.path.join(my_rh_path, 'files/scene')
    my_wspc_path = '~/cmap/dataset/result'
    try: 
        rh = RobotAtHome(my_rh_path, my_rgbd_path, my_scene_path, my_wspc_path)
        return rh
    except:
        logger.error("Something was wrong")

def load_dataset(sensor=None, scale=None):
    rh = load_rh()
    df = rh.get_sensor_observations('rgbdlsr')
    df = df.sort_values("timestamp")
    df = df.reset_index(drop=True)
    if sensor != None:
        # select RGBD1 camera
        df = df.loc[df.sensor_name==sensor]
        df = df.reset_index(drop=True)
    if scale != None:
        df = df[df.index % scale == 0]
        df = df.reset_index(drop=True)
    ids = get_ids(df)
    return rh, df, ids

def get_ids(df):
    ids = df.id.tolist()
    return ids

class CLIP:
    def __init__(self, model='ViT-B-16-SigLIP',
                 feature_path = './result/',
                 overwrite=False, fn=''):
        # model, preprocess = create_model_from_pretrained('ViT-B-16-SigLIP', pretrained='./ViT-B-16-SigLIP/open_clip_pytorch_model.bin')
        # tokenizer = get_tokenizer('ViT-B-16-SigLIP')
        # model, preprocess = create_model_from_pretrained('ViT-B-32-256', pretrained='./ViT-B-32-256/open_clip_pytorch_model.bin')
        # tokenizer = get_tokenizer('ViT-B-32-256')
        pt = './'+model+'/open_clip_pytorch_model.bin'
        if torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            self.cuda = False
            self.device = torch.device('cpu')
        self.model, self.preprocess = create_model_from_pretrained(model, pretrained=pt, device=self.device)
        self.tokenizer = get_tokenizer(model)
        try: self.dim = self.model.positional_embedding.size()[1]
        except Exception as e:
            print(e)
            self.dim = 1
        self.fn = 'exp_'+model+fn+'.pkl'
        self.feature_file = os.path.join(feature_path, self.fn)
        self.overwrite = overwrite

    def encode_rh(self, rh, df, ids,
                  label_list = []):
        if not(os.path.exists(self.feature_file)) or self.overwrite:
            start = time()
            features = []
            for id in tqdm(ids):
                [img_f, _] = rh.get_RGBD_files(id)
                image = Image.open(img_f)
                image_features = self.encode_image(image)
                features.append(image_features)
            print('average time: ', round((time()-start)/len(ids), 4))
            print('output dimension:', len(image_features))
            if not os.path.exists('./result'): os.makedirs('./result')
            with open(self.feature_file,"wb") as f:
                pickle.dump(features, f)
        else: 
            with open(self.feature_file,"rb") as f:
                features = pickle.load(f)
                
        image_features = np.array(features)
        features = pd.DataFrame(features)
        df_f = pd.concat([df, features], axis=1)

        text_features = self.encode_text(label_list = label_list)
    
        similarity = self.similarity(image_features, text_features)
        similarity = pd.DataFrame(similarity, columns=label_list)

        df_s = pd.concat([df, similarity], axis=1)
        df_f = pd.concat([df_s, features], axis=1)
        return df_s, df_f

    def encode_image(self, image):
        with torch.no_grad(), torch.cuda.amp.autocast():
            try: 
                image = self.preprocess(image).unsqueeze(0)
                if self.cuda: image = image.to(self.device, dtype=torch.float16)
                image_features = self.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features[0].tolist()
            except Exception as e:
                print(e)
                image_features = [0.0] * self.dim
                pass
        return image_features

    def encode_text(self, label_list):
        text = self.tokenizer(label_list, context_length=self.model.context_length).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast(): text_features = self.model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def similarity(self, image_features, text_features):
        if self.cuda: return image_features @ text_features.cpu().numpy().T
        else: return image_features @ text_features.numpy().T
    
def plot(df_s, label_list, show=True):
    for l in label_list:
        _df= df_s.sort_values(l, ascending=False)
        _df = _df.reset_index(drop=True)
        plt.figure(l, figsize=(8,6))
        for i in range(8):
            [img_f, _] = rh.get_RGBD_files(_df['id'][i])
            img = Image.open(img_f)
            plt.subplot(2,4,i+1)
            plt.imshow(img)
            plt.title(round(_df[l][i],3))
            plt.axis('off')
        plt.tight_layout()
    if show:
        plt.show()

if __name__ == '__main__':
    rh, df, ids = load_dataset('RGBD_1', scale=1)

    # for particular session
    dfs = separate_session(df)
    df = dfs[0]
    df = df.sort_values("timestamp")
    df = df.reset_index(drop=True)
    ids = get_ids(df)

    # clip = CLIP(model='ViT-B-16-SigLIP', overwrite=True)
    clip = CLIP(model='ViT-B-32-256', overwrite=True)
    labels = ["a shampoo", "bathroom", "a stove", "kitchen", "a television", "livingroom"]
    df_s, df_f = clip.encode_rh(rh, df, ids, label_list=labels)
    plot(df_s, label_list = labels, show=True)