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
os.chdir(os.path.expanduser("~")+'/cmap')
model, preprocess = create_model_from_pretrained('ViT-B-16-SigLIP', pretrained='./ViT-B-16-SigLIP/open_clip_pytorch_model.bin')
tokenizer = get_tokenizer('ViT-B-16-SigLIP')

from robotathome import RobotAtHome
from robotathome import logger, log, set_log_level
from robotathome import time_win2unixepoch, time_unixepoch2win
from robotathome import get_labeled_img, plot_labeled_img

log.set_log_level('INFO')  # SUCCESS is the default
level_no, level_name = log.get_current_log_level()
print(f'Current log level name: {level_name}')
my_rh_path = '~/cmap/dataset'
my_rgbd_path = os.path.join(my_rh_path, 'files/rgbd')
my_scene_path = os.path.join(my_rh_path, 'files/scene')
my_wspc_path = '~/cmap/dataset/result'
try: 
      rh = RobotAtHome(my_rh_path, my_rgbd_path, my_scene_path, my_wspc_path)
except:
      logger.error("Something was wrong")

df = rh.get_sensor_observations('rgbdlsr')
df = df.sort_values("timestamp")
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
# print(df.head())
df = df.loc[df.sensor_name=="RGBD_1"]
df = df.reset_index(drop=True)
ids = df.id.tolist()
# print(df.head())
# print(df.info())


################ For debug ###############
# df = df[df.index % 100 == 0]             #
# df = df.reset_index(drop=True)           #
# ids = df.id.tolist()                     #
##########################################

feature_path = './result/'
labels_list = ["a shampoo", "bathroom", "a stove", "kitchen", "a television", "livingroom"]
overwrite = True

feature_file = os.path.join(feature_path, 'semi_exp.pkl')
if not(os.path.exists(feature_file)) or overwrite:
    start = time()
    features = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for id in tqdm(ids):
            try: 
                [img_f, _] = rh.get_RGBD_files(id)
                image = Image.open(img_f)
                image = preprocess(image).unsqueeze(0)
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features[0].tolist()
            except:
                image_features = [0] * 768
            features.append(image_features)
    print('average time: ', (time()-start)/len(ids))

    if not os.path.exists('./result'): os.makedirs('./result')
    with open(feature_file,"wb") as f:
        pickle.dump(features, f)
else: 
    with open(feature_file,"rb") as f:
        features = pickle.load(f)

image_features = np.array(features)
features = pd.DataFrame(features)
df_f = pd.concat([df, features], axis=1)

text = tokenizer(labels_list, context_length=model.context_length)
with torch.no_grad(), torch.cuda.amp.autocast(): text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity =  image_features @ text_features.numpy().T
similarity = pd.DataFrame(similarity, columns=labels_list)

df_s = pd.concat([df, similarity], axis=1)
df_f = pd.concat([df_s, features], axis=1)

for l in labels_list:
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
plt.show()