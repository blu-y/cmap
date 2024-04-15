from utils import PCA, CLIP
from glob import glob
from time import time
import PIL
import os
from datetime import datetime
import csv
from tqdm import tqdm

# src is folder(with images) or csv file(embeddings)
src = '/home/iram/images/rgb_04082250/'
# src = './cmap/results/2024-04-03_15-29-04_features.csv'
model = 'ViT-B-32'

if src[-1] == '/' or src[-4:] == '.csv':
    if src[-1] == '/':
        image_list = glob(src+'*.png')
        image_list.sort()
        clip = CLIP(model=model)
        features = []
        for file in tqdm(image_list):
            t = file.split('/')[-1][:-4]
            image = PIL.Image.open(file)
            f = clip.encode_image(image)
            p = [0, 0, 0]
            o = [0, 0, 0, 1]
            features.append([t] + p + o + f)
        # save features
        fn = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_features.csv'
        fn = os.path.join('./cmap/results', fn)
        with open(fn, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['time', 'px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow'] + ['f'+str(i) for i in range(clip.dim)])
            for row in features:
                writer.writerow(row)
    src = fn
    pca = PCA()
    profile = pca.fit(fn=src)
else:
    print('Invalid source')

