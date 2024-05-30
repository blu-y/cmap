import sys
import os
sys.path.append(os.getcwd())
# from cmap.src.utils import CLIP
import matplotlib.pyplot as plt
import numpy as np
import PIL
import time
import torch
from open_clip import create_model_from_pretrained, get_tokenizer

class CLIP:
    def __init__(self, model='ViT-B-32'):
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
            # print(e)
            self.dim = 1

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
    
def plot(feature1, image1, feature2, image2):

    label = range(len(feature1))
    fig, ax = plt.subplots(2, 3, figsize=(16, 6))
    ax[0,0].imshow(image1)
    ax[0,0].axis('off')
    ax[1,0].bar(label, feature1)

    ax[0,1].imshow(image2)
    ax[0,1].axis('off')
    ax[1,1].bar(label, feature2)

    diff = np.array(feature1) - np.array(feature2)
    diff /= np.linalg.norm(diff)
    diff = diff.tolist()

    # Normalize diff
    ax[1,2].bar(label, diff)

    plt.tight_layout()
    plt.show()
    return diff


if __name__ == "__main__":
    # clip = CLIP(model='ViT-B-16-SigLIP')
    # clip = CLIP(model='ViT-B-32')
    # clip = CLIP(model='ViT-B-32-256')
    clip = CLIP(model='ViT-L-14-quickgelu')

    image1 = PIL.Image.open('./voxelmapper/images/240529_135724_37.png')
    image2 = PIL.Image.open('./voxelmapper/images/240529_135724_37.png_cropped.png')
    image3 = PIL.Image.open('./voxelmapper/images/240529_135724_37.png_cropped2.png')
    image4 = PIL.Image.open('./voxelmapper/images/240529_144052_00.png_cropped.png')

    start_time = time.time()
    feature1 = clip.encode_image(image1)
    elapsed_time1 = time.time() - start_time
    print(f"Elapsed time for feature1: {elapsed_time1} seconds")

    start_time = time.time()
    feature1 = clip.encode_image(image1)
    elapsed_time1 = time.time() - start_time
    print(f"Elapsed time for feature1: {elapsed_time1} seconds")

    start_time = time.time()
    feature2 = clip.encode_image(image2)
    elapsed_time2 = time.time() - start_time
    print(f"Elapsed time for feature2: {elapsed_time2} seconds")

    start_time = time.time()
    feature3 = clip.encode_image(image3)
    elapsed_time3 = time.time() - start_time
    print(f"Elapsed time for feature3: {elapsed_time3} seconds")

    start_time = time.time()
    feature4 = clip.encode_image(image4)
    elapsed_time4 = time.time() - start_time
    print(f"Elapsed time for feature4: {elapsed_time4} seconds")
    diff = plot(feature1, image1, feature2, image2)
    texts = ["asd", "fire extinguisher", "fire", "red", "test", "nothing"]
    # while True:
    #     text = input("Input text(nothing to end): ")
    #     if text == "":
    #         break
    #     else :
    #         texts.append(text)
    featuret = clip.encode_text(texts)
    similarity = clip.similarity([feature1, feature2, feature3, feature4, diff], featuret)
    print("Similarity between image1 and text: ")
    print(texts)
    print(similarity)

    