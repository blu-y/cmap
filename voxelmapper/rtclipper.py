import sys
import os
sys.path.append(os.getcwd())
# from cmap.src.utils import CLIP
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import time
import torch
from open_clip import create_model_from_pretrained, get_tokenizer

class CLIP:
    def __init__(self, model='ViT-B-32'):
        pt = './models/'+model+'/open_clip_pytorch_model.bin'
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

def textlist():
    texts = []
    while True:
        text = input("Input text(nothing to end): ")
        if text == "": 
            if len(texts) == 0:
                texts = ['desk', 'chair', 'white board', 'office', 'food', 'umbrella', 'fire extinguisher']
            break
        else : texts.append(text)
    print("Texts list:", texts)
    return texts

def graphs(height, texts, similarity, fps):
    g = np.zeros((height, height//2, 3), dtype=np.uint8)
    for i, text in enumerate(texts):
        g = cv2.putText(g, text, (0, (2*(i+1))*height//(2*(len(texts)+2))), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        g = cv2.putText(g, str(round(similarity[i],5)), (height//5, (2*(i+1)+1)*height//(2*(len(texts)+2))), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    g = cv2.putText(g, f"FPS: {fps:.2f}", (0, height-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return g

if __name__ == "__main__":
    model = 'ViT-B-16-SigLIP' # 'ViT-B-32', 'ViT-B-32-256', 'ViT-B-16-SigLIP', 'ViT-L-14-quickgelu'
    resolution = [1024, 576]

    texts = textlist()
    clip = CLIP(model=model)
    text_f = clip.encode_text(texts)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    t = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to capture image')
            break
        image_f = clip.encode_image(PIL.Image.fromarray(frame))
        similarity = clip.similarity(image_f, text_f)
        fps = 1 / (time.time() - t)
        t = time.time()
        graph = graphs(frame.shape[0], texts, similarity, fps)
        img = np.concatenate((frame, graph), axis=1)
        cv2.imshow('Similarity', img)
        key = cv2.waitKey(1)
        if key == 27: break
        if key == 13: 
            texts = textlist()
            text_f = clip.encode_text(texts)
            t = time.time()
    cap.release()
    cv2.destroyAllWindows()

    



