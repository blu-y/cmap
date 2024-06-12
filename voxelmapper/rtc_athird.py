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

import threading
from threading import Lock

class Camera:
    last_frame = None
    last_ready = None
    lock = Lock()
    capture=None

    def __init__(self, rtsp_link):
        self.capture = cv2.VideoCapture(rtsp_link)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        thread = threading.Thread(target=self.rtsp_cam_buffer, args=(), name="rtsp_read_thread")
        thread.daemon = True
        thread.start()

    def rtsp_cam_buffer(self):
        while True:
            with self.lock:
                self.last_ready = self.capture.grab()

    def getFrame(self):
        if (self.last_ready is not None):
            self.last_ready,self.last_frame=self.capture.retrieve()
            return self.last_frame.copy()
        else:
            return -1

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

def textlist(default=False):
    texts = []
    if default: return ['desk', 'chair', 'white board', 'bottle', 'cellphone', 'umbrella', 'fire extinguisher']
    while True:
        text = input("Input text(nothing to end): ")
        if text == "": 
            break
        else : texts.append(text)
    print("Texts list:", texts)
    return texts

def graphs(width, texts, similarity):
    g = np.zeros((width, width, 3), dtype=np.uint8)
    g[:,-3:,:] = 255
    for i, text in enumerate(texts):
        g = cv2.putText(g, text, (width//10, (2*(i+1))*width//(2*(len(texts)+1))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        g = cv2.putText(g, str(round(similarity[i],4)), (width//5, (2*(i+1)+1)*width//(2*(len(texts)+1))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return g

if __name__ == "__main__":
    model = 'ViT-B-16-SigLIP' # 'ViT-B-32', 'ViT-B-32-256', 'ViT-B-16-SigLIP', 'ViT-L-14-quickgelu'
    w = 1024
    h = 576
    n = 3
    cam = Camera(0)
    texts = textlist(default=True)
    clip = CLIP(model=model)
    text_f = clip.encode_text(texts)
    t = time.time()
    while True:
        frame = cam.getFrame()
        # if not ret:
        if isinstance(frame, int):
            print('Failed to capture image')
            break
        frames = []
        cv2.imshow('original', frame)
        key = cv2.waitKey(1)
        for i in range(n):
            frames.append(frame[:,:w//3,:])
            frames.append(frame[:,w//3:w//3*2,:])
            frames.append(frame[:,w//3*2:w//3*3,:])
        imgs = []
        for i in range(n):
            image_f = clip.encode_image(PIL.Image.fromarray(frames[i]))
            similarity = clip.similarity(image_f, text_f)
            graph = graphs(frames[i].shape[1], texts, similarity)
            imgs.append(np.concatenate((frames[i], graph), axis=0))
        img = np.concatenate([imgs[i] for i in range(n)], axis=1)
        fps = 1 / (time.time() - t)
        t = time.time()
        img = cv2.putText(img, f"FPS: {fps:.2f}", (img.shape[1]-100, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow('Similarity', img)
        key = cv2.waitKey(1)
        if key == 27: break
        if key == 13: 
            texts = textlist()
            text_f = clip.encode_text(texts)
            t = time.time()
    cam.capture.release()
    cv2.destroyAllWindows()

    