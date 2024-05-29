import sys
import os
sys.path.append(os.getcwd())
from cmap.src.utils import CLIP
import matplotlib.pyplot as plt
import numpy as np
import PIL
import time

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
    clip = CLIP(model='ViT-B-16-SigLIP')

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

    