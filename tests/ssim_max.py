"""
check for memorization with max ssim value between random sample and training data
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from skimage.metrics import structural_similarity as ssim

if __name__ == "__main__":

    TRAIN_PATH = '/path/to/training/data'
    SAMPLE_PATH = '/path/to/samples'

    CARD = random.randint(0, 100)

    files1 = os.listdir(TRAIN_PATH)
    files2 = os.listdir(SAMPLE_PATH)

    im2 = Image.open(os.path.join(SAMPLE_PATH, files2[CARD]))
    im2_arr = np.array(im2)

    all_scores = []
    max_score = -1
    for file1 in tqdm(files1):
        im1 = Image.open(os.path.join(TRAIN_PATH, file1))
        im1_arr = np.array(im1)
        score = ssim(im1_arr, im2_arr, data_range=255, channel_axis=2)
        all_scores.append(score)
        if score > max_score:
            max_score = score
            max_im1_arr = im1_arr

    print('max score: ', max_score)

    fig = plt.figure()
    fig.set_facecolor('white')
    fig_ax = fig.add_subplot(1, 2, 1)
    fig_ax.imshow(max_im1_arr)
    fig_ax.set_title('closest training image')
    plt.axis('off')
    fig_ax = fig.add_subplot(1, 2, 2)
    fig_ax.imshow(im2_arr)
    fig_ax.set_title('sample')
    plt.axis('off')
    plt.show()
