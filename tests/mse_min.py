"""
check for memorization with min mse value between random sample and training data
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from skimage.metrics import mean_squared_error as mse

if __name__ == "__main__":

    TRAIN_PATH = '/path/to/training/data'
    SAMPLE_PATH = '/path/to/samples'

    CARD = random.randint(0, 100)

    files1 = os.listdir(TRAIN_PATH)
    files2 = os.listdir(SAMPLE_PATH)

    im2 = Image.open(os.path.join(SAMPLE_PATH, files2[CARD]))
    im2_arr = np.array(im2)

    min_error = 1000
    for file1 in tqdm(files1):
        im1 = Image.open(os.path.join(TRAIN_PATH, file1))
        im1_arr = np.array(im1)
        error = mse(im1_arr, im2_arr)
        if error < min_error:
            min_error = error
            min_im1_arr = im1_arr

    print('min error: ', min_error)

    fig = plt.figure()
    fig.set_facecolor('white')
    fig_ax = fig.add_subplot(1, 2, 1)
    fig_ax.imshow(min_im1_arr)
    fig_ax.set_title('closest training image')
    plt.axis('off')
    fig_ax = fig.add_subplot(1, 2, 2)
    fig_ax.imshow(im2_arr)
    fig_ax.set_title('sample')
    plt.axis('off')
    plt.show()
