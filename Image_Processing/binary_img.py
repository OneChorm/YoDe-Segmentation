import cv2
import numpy as np


def get_binary_img(img):
    # gray img to bin image
    bin_img = np.zeros(shape=(img.shape), dtype=np.uint8)
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            if img[i][j] == 0 :
                 bin_img[i][j] = 0
            else: bin_img[i][j] = 255

    return bin_img
