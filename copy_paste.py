import cv2
import numpy as np
from matplotlib import pyplot as plt

def cut(img1,img2):
    img = cv2.imread(img1)
    rows, cols, ch = img.shape
    piece = np.zeros((100,100,ch),np.uint8)
    a = 0
    b = 0
    for i in range(100,200):
        for j in range(100,200):
            piece[a,b] = img[i,j]
            img[i,j] = 0
            b += 1
        b = 0
        a += 1
    ima = cv2.imread(img2)
    ima[100:200, 100:200] = piece
    return ima
