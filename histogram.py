import cv2
import numpy as np
from matplotlib import pyplot as plt
def histogram(imgs):
    img = cv2.imread(imgs)
    color = ('b','g','r')
    b,g,r = cv2.split(img)
    cv2.imshow('image blue ',b)
    cv2.imshow('image red ',r)
    cv2.imshow('image green ',g)
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,128])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()
