import cv2
import numpy as np
import matplotlib.pyplot as plt
def rgb(imgs):
    img = cv2.imread(imgs)
    b,g,r = cv2.split(img)
    cv2.imshow('image blue ',b)
    cv2.imshow('image red ',r)
    cv2.imshow('image green ',g)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
