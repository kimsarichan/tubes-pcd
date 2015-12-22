import cv2
import numpy as np
import matplotlib.pyplot as plt
def rotate(imgs):
    img = cv2.imread(imgs)
    img = cv2.resize(img, (256, 256))
    rows ,cols ,k = img.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),-135,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst
cv2.imshow('image after ',rotate('jbc.jpg'))
cv2.waitKey(0)
cv2.destroyAllWindows()








