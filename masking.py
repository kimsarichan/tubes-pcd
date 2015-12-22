import numpy as np
import cv2
import matplotlib.pyplot as plt
def mask(img1,imgs):
    im = cv2.imread(img1)
    img2= cv2.imread(imgs)
    im=cv2.resize(im,(512,512))
    img2=cv2.resize(img2,(512,512))
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask=cv2.bitwise_not(mask)
    masked_data = cv2.bitwise_and(im, im, mask=mask)
    return masked_data
