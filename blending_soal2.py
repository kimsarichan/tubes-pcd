import cv2
import numpy as np
import matplotlib.pyplot as plt
def blending(img,imgs):
    img1 = cv2.imread(img)
    img2 = cv2.imread(imgs)
    resized_image1 = cv2.resize(img1, (500, 500))
    resized_image2 = cv2.resize(img2, (500, 500))
    j=0
    i=1

    while j<=0.5 :
        result = cv2.addWeighted(resized_image1,j,resized_image2,i,0)
        cv2.imshow('final',result)
        cv2.waitKey(200)
        j+=0.10
        i-=0.10

    cv2.waitKey(0)
    cv2.destroyAllWindows()
