import cv2
import numpy as np

img = cv2.imread('messi.jpg')
img = cv2.resize(img, (512, 512))
def zoomout(img):
    rows,cols,ch = img.shape
    zOut = np.zeros((rows,cols,ch),np.uint8)
    m=0
    n=0
    for i in range(rows-1):
        for j in range(cols-1):
            zOut[m,n]=img[i,j]
            if(j % 2 == 0 & n<j):
                n = n + 16
        if(i % 2 == 0 & m<i):
            m = m + 16    
        n=0
    return zOut
img2= zoomout(img)
img2 = cv2.resize(img2, (500, 500))
cv2.imshow('before',img)
cv2.imshow('zoomout',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
