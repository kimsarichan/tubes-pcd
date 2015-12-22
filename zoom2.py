import cv2
import numpy as np

img = cv2.imread('messi.jpg')
r,c,t = img.shape
zOut = np.zeros((r*2,c*2,t),np.uint8)
zIn = np.zeros((r/2,c/2,t),np.uint8)
r2,c2,t2 = zIn.shape

#zoom out
m = 0
n = 0
for i in range(0,r-1) :
    for j in range(0,c-1) :
        zOut[m,n] = img[i,j]
        zOut[m,n+1] = img[i,j]
        zOut[m+1,n] = img[i,j]
        zOut[m+1,n+1] = img[i,j]
        n = n + 2
    m = m + 2
    n = 0


#zoom in
m = 0
n = 0
for i in range(0,r2-1) :
    for j in range(0,c2-1) :
        zIn[i,j] = ((img[m,n]+img[m,n+1]+img[m+1,n]+img[m+1,n+1])/2)
        n = n + 2
    m = m + 2
    n = 0

cv2.imshow('img',img)
cv2.imshow('imgZoomOut',zOut)
cv2.imshow('imgZoomIn',zIn)
cv2.waitKey(0)
cv2.destroyAllWindows()
