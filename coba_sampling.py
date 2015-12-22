import cv2
import numpy as np

img = cv2.imread('messi.jpg')
r,c,t = img.shape
sampl = np.zeros((r,c,t),np.uint8)
part = 16

m = 0
n = 0
k=img[m,n]+img[m,n+1]+img[m+1,n]+img[m+1,n+1]
for i in range (0,r-1):
    for j in range(0,c-1):
        sampl[i,j]=k
        if(i%4==0 and j%4==0):
            k=(img[m,n]+img[m,n+1]+img[m+1,n]+img[m+1,n+1])/4
        n=n+1
    m=m+1
    n = 0

cv2.imshow('asli',img)
cv2.imshow('sampling',sampl)
cv2.waitKey(0)
cv2.destroyAllWindows()
