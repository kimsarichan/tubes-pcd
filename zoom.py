import cv2
import numpy as np

img = cv2.imread('jbc.jpg')
img= cv2.resize(img, (256, 256))
def zoomin(img):
    rows,cols,ch = img.shape
    zin = np.zeros((rows*2,cols*2,ch),np.uint8)
    m=0
    n=0
    for i in range(rows-1):
        for j in range(cols-1):
            zin[m,n]=img[i,j]
            zin[m,n+1]=img[i,j]
            zin[m+1,n]=img[i,j]
            zin[m+1,n+1]=img[i,j]
            n=n+2
        m=m+2
        n=0
    return zin

def zoomout(img):
    rows,cols,ch = img.shape
    zout = np.zeros((rows/2,cols/2,ch),np.uint8)
    r2,c2,t2 = zout.shape
    m=0
    n=0
    for i in range(0,rows-1) :
        for j in range(0,cols-1) :
            zout[i,j] = ((img[m,n]+img[m,n+1]+img[m+1,n]+img[m+1,n+1]))
            n=n+2
        m =m+2
        n =0
    return zout
    
cv2.imshow('before',img)
cv2.imshow('zoomin',zoomin(img))
cv2.waitKey(0)
cv2.destroyAllWindows()
