import math
import cv2
import numpy as np
from matplotlib import pyplot as plt

robertOp2 = [[0,0,-1],
             [0,1,0],
             [0,0,0]]
robertOp1 = [[-1,0,0],
             [0,1,0],
             [0,0,0]]
prewwitOpX = [[-1,0,1],
              [-1,0,1],
              [-1,0,1]]
prewwitOpY = [[-1,-1,-1],
              [0,0,0],
              [1,1,1]]
prewwitOpDn = [[-1,-1,-1],
              [1,-2,1],
              [1,1,1]]
prewwitOpDne = [[1,-1,-1],
              [1,-2,-1],
              [1,1,1]]
prewwitOpDe = [[1,1,-1],
              [1,-2,-1],
              [1,1,-1]]
prewwitOpDes = [[1,1,1],
              [1,-2,-1],
              [1,-1,-1]]
prewwitOpDs = [[1,1,1],
              [1,-2,1],
              [-1,-1,-1]]
prewwitOpDsw = [[1,1,1],
              [-1,-2,1],
              [-1,-1,1]]
prewwitOpDw = [[-1,1,1],
              [-1,-2,1],
              [-1,1,1]]
prewwitOpDwn = [[-1,-1,1],
              [-1,-2,1],
              [1,1,1]]
sobelOpX = [[-1,0,1],
            [-2,0,2],
            [-1,0,1]]
sobelOpY = [[-1,-2,-1],
            [0,0,0],
            [1,2,1]]
freichanOpX = [[-1,0,1],
              [-1*math.sqrt(2),0,math.sqrt(2.0)],
              [-1,0,1]]
freichanOpY = [[-1,-1*math.sqrt(2.0),-1],
              [0,0,0],
              [1,math.sqrt(2.0),1]]
laplaceOp = [[0,1,0],
             [1,-4,1],
             [0,1,0]]

robertOp1, robertOp2 = np.array(robertOp1), np.array(robertOp2)
prewwitOpX, prewwitOpY = np.array(prewwitOpX), np.array(prewwitOpY)
prewwitOpDn = np.array(prewwitOpDn)
prewwitOpDne = np.array(prewwitOpDne)
prewwitOpDe = np.array(prewwitOpDe)
prewwitOpDes = np.array(prewwitOpDes)
prewwitOpDs = np.array(prewwitOpDs)
prewwitOpDsw = np.array(prewwitOpDsw)
prewwitOpDw = np.array(prewwitOpDw)
prewwitOpDwn = np.array(prewwitOpDwn)
sobelOpX, sobelOpY = np.array(sobelOpX), np.array(sobelOpY)
freichanOpX, freichanOpY = np.array(freichanOpX), np.array(freichanOpY)
laplaceOp = np.array(laplaceOp)

##img = cv2.GaussianBlur(gray,(3,3),0)
##
##laplacian = cv2.Laplacian(img,cv2.CV_64F)
##sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
##sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y
##sobel = sobelx + sobely
##canny = cv2.Canny(img,100,200)
def edge(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    robert1,robert2 = cv2.filter2D(img, -1, robertOp1),cv2.filter2D(img, -1, robertOp2)
    robert = robert1+robert2
    prewwitX,prewwitY = cv2.filter2D(img, -1, prewwitOpX),cv2.filter2D(img, -1, prewwitOpY)
    prewwit = prewwitX+prewwitY
    sobelX,sobelY = cv2.filter2D(img, -1, sobelOpX),cv2.filter2D(img, -1, sobelOpY)
    sobel = sobelX+sobelY
    freichanX,freichanY = cv2.filter2D(img, -1, freichanOpX),cv2.filter2D(img, -1, freichanOpY)
    freichan = freichanX+freichanY
    laplace = cv2.filter2D(img, -1, laplaceOp)
    prewwitDn = cv2.filter2D(img, -1, prewwitOpDn)
    prewwitDne = cv2.filter2D(img, -1, prewwitOpDne)
    prewwitDe = cv2.filter2D(img, -1, prewwitOpDe)
    prewwitDes = cv2.filter2D(img, -1, prewwitOpDes)
    prewwitDs = cv2.filter2D(img, -1, prewwitOpDs)
    prewwitDsw = cv2.filter2D(img, -1, prewwitOpDsw)
    prewwitDw = cv2.filter2D(img, -1, prewwitOpDw)
    prewwitDwn = cv2.filter2D(img, -1, prewwitOpDwn)
    prewwit2 = prewwitDn+prewwitDne+prewwitDe+prewwitDes+prewwitDs+prewwitDsw
    +prewwitDw+prewwitDwn
    return robert, prewwit, sobel, freichan, laplace, prewwit2

##cv2.imshow('robert',robert)
##cv2.imshow('prewwit',prewwit)
##cv2.imshow('sobel',sobel)
##cv2.imshow('freichan',freichan)
##cv2.imshow('laplace',laplace)
##cv2.imshow('prewwit diferensial',prewwitD)
##
##plt.subplot(4,2,1),plt.imshow(img,cmap = 'gray')
##plt.title('Original'), plt.xticks([]), plt.yticks([])
##plt.subplot(4,2,2),plt.imshow(laplacian,cmap = 'gray')
##plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
##plt.subplot(4,2,3),plt.imshow(sobel,cmap = 'gray')
##plt.title('Sobel'), plt.xticks([]), plt.yticks([])
##plt.subplot(4,2,4),plt.imshow(sobelx,cmap = 'gray')
##plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
##plt.subplot(4,2,5),plt.imshow(sobely,cmap = 'gray')
##plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
##plt.subplot(4,2,6),plt.imshow(canny,cmap = 'gray')
##plt.title('Canny'), plt.xticks([]), plt.yticks([])
##plt.subplot(4,2,7),plt.imshow(prewwit,cmap = 'gray')
##plt.title('Prewwit'), plt.xticks([]), plt.yticks([])
##
##plt.show()
##
##cv2.waitKey(0)
##cv2.destroyAllWindows()
