import cv2
import numpy as np
import time

img1 = cv2.imread('messi.jpg')
img2 = cv2.imread('jbc.jpg')
h = np.zeros((300,256,3))

bins = np.arange(256).reshape(256,1)
color = [ (255,0,0),(0,255,0),(0,0,255) ]
for ch, col in enumerate(color):
    hist_item1 = cv2.calcHist([img1],[ch],None,[256],[0,255])
    hist_item2 = cv2.calcHist([img2],[ch],None,[256],[0,255])
    cv2.normalize(hist_item1,hist_item1,0,255,cv2.NORM_MINMAX)
    cv2.normalize(hist_item2,hist_item2,0,255,cv2.NORM_MINMAX)
    sc= cv2.compareHist(hist_item1, hist_item2, cv2.CV_COMP_CORREL)
    #sc= cv2.compareHist(hist_item1, hist_item2, cv.CV_COMP_CORREL)
#cv2.compareHist(imageHistogramList[i], imageHistogramList[j], cv.CV_COMP_CORREL)
    printsc
    hist=np.int32(np.around(hist_item))
    pts = np.column_stack((bins,hist))
    cv2.polylines(h,[pts],False,col)

h=np.flipud(h)
cv2.imwrite('hist.png',h)
