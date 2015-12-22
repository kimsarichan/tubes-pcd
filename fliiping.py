import cv2
import numpy as np


def flip_vertical(imgs):
   img = cv2.imread(imgs)
   img = cv2.resize(img, (256, 256))
   rows,cols,ch = img.shape
   flipVertical = np.zeros((rows,cols,ch),np.uint8)
   rows2 = rows - 1
   for i in range (0,rows-1):
      for j in range(0,cols-1):
         flipVertical[rows2,j]=img[i,j]
      rows2=rows2-1
   return flipVertical

def flip_horizontal(imgs):
   img = cv2.imread(imgs)
   img = cv2.resize(img, (256, 256))
   rows,cols,ch = img.shape
   flipHorizontal = np.zeros((rows,cols,ch),np.uint8)
   cols2 = cols - 1
   for i in range (0,cols-1):
      for j in range(0,rows-1):
         flipHorizontal[j,cols2]=img[j,i]
      cols2=cols2-1
   return flipHorizontal


