import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg',0)

laplacian = cv2.Laplacian(img,cv2.CV_8U)
#sobel
sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3)
#prewitt 1
kernel1 = np.matrix([[-1,0,1],[-1,0,1],[-1,0,1]])
prewittx = cv2.filter2D(img,-1,kernel1)
kernel2 = np.matrix([[-1,-1,-1],[0,0,0],[1,1,1]])
prewitty = cv2.filter2D(img,-1,kernel2)

gab_prewitt=np.sqrt((kernel1*kernel1)+(kernel2*kernel2))
gabung=prewittx+prewitty
#show
plt.subplot(2,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(sobely,cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(prewittx,cmap='gray')
plt.title('Prewitt X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(gabung,cmap='gray')
plt.title('Prewitt Y'), plt.xticks([]), plt.yticks([])


plt.show()
