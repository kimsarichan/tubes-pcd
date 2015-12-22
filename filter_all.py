import numpy as np
import random
import cv2
import Image, ImageFilter
#SALT AND PAPER NOISE

def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres :
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

img = cv2.imread('messi.jpg')
row,column , h = img.shape
noise_img = sp_noise(img,0.05)
cv2.imwrite("noise.jpg",noise_img)
#average
blur = cv2.blur(noise_img,(3,3))
#median
median = cv2.medianBlur(noise_img,3)
#bilateral
bilateral  = cv2.bilateralFilter(noise_img,9,75,75)
#modus
im=Image.open('messi.jpg')
modus=im.filter(ImageFilter.ModeFilter(3))
cv2.imshow("sp_noise.jpg", noise_img)
cv2.imshow("average", blur)
cv2.imshow("median", median)
cv2.imshow("bilateral", bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()
