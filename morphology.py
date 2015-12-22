import cv2
import numpy as np
def masking(imgs):
    image = cv2.imread(imgs,0) # Only for grayscale image
    gray_blur = cv2.medianBlur(image, 5)
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
    kernel = np.ones((2, 2), np.uint8)
    print kernel
    #closing eroting
    op = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(op, cv2.MORPH_CLOSE, kernel, iterations=8)
    return closing