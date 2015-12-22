import cv2
import numpy as np
def sampling(imgs,kernel):
    img = cv2.imread(imgs)
    img = cv2.resize(img, (512, 512))
    r,c,t = img.shape
    coba = np.zeros((r,c,t),np.uint8)
    x=kernel
    sum=[0,0,0]
    for i in range(0,r,x) :
        for j in range(0,c,x) :
             for m in range(x)   :
                 for n in range(x) :
                    for k in range(t):
                        sum[k]=img[j+n,i+m][k]
             for k in range(3):
                 sum[k] =sum[k]/(x*x)
             for m in range(x):
                 for n in range(x):
                     coba[j+n,i+m]=sum
             sum=[0,0,0]
    return coba

#cv2.imshow('imgsampling',sampling('messi.jpg'))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
