
import cv2
import numpy as np
from matplotlib import pyplot as plt
def equalization(img_name):
    img = cv2.imread(str(img_name),0)
    #b,g,r = cv2.split(img)
    equ = cv2.equalizeHist(img)
    res = cv2.cvtColor(equ,cv2.COLOR_GRAY2BGR)
    #res = np.hstack((b,equ)) #stacking images side-by-side
    cv2.imshow("result",res)
    #blue'
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()

    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*256/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    img= cdf[img]

    plt.plot(cdf, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

    #return res

def equ(img_name):
    img = cv2.imread(str(img_name))
    b,g,r = cv2.split(img)
    hist,bins = np.histogram(img.flatten(),256,[0,256])

    cdf = hist.cumsum()

    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*256/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    img= cdf[img]
    return img
