import cv2
def pseudo_color(img_name):
    im_gray = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
    return im_color
