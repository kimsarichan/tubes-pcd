import Image
import numpy as np

im = Image.open("jbc.jpg")
row,col =  im.size
data = np.zeros([row, col])
pixels = im.load()
for i in range(row):
    for j in range(col):
        r,g,b =  pixels[i,j]
        data[i,j] = [r,g,b]
