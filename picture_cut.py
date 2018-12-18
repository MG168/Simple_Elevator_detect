# -*- coding: UTF-8 -*-
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from direction import direction
# img = Image.open('/home/ltz/my_code2/dianti_detect/photo1/capture02.jpg')
def cut_number(img):

    r,g,b=img.split()
    gray=img.convert('L')
    plt.figure('cut')
    plt.subplot(1,2,1),plt.title('origin')
    plt.imshow(gray)
    R = np.asarray(r)
    G = np.asarray(g)
    B = np.asarray(b)

    # print R[27][1562]
    # print G[27][1562]
    # print B[27][1562]

    # box =(1020,0,1030,50) # 门
    # box = (800,830,1230,1066) # 地板
    # box1 = (1348,121,1369,160) # 第一方向
    # box2 = (1198,80,1216,110) # 第二方向

    if direction(img) == 1:
        box = (1348,121,1369,160)
    else:
        box = (1198,80,1216,110)

    roi=gray.crop(box)
        # print '2'
    roi.save("./roi.jpg")
    plt.subplot(1,2,2), plt.title('roi')
    plt.imshow(roi),plt.axis('off')
    plt.show()
# cut_number(img)
