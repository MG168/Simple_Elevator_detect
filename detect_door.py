# -*- coding: UTF-8 -*-
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from direction import direction
# img1 = '/home/ltz/my_code2/dianti_detect/dianti_photo/6.9/ch06_2017060908[20170703-1606442].JPG' # 关门
# # img2 = '/home/ltz/my_code2/dianti_detect/dianti_photo/6.9/ch06_2017060908[20170703-1612464].JPG' # 开门
# img2 = '/home/ltz/my_code2/dianti_detect/dianti_photo/6.9/ch06_2017060908[20170703-1610320].JPG' # 关门

# 两张图片差异过大则输出1，反之则输出0
def D(img1,img2):
    gray_different = 100
    num_different = 500
    # img1=Image.open(img1)
    # img2=Image.open(img2)

    gray1=img1.convert('L')                               # 转化为灰度图
    gray2=img2.convert('L')
    if direction(img2) == direction(img1):
        if direction(img2) == 1:
            box = (775,0,1113,30)
        else:
            box = (628,5,940,30)
    else :
        print('摄像头位置改变')                            # 输出
        box = (0,0,1,1)

    roi1=gray1.crop(box)
    roi2=gray2.crop(box)

    roi1.save("./D_F.jpg")
    roi2.save("./D_S.jpg")

    plt.figure("-")
    plt.subplot(1,2,1), plt.title('F')
    plt.imshow(roi1),plt.axis('off')

    plt.subplot(1,2,2), plt.title('S')
    plt.imshow(roi2),plt.axis('off')

    plt.show()

    m1 = np.asarray(roi1)
    m2 = np.asarray(roi2)

    # print 'm1:',m1
    # print 'm2:',m2
    emptyimg = np.zeros(m1.shape,np.uint8)
    def pic_sub(dest,s1,s2):
        for x in range(dest.shape[0]):
            for y in range(dest.shape[1]):
                if(s2[x,y] > s1[x,y]):
                    dest[x,y] = s2[x,y] - s1[x,y]
                else:
                    dest[x,y] = s1[x,y] - s2[x,y]
    pic_sub(emptyimg,m1,m2)
    m = emptyimg

    a = m.shape[0]
    b = m.shape[1]

    a = int(a)
    b = int(b)
    c = 0
    for i in range(a):
        for j in range(b):
            if abs(m[i][j]) > gray_different:
                c=c+1
    print('door_different:',c)
    if c > num_different:
        diff = 1
    else:
        diff =0
    print(diff)
    return diff
#
# D(img1,img2)
