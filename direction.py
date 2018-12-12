# -*- coding: UTF-8 -*-
'''
match with pattern
Author   : MG Studio
Datetime : 2018/9/21
Filename : direction.py
'''

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def direction(img):
    r,g,b=img.split()   # split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则仅分隔 num 个子字符串,此处分割图片rgb值
    R = np.asarray(r)
    G = np.asarray(g)
    B = np.asarray(b)

    if B[27][1562] >= 130 and R[27][1562] < 150 and B[27][1562]>R[27][1562]:    # 这里用了模板：判断摄像头的两个方向
        return 1
    else:
        return 2
