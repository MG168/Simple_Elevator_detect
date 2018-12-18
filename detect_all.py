# -*- coding: UTF-8 -*-
import detect_people
import detect_door
import picture_cut
import test
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage import io
import numpy as nppy
# import cv2
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import time
np.set_printoptions(threshold=50)

# while 1:
# time.sleep(1)

photo = []
for filename in os.listdir(r"./photo"):
    photo.append('./photo/'+filename)
# img1 = '/home/ltz/my_code2/dianti_detect/dianti_photo/6.12/ch06_2017061208[20170704-1020130].JPG'
# img2 = '/home/ltz/my_code2/dianti_detect/dianti_photo/6.12/ch06_2017061208[20170704-1020231].JPG'
img1,img2=photo[1],photo[0]
img1=Image.open(img1)
img2=Image.open(img2)

people = detect_people.D(img1,img2)  # 对比前后两张图片的差异值
door = detect_door.D(img1,img2)
if people == 1:
    if door ==1:
        print('有人，电梯门打开，记录')
        picture_cut.cut_number(img2)
        test.evaluate()
    else:
        print('有人，但是电梯门没变化')
else:
    print('无人员变动')
