# -*- coding: utf-8 -*-
import tensorflow as tf
from skimage import io
import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def getTestPicArray(filename) :
    im = Image.open(filename)  # 打开为RGB三通道数据
    x_s = 28
    y_s = 28
    out = im.resize((x_s, y_s), Image.ANTIALIAS)    # 按照固定尺寸处理图片

    im_arr = np.array(out.convert('L'))  # 通过convert将RGB转为灰度，array为一维向量

    num0 = 0
    num255 = 0
    threshold = 100

    # 对比每一像素值是否超过阈值，并统计
    for x in range(x_s):
        for y in range(y_s):
            if im_arr[x][y] > threshold : num255 = num255 + 1
            else : num0 = num0 + 1

    # 对比上述差异值，反转后阈值一下置0
    if(num255 > num0) :
        print("convert!")
        for x in range(x_s):
            for y in range(y_s):
                im_arr[x][y] = 255 - im_arr[x][y]
            if(im_arr[x][y] < threshold) :  im_arr[x][y] = 0
            #if(im_arr[x][y] > threshold) : im_arr[x][y] = 0
            #else : im_arr[x][y] = 255
            #if(im_arr[x][y] < threshold): im_arr[x][y] = im_arr[x][y] - im_arr[x][y] / 2

    out = Image.fromarray(np.uint8(im_arr))  # np.uint8为无符号整数，Image.fromarray图像格式转换，与灰度反向操作
    out.save('a.jpg')
    #print im_arr
    # nm = im_arr.reshape((1, 784))

    nm = im_arr.astype(np.float32)    # 变化数组类型
    # nm = (nm - (1.0 / 255.0))/255
    nm[nm<100]=0
    nm = tf.reshape(nm, [-1, 28, 28, 1])  # tf.reshape(tensor, shape, name=None)，函数的作用是将tensor变换为参数shape的形式

    return nm


def evaluate():
    sess = tf.Session()
    x_im = getTestPicArray('/home/ltz/my_code2/dianti_detect/down/roi.jpg')  # 按照上述定义输出tensor
    saver = tf.train.import_meta_graph("Model/model.ckpt.meta")  # 载入之前训练好的模型
    saver.restore(sess,tf.train.latest_checkpoint('./Model'))

    #
    # im = cv2.imread('a.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    #
    # im = cv2.resize(im, (28, 28), interpolation=cv2.INTER_CUBIC)
    # # 图片预处理
    # # img_gray = cv2.cvtColor(im , cv2.COLOR_BGR2GRAY).astype(np.float32)
    # # 数据从0~255转为-0.5~0.5
    # img_gray = (im - (255 / 2.0)) / 255
    # # img_gray = im / 255
    # x_image = tf.reshape(im, [-1, 28, 28, 1])

    # # x_img = np.reshape(img_gray, [-1, 784])

    W_conv1 = sess.run('w1:0')
    b_conv1 = sess.run('b1:0')
    cov1=tf.nn.relu(conv2d(x_im,W_conv1)+b_conv1)
    p1=max_pool_2x2(cov1)

    W_conv2 = sess.run('w2:0')
    b_conv2 = sess.run('b2:0')
    cov2=tf.nn.relu(conv2d(p1,W_conv2)+b_conv2)
    p2=max_pool_2x2(cov2)

    Wf1 = sess.run('wf1:0')
    bf1 = sess.run('bf1:0')
    h_pool2_flat = tf.reshape(p2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, Wf1) + bf1)

    # graph = tf.get_default_graph()
    # keep_prob = graph.get_tensor_by_name("prob:0")
    # init = tf.global_variables_initializer() #加载模型绝对不能添加变量初始化  这条语句之后的变量初始化
    # sess.run(init)
    # keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=1.0)

    W_fc2 = sess.run('wfull2:0')
    b_fc2 = sess.run('bf2:0')



    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    # print(b_fc2)
    result=sess.run(y_conv)
    # print(sess.run(y_conv))
    print("Elevator floor:",np.argmax(result))
