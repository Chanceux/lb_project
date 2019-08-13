#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

__title__ = '3-1_Logistic_Regression.py'

__author__ = '李冰'

__mtime__ = '2019/8/13 10:41'


"""

# 非线性回归
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy来生成200个随机点,从-0.5到0.5之间，均匀生成200个随机点
# 并对其增加一个维度，200行，1列
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# 增加干扰项，生成一些随机值，数据类型和x_data是一样的
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise


# 定义两个placeholder，根据样本来进行定义的
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层,用10个神经元
# 输入一个x_data,经过神经网络，可以得到一个y，这个y为预测值，希望达到的效果是:y能尽可能的接近y_data
# 权值，先随机赋值，1行10列。[1,10]代表1个输入
Weight_L1 = tf.Variable(tf.random_normal([1, 10]))
# 偏置值，先进行初始化为0，创建一个1行10列的全0矩阵
biases_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(x, Weight_L1) + biases_L1
# 中间层的输出L1,利用双曲正切函数作为激活函数
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义神经网络的输出层，1个神经元
# 权值为10行一列
Weight_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
# 输出层的输入，为中间层的输出，因此在这里输入为L1
Wx_plus_b_L2 = tf.matmul(L1, Weight_L2) + biases_L2
# 最后为预测的结果
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
# 使用梯度下降法，学习率为0.1，最小化loss函数
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 定义会话
with tf.Session() as sess:
    # 首先进行变量的初始化
    sess.run(tf.global_variables_initializer())
    # 训练2000次
    for _ in range(2000):
        # feed_dict为x和y进行赋值
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

        # 获得预测值
        prediction_value = sess.run(prediction, feed_dict={x: x_data})

        # 画图
    plt.figure()
    # 使用散点图的方式
    plt.scatter(x_data, y_data)
    # ‘r-’代表画的是一条红色实线，线宽设为5
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    # 最后把整个图显示出来
    plt.show()



