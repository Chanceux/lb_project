#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

__title__ = '3-2_MNIST.py'

__author__ = '李冰'

__mtime__ = '2019/8/13 16:47'


"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集,会直接在网上进行下载
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每个批次的大小,即一次性放入100张图片到神经网络中去进行训练，以矩阵的形式放进去
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples//batch_size

# 定义两个placeholder
# None表示任意的值，即一个一个批次的传，784表示每张图片为28*28的向量
x = tf.placeholder(tf.float32, [None, 784])
# 10代表有10个标签：0——9
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W)+b)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
# 使用梯度下降法,0.2的学习率
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()
# 结果存放在一个布尔型列表中，argmax返回一维张量中最大的值所在的位置
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # 初始化变量
    sess.run(init)
    # 一共迭代21个周期
    for epoch in range(30):
        # 每个周期里面加一个循环，n_batch表示一共有多少个批次
        for batch in range(n_batch):
            # batch_size为100,batch_xs保存图片的数据，batch_ys保存图片的标签
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter" + str(epoch) + ",Testing Accuracy" + str(acc))


