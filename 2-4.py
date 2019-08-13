#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

__title__ = '2-4.py'

__author__ = '李冰'

__mtime__ = '2019/8/12 18:24'


"""
# tensorflow的简单实例
import tensorflow as tf
import numpy as np
# 使用numpy生成100个随机点
x_data = np.random.rand(100)
y_data = x_data*0.1 + 0.2

# 构造一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b

# 二次代价函数,影响loss函数的只有b和k这两个变量
loss = tf.reduce_mean(tf.square(y_data-y))
# 定义一个梯度下降法来进行训练的优化器,学习率为0.2
optimizer = tf.train.GradientDescentOptimizer(0.2)
# 定义一个最小化代价函数
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run([k, b]))

