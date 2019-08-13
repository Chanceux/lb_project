#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

__title__ = '2-3.py'

__author__ = '李冰'

__mtime__ = '2019/8/12 17:39'


"""
# Fetch and Feed
# Fetch 就是指在会话里面，可以同时去执行多个op，得到运行的结果
import tensorflow as tf
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
add = tf.add(input2, input3)
mul = tf.multiply(input1, add)
with tf.Session() as sess:
    # 在这里就用到了Fetch，可以同时运行mul和add这两个op
    result = sess.run([mul, add])
    print(result)


# Feed
# 先定义一个占位符，placeholder,32位的字符串
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)


with tf.Session() as sess:
    # feed的数据以字典的形式传入；运行output的时候，再进行赋值
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))






