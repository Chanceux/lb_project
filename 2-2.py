#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

__title__ = '2-2.py'

__author__ = '李冰'

__mtime__ = '2019/8/12 16:31'


"""
# 变量的介绍
import tensorflow as tf
x = tf.Variable([1, 2])
a = tf.Variable([3, 3])
# 增加一个减法op
sub = tf.subtract(x, a)
# 增加一个加法op
add = tf.add(x, sub)
# 初始化所有的变量
init = tf.global_variables_initializer()
# 定义一个会话
with tf.Session() as sess:
    # 先进行变量初始化的操作
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

# 创建一个变量，初始化为0
state = tf.Variable(0, name='counter')
# 创建一个op，作用是使state加1
new_value = tf.add(state, 1)
# 赋值op，即将后面的new_value的值赋值为state
update = tf.assign(state, new_value)
init = tf.global_variables_initializer()

# 定义会话
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    # 执行一个for循环，让update循环5次
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
