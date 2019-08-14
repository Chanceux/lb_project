#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

__title__ = '2-1.py'

__author__ = '李冰'

__mtime__ = '2019/8/12 16:04'


"""
# 创建图，启动图
import tensorflow as tf
# 创建一个常量op
m1 = tf.constant([[3, 3]])
# 创建一个常量op
m2 = tf.constant([[2], [3]])
# 创建一个矩阵乘法op,把m1和m2传入
product = tf.matmul(m1, m2)
print(product)
# 输出为：Tensor("MatMul:0",shape=(1, 1), dtype= int32)
# 意思就是说输出的矩阵为1*1的，数据类型为int类型


# 定义一个会话，启动默认图
sess = tf.Session()
# 调用sess的run方法来执行矩阵乘法op
# run(product)触发了图中3个op

result = sess.run(product)
print(result)
# 关闭会话
sess.close()
# 输出为[[15]]

# 或者也可以自己定义一个会话，这样执行完成后，就不需要关闭会话
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
# 输出为[[15]]





