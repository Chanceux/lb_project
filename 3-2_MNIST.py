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

