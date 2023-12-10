'''
维度变化之 expand_dims,squeeze

数组升维：expand_dims
数组降维：squeeze
'''
import tensorflow as tf
import numpy as np
a = tf.range([24])
print(a)
print(a.shape)
print(a.ndim)
#升维
b = tf.expand_dims(a,axis=0)
print(b)
print(b.shape)
print(b.ndim)
#降维
c = tf.squeeze(b,axis=0)
print(c)
print(c.shape)
print(c.ndim)