'''
使用 tf.bordcast_to 进行数组广播
'''
import tensorflow as tf
import numpy as np

a = tf.constant([1,2,3])
print(a)

x =1 
print(a+x)

b = tf.broadcast_to(a,[3,3])
print(b)

x = 10
print(b*10)