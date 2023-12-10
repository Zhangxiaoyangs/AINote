'''
矩阵转置：transpose
'''
import tensorflow as tf
import numpy as np

#########################
##矩阵转置：transpose
a = tf.range([12])
a = tf.reshape(a,[4,3])
print(a)

b = tf.transpose(a)#行类交换
print(b)

#########################
##举例： 1张4*4像素的彩色图片
a =tf.random.uniform([4,4,3],minval=0,maxval=10,dtype=tf.int32)
print(a)

# 指定变化的轴索引
b=tf.transpose(a,perm=[0,2,1])
print(b)
# 变回来
c=tf.transpose(b,perm=[0,2,1])
print(c)