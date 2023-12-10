'''
使用Tensorflow中提供的类似numpy的功能生产数据
'''
import tensorflow as tf
import numpy as np

a = np.ones(12)
print(a)
a = tf.convert_to_tensor(a)
print(a)
a = tf.zeros(12)
print(a)
#生产一个4列3行的数组
a = tf.zeros([4,3])
print(a)
#生产4个6列3行的数组
a = tf.zeros([4,6,3])
print(a)
#生产1个一样的数组
b = tf.zeros_like(a)
print(b)

#指定生产4列3行 值为10的数组
b = tf.fill([4,3],10.)
print(b)

#指定生产随机数
b = tf.random.normal([4,3])
print(b)

#指定生产随机数
b = tf.random.truncated_normal([3,2])
print(b)

#指定生产指定随机数
b = tf.random.uniform([3,2],minval=0,maxval=10)
print(b)

#指定for 循环
b = tf.range([12],dtype=tf.int32)
print(b)

#打乱
b = tf.random.shuffle(b)
print(b)
