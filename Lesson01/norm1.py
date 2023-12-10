'''
1 范数的计算 -  tf.norm
使用tensorflow提供的1范数函数计算向量
所有制的最绝值之和
'''
import tensorflow as tf
import numpy as np

#########################

def log(prefix="",val=""):
    print(prefix,val,"\n")
#数组
a = tf.range(12,dtype=tf.float32)
a = tf.reshape(a,[4,3])
a = a-5
log("a:",a)
# 1范数：所有制的最绝值之和
b = tf.norm(a,ord=1)
log("a的1范数b",b)
print(tf.reduce_sum(tf.abs(a)))


# 指定计算轴:axis=0
b = tf.norm(a,ord=1,axis=0)
log("a的axis=0的1范数b:",b)

# 指定计算轴:axis=1
b = tf.norm(a,ord=1,axis=1)
log("a的axis=1的1范数b:",b)
print(tf.reduce_sum(b))
