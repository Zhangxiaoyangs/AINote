'''
最大最小和平均值
'''

import tensorflow as tf
import numpy as np
#########################

def log(prefix="",val=""):
    print(prefix,val,"\n")

a = tf.range(12,dtype=tf.float32)
a = tf.reshape(a,(4,3))
log("a数组：",a)    
#########################

b = tf.reduce_min(a)
log("最小值：",b)    
b = tf.reduce_max(a)
log("最大值：",b)
b = tf.reduce_mean(a)
log("平均值：",b)  
#########################

b = tf.reduce_min(a,axis=0)
log("a数组axis=0最小值：",b)    
b = tf.reduce_max(a,axis=0)
log("a数组axis=0最大值：",b)
b = tf.reduce_mean(a,axis=0)
log("a数组axis=0平均值：",b)  
#########################
b = tf.reduce_min(a,axis=1)
log("a数组axis=1最小值：",b)    
b = tf.reduce_max(a,axis=1)
log("a数组axis=1最大值：",b)
b = tf.reduce_mean(a,axis=1)
log("a数组axis=1平均值：",b)  
