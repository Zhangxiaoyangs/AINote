'''
最大和最小的索引位置
tensorflow
tf.argmax
tf.argmin
'''

import tensorflow as tf
import numpy as np
#########################

def log(prefix="",val=""):
    print(prefix,val,"\n")

#自定义随机数组
a = tf.random.uniform((3,10),minval=0,maxval=10,dtype=tf.int32)    
log("array : ",a)

#取最大索引数组。通常用于取得模型预测结果
b = tf.argmax(a,axis=1)
log("a数组axis=1最大值索引位置：",b)
#取最小索引数组
b = tf.argmin(a,axis=1)
log("a数组axis=1最小值索引位置：",b)