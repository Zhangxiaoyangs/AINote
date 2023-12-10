'''
数组比较
'''

import tensorflow as tf
import numpy as np
#########################

def log(prefix="",val=""):
    print(prefix,val,"\n")

a = tf.random.uniform((1,10),minval=0,maxval=10,dtype=tf.int32)
b= tf.random.uniform((1,10),minval=0,maxval=10,dtype=tf.int32)
log("a:",a)
log("b:",b)
#a,b 数组比较
log("b:",b==a)

#相同元素输出
log("a[b==a]:",a[b==a])

#数组相同元素的下标
log("数组相同元素的下标 :",tf.where(a==b))

