'''
TF 中top_k() 函数的使用
'''

import tensorflow as tf
import numpy as np
#########################
def log(prefix="",val=""):
    print(prefix,val,"\n")

#########################
# 声明数组a
a = tf.random.uniform([3,5],maxval=10,dtype=tf.int32)
log("a== ",a)
#########################
# 取数组每行前3位
b = tf.math.top_k(a,k=3,sorted=True)
# 前3位数值
log("b的数值：",b.values)

# 前3位索引
log("b的数值：",b.indices)
