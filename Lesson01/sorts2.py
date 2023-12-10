'''
2维张量排序
数值排序：tf.sort
索引排序：tf.argsort
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

#数值升序
b = tf.sort(a,axis=1,direction="ASCENDING")
log("数值升序== ",b)

#数值降序
b = tf.sort(a,axis=1,direction="DESCENDING")
log("数值降序== ",b)
#索引升序
b = tf.argsort(a,axis=1,direction="ASCENDING")
log("索引升序== ",b)
#索引降序
b = tf.argsort(a,axis=1,direction="DESCENDING")
log("索引降序== ",b)

#按照索引位置B，冲数组a中手机数据
c = tf.gather(a,b)
log("c== ",c)