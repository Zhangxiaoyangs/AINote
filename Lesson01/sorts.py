'''
张量排序
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
a = tf.random.shuffle(tf.range(10))
log("a== ",a)

#数值升序
b = tf.sort(a,direction="ASCENDING")
log("数值升序== ",b)

#数值降序
b = tf.sort(a,direction="DESCENDING")
log("数值降序== ",b)
#索引升序
b = tf.argsort(a,direction="ASCENDING")
log("索引升序== ",b)
#索引降序
b = tf.argsort(a,direction="DESCENDING")
log("索引降序== ",b)

#按照索引位置B，冲数组a中手机数据
c = tf.gather(a,b)
log("c== ",c)