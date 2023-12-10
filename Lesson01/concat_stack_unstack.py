'''
数组的合并
使用tf.concat,tf.stack,tf.unstack 合并数组
'''
import tensorflow as tf
######################################
# 数组合并
# concat
a = tf.zeros([2,4,3])
b = tf.ones([2,4,3])
print(a)
print(b)
# 数组0轴合并  [4,4,3]
c = tf.concat([a,b],axis=0)
print(c)
# 数组1轴合并  [2,8,3]
c = tf.concat([a,b],axis=1)
print(c)

# 数组2轴合并  [2,4,6]
c = tf.concat([a,b],axis=2)
print(c)

#扩充一维，例如把多张图片放入一个大数组中-》2.2.4.3
c = tf.stack([a,b],axis=0)
print(c)
#降低维度，拆分数组
n,m = tf.unstack(c,axis=0)
print(n)
print(m)