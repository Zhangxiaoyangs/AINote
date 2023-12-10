'''
自定义变量和常量
'''
import tensorflow as tf

a = tf.Variable(1)
b = tf.Variable(1.)
c = tf.Variable([1.0])
d = tf.Variable(1,dtype=tf.float32)

print("--"*40)
print(a)
print(b)
print(c)
print(d)

print("--"*40)
# 加数组 转换成了 tf.Tensor 类型
print(b+c)
print(b+c[0])
print("--"*40)
x1 = tf.constant(1)
x2 = tf.constant(1.)
x3 = tf.constant([1.0])
x4 = tf.constant(1,dtype=tf.float32)

print(x1)
print(x2)
print(x3)
print(x4)
print("--"*40)
