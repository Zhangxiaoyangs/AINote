'''
维度变化之reshape
'''
import tensorflow as tf
#10张彩色图片，宽和高28，3颜色的RGB
print("10张彩色图片，宽和高28，3颜色的RGB")
a = tf.random.normal([10,28,28,3])
# print(a)
print(a.shape) # 形状
print(a.ndim)  # 维度
# 把宽度和高度相乘 得到784
print("把宽度和高度相乘 得到784")
b = tf.reshape(a,[10,784,3])
# print(b)
print(b.shape) # 形状
print(b.ndim)  # 维度
# 把宽度和高度相乘 -1 = 784 自动计算的结果
print("把宽度和高度相乘 -1 = 784 自动计算的结果")
c = tf.reshape(a,[10,-1,3])
# print(c)
print(c.shape) # 形状
print(c.ndim)  # 维度
# 把宽度和高度相乘 784*3 乘以颜色
print("把宽度和高度相乘 784*3 乘以颜色")
d = tf.reshape(a,[10,784*3])
# print(d)
print(d.shape) # 形状
print(d.ndim)  # 维度

# 784*3 也可以写成 -1
print(" 784*3 也可以写成 -1")
e = tf.reshape(a,[10,-1])
print(d)
print(e.shape) # 形状
print(e.ndim)  # 维度