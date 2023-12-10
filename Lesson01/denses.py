'''
Dense
神经元

全连接层Dense的参数
'''
import tensorflow as tf
#Dense : y = wx+b
rows = 1
n = 3
net = tf.keras.layers.Dense(2) #1个隐藏层，1个神经元
net.build((rows,n))# 每个训练数据有N个特征
print("net.W",net.kernel)#参数个数
print("net.B",net.bias)#和Dense数一样