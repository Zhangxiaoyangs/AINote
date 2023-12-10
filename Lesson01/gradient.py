'''
求导 gradient
API = https://www.tensorflow.org/api_docs/python/tf/GradientTape


公式：
f(x)=x^n
微积分（导数）：
f'(x)= n*x^(n-1)

例：
y=x^2
微积分（导数）：
dy/dx=2x^(2-1)=2x
-------------------------------------
4*3^(4-1)

'''
import tensorflow as tf

x = tf.constant(3.)

with tf.GradientTape() as tape:
    tape.watch([x])
    y=x**4

dy_dx = tape.gradient(y,x)
print(dy_dx)
