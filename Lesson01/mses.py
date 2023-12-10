'''
使用均方误差函数 MSE

均方误差（MSE）是一种用于衡量预测值与真实值之间差异的度量。
它通常用作回归问题的损失函数1。均方误差是预测值与真实值之间差异的平方和的平均值1。
如果您正在使用机器学习模型进行回归分析，均方误差可以帮助您评估模型的准确性1。
'''
import tensorflow as tf

rows = 2
out = tf.random.uniform([rows,10])
print("out: ",out)
#（预测的）最大值的索引
print("预测值: ",tf.math.argmax(out,axis=1),"\n")
#（实际的）索引
y = tf.range(rows)
print("y:",y,"\n")
#one_hot 编码
y = tf.one_hot(y,depth=10)
print("Y_One_Hot :",y,"\n")
#损失函数
loss = tf.keras.losses.mse(y,out)
print("row loss:",loss,"\n")
#总损失
loss = tf.reduce_mean(loss)
print("总体损失:",loss,"\n")