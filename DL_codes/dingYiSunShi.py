# -*- coding:utf-8 -*-
import tensorflow as tf  
from numpy.random import RandomState  

# 一次性获得N个样例的结果
batch_size = 8  

#这两个是将要输入的项
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")  
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')  
# 随机生成的参数后期需要学习  stddev定义标准差
w1= tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))  
y = tf.matmul(x, w1)  
  
# 定义损失函数使得预测少了的损失大，于是模型应该偏向多的方向预测。  
loss_less = 10  
loss_more = 1  

#y_是实际值，y是预测结果
loss = tf.reduce_sum(tf.select(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))  
# 常用的优化算法 还有GradientDescentOptimizer MomentumOptimizer
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)  
  
rdm = RandomState(1)  
X = rdm.rand(128,2)  

# 加入噪声凸显损失函数意义
Y = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1, x2) in X]  
  
with tf.Session() as sess:  
    init_op = tf.initialize_all_variables()  
    sess.run(init_op)  
    STEPS = 5000  
    for i in range(STEPS):
		# 每batch_size一个轮
        start = (i*batch_size) % 128  
        end = (i*batch_size) % 128 + batch_size  
		# 学习过程 feed_dict表示数据来源
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})  
        print sess.run(w1) 
