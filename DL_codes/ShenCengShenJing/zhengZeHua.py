# -*- coding:utf-8 -*-

import tensorflow as tf
# w为需要计算正则化损失的参数
weights = tf.constant([[1.0,-2.0],[-3.0,4.0]])

with tf.Session() as sess:
	#损失函数为J(ceita),优化时加上正则项lambda*R(w)，
	#w为权重，lamda为模型复杂损失在在总损失里面的比例
	# L1正则化 .5为 J(ceita)+lambdaR(w) 中的lambda  
	# 计算过程：(|1|+|-2|+|-3|+|4|)x0.5 = 5.0
	print sess.run(tf.contrib.layers.l1_regularizer(.5)(weights))
	print sess.run(tf.contrib.layers.l2_regularizer(.5)(weights))
