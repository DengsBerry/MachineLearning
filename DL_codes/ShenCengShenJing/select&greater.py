# -*- coding:utf-8 -*-
import tensorflow as tf

with tf.Session() as sess:
     v1 = tf.constant([1.0,2.0,3.0,4.0])
     v2 = tf.constant([4.0,3.0,2.0,1.0])
     #greater可比较两者的大小输出真假值 维度不够的话补充0
	 greater = tf.greater(v1,v2)
	 #select后面参数：第一个为要比较的量 true的时候输出第二个 否则第三个
     select = tf.select(tf.greater(v1,v2),v1,v2)
     print(sess.run(greater))
     print(sess.run(select))
