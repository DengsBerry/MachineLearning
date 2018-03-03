# -*- coding:utf-8 -*-
import tensorflow as tf

with tf.Session() as sess:
     v1 = tf.constant([1.0,2.0,3.0,4.0])
     v2 = tf.constant([4.0,3.0,2.0,1.0])
     greater = tf.greater(v1,v2)
     select = tf.select(tf.greater(v1,v2),v1,v2)
     print(sess.run(greater))
     print(sess.run(select))
