import tensorflow as tf

v1 = tf.Variable(0.0, dtype=tf.float32)
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均的类（class）。初始化时给定了衰减率（0.99）和控制衰减率的变量step。
ema = tf.train.ExponentialMovingAverage(0.99, step)
# 定义一个更新变量滑动平均的操作。这里需要给定一个列表，每次执行这个操作时
# 这个列表中的变量都会被更新。
maintain_averages_op = ema.apply([v1])             
                
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
sess.run(init_op)

    # 更新变量v1的值到5。
sess.run(tf.assign(v1, 5))
    # 更新v1的滑动平均值。衰减率为min{0.99,(1+step)/(10+step)= 0.1}=0.1,
    # 所以v1的滑动平均会被更新为0.10+0.95=4.5。
sess.run(maintain_averages_op)
    print sess.run([v1, ema.average(v1)])        #  输出[5.0, 4.5]
    

    # 更新step的值为10000。
    sess.run(tf.assign(step, 10000))  
    # 更新v1的值为10。
sess.run(tf.assign(v1, 10))
    # 更新v1的滑动平均值。衰减率为min{0.99,(1+step)/(10+step)  0.999}=0.99,
    # 所以v1的滑动平均会被更新为0.994.5+0.0110=4.555。
    sess.run(maintain_averages_op)
print sess.run([v1, ema.average(v1)])      #  输出[10.0, 4.5549998]