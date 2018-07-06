#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: main.py 
@time: 2018/07/06 
"""

import tensorflow  as tf
import numpy as np

def main():
    g1 = tf.Graph()
    with g1.as_default():
        #计算图g1初始化时为0
        v = tf.get_variable("v",shape=[1],initializer=tf.zeros_initializer())

    g2 = tf.Graph()
    with g2.as_default():
        # 计算图g2初始化时为1
        v = tf.get_variable("v",shape=[1],initializer=tf.ones_initializer())

    with tf.Session(graph=g1) as sess:
        tf.global_variables_initializer().run()
        with tf.variable_scope("",reuse=True):
            print(sess.run(tf.get_variable("v")))

    with tf.Session(graph=g2) as sess:
        tf.global_variables_initializer().run()
        with tf.variable_scope("",reuse=True):
            print(sess.run(tf.get_variable("v")))

    g = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    #第一个参数保证了在使用GPU出现问题时，切换gpu或者cpu,所以下面在选择时GPU：1时不会报错
    sess = tf.Session(graph=g,config=config)
    with g.as_default():
        a = tf.constant([1.0,2.0],name="a")
        b = tf.constant([2.0,3.0],name="b")
        c = tf.constant([3.0,4.0],name="c")
        result = a + b * c
    print(sess.run(result))
    with g.device('/gpu:1'):
        #选择GPU：1跑代码
        print(sess.run(result))

def test():
    #一个两层的神经网络

    batch_size = 8
    #训练数据batch的大小

    #初始化权值矩阵
    w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
    w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

    #输入
    x_input = tf.placeholder(tf.float32,shape=(None,2),name="x-input")
    y_input = tf.placeholder(tf.float32,shape=(None,1),name="y-input")

    #前向传播
    a = tf.matmul(x_input,w1)
    y = tf.matmul(a,w2)

    #损失函数和反向传播
    cross_entropy = -tf.reduce_mean(y_input*tf.log(tf.clip_by_value(y,1e-10,1.0)))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    #随机产生训练集
    rdm = np.random.RandomState(1)
    dataset_size = 128
    X = rdm.rand(dataset_size,2)
    Y = [[int(x1+x2<1)] for (x1,x2) in X]


    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print(sess.run(w1))
        print(sess.run(w2))
        STEPS = 5000
        for i in range(STEPS):
            start = (i*batch_size)%dataset_size
            end = min(start+batch_size,dataset_size)

            sess.run(train_step,feed_dict={x_input:X[start:end],y_input:Y[start:end]})
            if i%1000 ==0:
                total_cross_entropy = sess.run(cross_entropy,feed_dict={x_input:X,y_input:Y})
                print("After %d training_step(s),cross entropy on all data is %g" %(i,total_cross_entropy))

        print(sess.run(w1))
        print(sess.run(w2))




if __name__ == '__main__':
    test()