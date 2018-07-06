#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: main.py 
@time: 2018/07/06 
"""

import tensorflow as tf

def get_weight(shape,lamd):
    #生成变量
    var = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    #正则化对损失函数的影响
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lamd)(var))
    return var

def ExponentialMovingAverage():

    #计算滑动平均的初始值，影子变量初始值相同
    v1 = tf.Variable(0,dtype=tf.float32)
    #模拟迭代次数，不做权值的更新
    step = tf.Variable(0,trainable=False)
    #定义滑动平均的类，给定衰减率0.99和step,decay越大，越受到影子变量的影响，模型越稳定
    ema = tf.train.ExponentialMovingAverage(0.99,step)
    #给定列表，每次执行此操作，更新列表的变量
    maintain_averages_op = ema.apply([v1])
    with tf.Session() as sess:
        #初始化所有变量，影子变量和v都为0
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print(sess.run([v1,ema.average(v1)]))
        #将v设置为5，影子变量此时为0
        sess.run(tf.assign(v1,5))

        #此时的dacay = min(decay,(1+step)/(10+step)) = 0.1
        #影子变量 = decay*影子变量+(1-decay)*v = 0.1*0 +0.9*5 = 4.5
        sess.run(maintain_averages_op)
        print(sess.run([v1,ema.average(v1)]))
        # 影子变量 = decay*影子变量+(1-decay)*v = 0.1*4.5 +0.9*5 = 4.95
        sess.run(maintain_averages_op)
        print(sess.run([v1, ema.average(v1)]))

        # 此时的dacay = min(decay,(1+step)/(10+step)) = 0.99
        sess.run(tf.assign(step,10000))
        # 影子变量 = decay*影子变量+(1-decay)*v = 0.99*4.95 +0.01*10 = 5.0005
        sess.run(tf.assign(v1,10))

        sess.run(maintain_averages_op)
        print(sess.run([v1,ema.average(v1)]))

        sess.run(maintain_averages_op)
        print(sess.run([v1, ema.average(v1)]))

def main():
    x_input = tf.placeholder(tf.float32,shape=(None,2))
    y_input = tf.placeholder(tf.float32,shape=(None,1))

    batch_size = 8
    layer_dimension = [2,10,10,10,1]
    n_layers = len(layer_dimension)
    cur_layer = x_input
    in_dimension = layer_dimension[0]

    for i in range(1,n_layers):
        out_dimension = layer_dimension[i]
        weight = get_weight([in_dimension,out_dimension],0.001)
        bias = tf.Variable(tf.constant(0.1,shape=[out_dimension]))
        cur_layer = tf.nn.relu(tf.matmul(cur_layer,weight)+bias)
        in_dimension = layer_dimension[i]

    mse_loss = tf.reduce_mean(tf.square(y_input-cur_layer))
    tf.add_to_collection('losses',mse_loss)

    #将正则化误差和均方差误差相加得到最后的误差函数

    loss = tf.add_n(tf.get_collection('losses'))

if __name__ == '__main__':
    ExponentialMovingAverage()