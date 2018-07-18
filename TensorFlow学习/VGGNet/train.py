#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: train.py 
@time: 2018/07/17 
"""

import os
import tensorflow as tf
import inference
import numpy as np
import cifar10_input
import cv2
BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001

TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"
data_dir = 'cifar-10-batches-bin'


def one_hot(x, n):
    """
    convert index representation to one-hot representation
    """
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]

def train():
    x_input = tf.placeholder(tf.float32,[BATCH_SIZE,24,24,3],name="x-input")
    y_input = tf.placeholder(tf.float32,[None,10],name="y-input")
    train_images, train_labels = cifar10_input.distorted_inputs(batch_size=BATCH_SIZE, data_dir=data_dir)
    #l2正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    #计算前向传播
    y = inference.inference(x_input,True,regularizer)
    #记录步数，代表训练轮数，设为不可训练
    global_step = tf.Variable(0,trainable=False)
    #滑动平均模型，加快开始时的训练速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    #对所有可训练参数应用滑动平均模型
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    #计算损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_input,logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #加上正则化损失，tf.add_n实现的是列表元素相加
    loss = cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    #指数衰减学习率，使得开始时学习率较大，后期趋向稳定
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,50000/BATCH_SIZE,LEARNING_RATE_DECAY)
    #优化器可以使global_step自增
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    #同时计算后向传播的参数和每一个参数的滑动平均值
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name="train")
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #持久化类
    saver = tf.train.Saver()
    #建立会话，开始训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.train.start_queue_runners(sess=sess)
        for i in range(TRAINING_STEPS):
            #小批量随机梯度下降
            xs,ys = sess.run([train_images,train_labels])
            ys = one_hot(ys,10)
            #后向传播，损失值,当前训练轮数
            _,loss_value,step,label = sess.run([train_op,loss,global_step,y],feed_dict={x_input:xs,y_input:ys})
            if i%1000 == 0:
                accuracy_score = sess.run(accuracy, feed_dict={y:label , y_input: ys})
                print("After %s training steps,validation accuracy = %g" % (step, accuracy_score))
                print("After %d training steps,loss on training batch is %g" %(step,loss_value))
                #持久化当前模型，保存文件会带有当前训练轮数
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()