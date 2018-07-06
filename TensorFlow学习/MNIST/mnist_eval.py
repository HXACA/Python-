#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: mnist_eval.py 
@time: 2018/07/06 
"""

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x_input = tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name='x-input')
        y_input = tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name='y-input')
        validate_feed = {x_input:mnist.validation.images,y_input:mnist.validation.labels}
        #测试集无需正则化
        y = mnist_inference.inference(x_input,None)
        #比较预测结果和标准label
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_input,1))
        #计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        #加载时重命名滑动平均变量
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                #根据checkpoint寻找最新的模型文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    #加载模型
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    print(ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
                    print("After %s training step,validation accuracy = %g" %(global_step,accuracy_score))
                else:
                    print("No file")
                    return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()