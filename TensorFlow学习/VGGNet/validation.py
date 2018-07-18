#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: validation.py 
@time: 2018/07/17 
"""
import time
import tensorflow as tf
import numpy as np
import train
import inference
import cifar10
import cifar10_input
batch_size = 128
EVAL_INTERVAL_SECS = 10
data_dir = 'cifar-10-batches-bin'

def one_hot(x, n):
    """
    convert index representation to one-hot representation
    """
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]

def evaluate():
    with tf.Graph().as_default() as g:
        test_images, test_labels = cifar10_input.inputs(batch_size=batch_size, data_dir=data_dir, eval_data=True)
        x_input = tf.placeholder(tf.float32, [batch_size, 24, 24,3], name="x-input")
        y_input = tf.placeholder(tf.float32,[None,10],name='y-input')
        #测试集无需正则化
        y = tf.nn.softmax(inference.inference(x_input,None,None))
        #比较预测结果和标准label
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_input,1))
        #计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        #加载时重命名滑动平均变量
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                tf.train.start_queue_runners(sess=sess)
                #根据checkpoint寻找最新的模型文件名
                ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    #加载模型
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    print(ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    xs, ys = sess.run([test_images, test_labels])
                    ys = one_hot(ys,10)
                    accuracy_score = sess.run(accuracy,feed_dict={x_input:xs,y_input:ys})
                    print("After %s training step,validation accuracy = %g" %(global_step,accuracy_score))
                else:
                    print("No file")
                    return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()