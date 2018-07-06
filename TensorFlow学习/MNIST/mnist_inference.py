#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: mnist_inference.py
@time: 2018/07/06 
"""

import tensorflow as tf
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape,regularizer):
    weights = tf.get_variable("weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer!=None:
        #计入集合，集合可以保存一组实体，这样可以减少确实函数的定义长度，增加可读性
        tf.add_to_collection("losses",regularizer(weights))
    return weights

def inference(input_tensor,regularizer):
    #变量管理，通过get_variable来获取变量，生成上下文管理器是reuse=True边饰只能获取已创建的变量
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases = tf.get_variable("biases",[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases = tf.get_variable("biases",[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weights)+biases
    return layer2