#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: inference.py 
@time: 2018/07/17 
"""

import tensorflow as tf


def inference(input_tensor,train,regularizer):

    with tf.variable_scope('layer1'):
        conv1_weights = tf.get_variable("weight",[3,3,3,16],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias",[16],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    with tf.variable_scope('layer2'):
        conv2_weights = tf.get_variable("weight", [3, 3, 16, 32],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(relu1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope('pool1'):
        pool1 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.variable_scope('layer3'):
        conv3_weights = tf.get_variable("weight",[3,3,32,64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias",[64],initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool1,conv3_weights,strides=[1,1,1,1],padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3,conv3_biases))

    with tf.variable_scope('layer4'):
        conv4_weights = tf.get_variable("weight", [3, 3, 64, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(relu3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.name_scope('pool2'):
        pool2 = tf.nn.max_pool(relu4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.variable_scope('layer5'):
        conv5_weights = tf.get_variable("weight",[3,3,64,128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable("bias",[128],initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(pool2,conv5_weights,strides=[1,1,1,1],padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5,conv5_biases))

    with tf.variable_scope('layer6'):
        conv6_weights = tf.get_variable("weight", [3, 3, 128, 128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv6_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv6 = tf.nn.conv2d(relu5, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_biases))

    with tf.variable_scope('layer7'):
        conv7_weights = tf.get_variable("weight", [1, 1, 128, 128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv7_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv7 = tf.nn.conv2d(relu6, conv7_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu7 = tf.nn.relu(tf.nn.bias_add(conv7, conv7_biases))

    with tf.name_scope('pool3'):
        pool3 = tf.nn.max_pool(relu7,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.variable_scope('layer8'):
        conv8_weights = tf.get_variable("weight",[3,3,128,128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv8_biases = tf.get_variable("bias",[128],initializer=tf.constant_initializer(0.0))
        conv8 = tf.nn.conv2d(pool3,conv8_weights,strides=[1,1,1,1],padding='SAME')
        relu8 = tf.nn.relu(tf.nn.bias_add(conv8,conv8_biases))

    with tf.variable_scope('layer9'):
        conv9_weights = tf.get_variable("weight", [3, 3, 128, 128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv9_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv9 = tf.nn.conv2d(relu8, conv9_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu9 = tf.nn.relu(tf.nn.bias_add(conv9, conv9_biases))

    with tf.variable_scope('layer10'):
        conv10_weights = tf.get_variable("weight", [1, 1, 128, 128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv10_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv10 = tf.nn.conv2d(relu9, conv10_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu10 = tf.nn.relu(tf.nn.bias_add(conv10, conv10_biases))

    # with tf.variable_scope('layer11'):
    #     conv11_weights = tf.get_variable("weight",[3,3,512,512],
    #                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
    #     conv11_biases = tf.get_variable("bias",[512],initializer=tf.constant_initializer(0.0))
    #     conv11 = tf.nn.conv2d(relu10,conv11_weights,strides=[1,1,1,1],padding='SAME')
    #     relu11 = tf.nn.relu(tf.nn.bias_add(conv11,conv11_biases))
    #
    # with tf.variable_scope('layer12'):
    #     conv12_weights = tf.get_variable("weight", [3, 3, 512, 512],
    #                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
    #     conv12_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.0))
    #     conv12 = tf.nn.conv2d(relu11, conv12_weights, strides=[1, 1, 1, 1], padding='SAME')
    #     relu12 = tf.nn.relu(tf.nn.bias_add(conv12, conv12_biases))
    #
    # with tf.variable_scope('layer13'):
    #     conv13_weights = tf.get_variable("weight", [1, 1, 512, 512],
    #                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
    #     conv13_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.0))
    #     conv13 = tf.nn.conv2d(relu12, conv13_weights, strides=[1, 1, 1, 1], padding='SAME')
    #     relu13 = tf.nn.relu(tf.nn.bias_add(conv13, conv13_biases))

    shape = relu10.get_shape().as_list()
    nodes = shape[1]*shape[2]*shape[3]
    reshaped = tf.reshape(relu10,[shape[0],nodes])
    print(shape)

    with tf.variable_scope('fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('fc2'):
        fc2_weights = tf.get_variable("weight", [1024, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('fc3'):
        fc3_weights = tf.get_variable("weight", [1024, 10],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [10], initializer=tf.constant_initializer(0.1))
        fc3 = tf.matmul(fc2, fc3_weights) + fc3_biases

    return fc3







