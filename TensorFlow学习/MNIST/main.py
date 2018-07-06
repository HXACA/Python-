#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: main.py 
@time: 2018/07/06 
"""
# TensorFlow封装了一个处理MNIST数据集的类
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#每张图片向量表示的大小，28*28
INPUT_NODE = 784
#需要分类的种类，0~9个数字
OUTPUT_NODE = 10

#隐藏层1的节点数量
LAYER1_NODE = 500

#小批量梯度下降的每次更新数据数
BATCH_SIZE = 128

#学习速率和学习速率衰减率
LEARNING_RATE = 0.8
LEARNING_RATE_DECAY = 0.99

#正则化系数
REGULARIZATION_RATE = 0.0001

#迭代次数
TRAINING_STEPS = 30000
#滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    # 不存在滑动平均类,激活函数为relu
    if avg_class==None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        layer2 = tf.matmul(layer1,weights2)+biases2
        return layer2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        layer2 = tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)
        return layer2


def inference2(input_tensor, avg_class,reuse=False):
    if avg_class == None:
        with tf.variable_scope('layer1',reuse=reuse):
            weights = tf.get_variable('weights', [INPUT_NODE, LAYER1_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

        with tf.variable_scope('layer2',reuse=reuse):
            weights = tf.get_variable('weights', [LAYER1_NODE, OUTPUT_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
            layer2 = tf.matmul(layer1, weights) + biases

    else:
        with tf.variable_scope('layer1', reuse=reuse):
            weights = tf.get_variable('weights', [INPUT_NODE, LAYER1_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
            layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights)) + avg_class.average(biases))

        with tf.variable_scope('layer2', reuse=reuse):
            weights = tf.get_variable('weights', [LAYER1_NODE, OUTPUT_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
            layer2 = tf.matmul(layer1, avg_class.average(weights)) + avg_class.average(biases)

    return layer2

def train():
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    x_input = tf.placeholder(tf.float32,[None,INPUT_NODE],name="x_input")
    y_input = inference2(x_input,None)

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    #计算前向传播
    y = inference2(x_input,None,True)
    #代表训练步数
    global_step = tf.Variable(0,trainable=False)
    #滑动平均，加快训练早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    #对所有参数变量应用滑动平均
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    #计算滑动平均下的预测值
    average_y = inference2(x_input,None,True)
    #计算平均交叉熵
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_input,logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #添加l2正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1)+regularizer(weights2)
    loss = cross_entropy_mean+regularization
    #设置学习速率指数衰减
    learning_rate = tf.train.exponential_decay(LEARNING_RATE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
    #优化器会使得global_step自增
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    #同时进行两个步骤，方向传播更新参数和更新每个参数的滑动平均值
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name="train")

    #计算滑动平均下的正确预测个数
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_input,1))
    #转为正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #开始训练
    with tf.Session() as sess:
        #初始化参数
        tf.global_variables_initializer().run()
        #验证集
        validate_feed = {x_input:mnist.validation.images,y_input:mnist.validation.labels}
        #测试集
        test_feed = {x_input:mnist.test.images,y_input:mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i %1000 == 0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print("After %d training steps,validation accuracy is %g" %(i,validate_acc))
            #从MNIST数据集中读取一部分
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x_input:xs,y_input:ys})

        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("After training ,test accuracy is %g" % (test_acc))


def save():
    v = tf.Variable(0,dtype=tf.float32,name="v")
    ema = tf.train.ExponentialMovingAverage(0.99)

    # maintain_averages_op = ema.apply(tf.all_variables())
    # for variables in tf.all_variables():
    #     print(variables.name)
    saver = tf.train.Saver(ema.variables_to_restore())
    print(ema.variables_to_restore())
    with tf.Session() as sess:
        # init_op = tf.global_variables_initializer()
        # sess.run(init_op)
        # sess.run(tf.assign(v,10))
        # sess.run(maintain_averages_op)
        saver.restore(sess,"./model.ckpt")
        print(sess.run(v))

if __name__ == '__main__':
    save()