#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: Transfer.py
@time: 2018/07/07
"""
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# 瓶颈层的节点数
BOTTLEENECK_TENSOR_SIZE = 2048

BOTTLEENECK_TENSOR_NAME = 'pool_3/_reshape:0'

JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

MODEL_DIR = './inception_dec_2015/'
MODEL_FILE = 'tensorflow_inception_graph.pb'

CACHE_DIR = './tmp/bottleneck'

INPUT_DATA = './flower_photos'

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

LEARNING_RATE = 0.01
STEPS = 5000
BATCH = 128

#构造数据集
def create_image_lists(testing_percentage,validation_percentage):
    result ={}
    # 得到数据集下所有文件夹的路径
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    #根节点标记
    is_root_dir = True
    #遍历
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        #图片文件可能的后缀名
        extensions = ['jpg','jpeg','JPG','JPEG']
        #保存该分类的文件
        file_list = []
        #返回path最后的文件名也就是该分类名
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            #拼接成图片文件可能的名字的字符串
            file_glob = os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            #返回所有匹配的文件路径列表
            file_list.extend(glob.glob(file_glob))
        #文件夹下无文件，直接处理下一个文件夹
        if not file_list:
            continue
        # 分类名用小写字母保存
        label_name = dir_name.lower()
        #训练集，测试集，验证集
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            chance = np.random.randint(100)
            if chance<validation_percentage:
                validation_images.append(base_name)
            elif chance<(validation_percentage+testing_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        #保存该分类信息，分类名，该分类的训练，测试，验证集
        result[label_name] = {
            'dir':dir_name,
            'training':training_images,
            'testing':testing_images,
            'validation':validation_images,
        }
    return result

#根据图片信息，查找图片的地址
def get_image_path(image_lists,img_dir,label_name,index,category):
    # 得到属于此label的所有图片的信息
    label_list = image_lists[label_name]
    # 得到属于此测试/训练/验证集的所有图片信息
    category_list = label_list[category]
    mod_index = index%len(category_list)
    # 获取文件名
    base_name = category_list[mod_index]
    # 获取分类名
    sub_dir = label_list['dir']
    full_path = os.path.join(img_dir,sub_dir,base_name)
    return full_path

def get_bottleneck_path(image_lists,label_name,index,category):
    return get_image_path(image_lists,CACHE_DIR,label_name,index,category)+'.txt'

def run_bottelneck_on_image(sess,image_data,image_data_tensor,bottleneck_tensor):
    #根据传入的图片信息，计算新的特征向量
    bottleneck_values = sess.run(bottleneck_tensor,{image_data_tensor:image_data})
    # 压缩为一个一维向量
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


#获取一张图片经过Inception-v3模型处理后的向量
#参数分别为：会话信息，数据集信息，类别名，图片的序号，所属训练/测试/验证集，输入层张量，瓶颈层张量
def get_or_create_bottleneck(sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor):
    label_list = image_lists[label_name]
    sub_dir = label_list['dir']
    sub_dir_path = os.path.join(CACHE_DIR,sub_dir)
    #查找该分类的文件夹是否已存在
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    #获得处理后的特征向量文件地址
    bottleneck_path = get_bottleneck_path(image_lists,label_name,index,category)
    #查找特征向量文件是否存在
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists,INPUT_DATA,label_name,index,category)
        image_data = gfile.FastGFile(image_path,'rb').read()
        #计算特征向量
        bottleneck_values = run_bottelneck_on_image(sess,image_data,jpeg_data_tensor,bottleneck_tensor)
        #逗号分隔，保存特征向量
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        #写入文件
        with open(bottleneck_path,'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path,'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values

def get_random_cached_bottlenecks(sess,n_classes,image_lists,how_many,category,
                                  jpeg_data_tensor,bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        #获取该图片经过模型计算的特征向量
        bottleneck = get_or_create_bottleneck(sess,image_lists,label_name,image_index,category,
                                              jpeg_data_tensor,bottleneck_tensor)
        ground_truth = np.zeros(n_classes,dtype=np.float32)
        #代表结果的one-hot向量
        ground_truth[label_index]=1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks,ground_truths

def get_test_bottlennecks(sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    lable_name_list = list(image_lists.keys())
    for label_index,label_name in enumerate(lable_name_list):
        category = 'testing'
        for index,unnused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess,image_lists,label_name,index,category,
                                                  jpeg_data_tensor,bottleneck_tensor)
            ground_truth = np.zeros(n_classes,dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks,ground_truths

def main(_):
    #读取训练集所有照片
    image_lists = create_image_lists(TEST_PERCENTAGE,VALIDATION_PERCENTAGE)
    #目标分类的数目
    n_classes = len(image_lists.keys())
    #读取已训练好的模型，并解析为对应的GraphDef Protocol Buffer中
    with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_FILE),'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    #将graph_def中的图加载到当前图中，返回的张量名称是瓶颈层和图像输入层的名称
    bottleneck_tensor,jpeg_data_tensor = tf.import_graph_def(graph_def,
                                            return_elements=[BOTTLEENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME])
    bottleneck_input = tf.placeholder(tf.float32,[None,BOTTLEENECK_TENSOR_SIZE],
                                      name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32,[None,n_classes],name='GroundTruthInput')
    #添加全连接层
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLEENECK_TENSOR_SIZE,n_classes],stddev=0.1))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input,weights)+biases
        final_tensor = tf.nn.softmax(logits)
    #计算损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_input,logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #定义优化器
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    #计算准确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.arg_max(final_tensor,1),tf.arg_max(ground_truth_input,1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(STEPS):
            train_bottlenecks,train_ground_truth = get_random_cached_bottlenecks(sess,n_classes,image_lists,
                                                            BATCH,'training',jpeg_data_tensor,bottleneck_tensor)
            sess.run(train_step,feed_dict={bottleneck_input:train_bottlenecks,ground_truth_input:train_ground_truth})
            if i%100 ==0 or i+1 ==STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(sess, n_classes, image_lists,
                                                                        BATCH, 'validation',jpeg_data_tensor,bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step,feed_dict={bottleneck_input:validation_bottlenecks,
                                                                          ground_truth_input:validation_ground_truth})
                print("Step %d: Validation accuracy on random sampled %d examples = %.lf%%" %(i,BATCH,validation_accuracy*100))

        test_bottlenecks, test_ground_truth = get_test_bottlennecks(sess, image_lists, n_classes,
                                                                    jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks,
                                                             ground_truth_input: test_ground_truth})
        print("Final test accuracy = %.lf%%" % (test_accuracy * 100))


if __name__ == '__main__':
    tf.app.run()