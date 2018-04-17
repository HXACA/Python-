#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: Main.py 
@time: 2018/04/08 
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(1)

def initialize_parameters(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    #print(L)
    for i in range(1,L):
        parameters['W'+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1])*np.sqrt(2.0/layer_dims[i-1])
        parameters['b'+str(i)] = np.zeros((layer_dims[i],1))
        assert(parameters['W'+str(i)].shape == (layer_dims[i],layer_dims[i-1]))
        assert(parameters['b'+str(i)].shape == (layer_dims[i],1))

    return parameters

def linear_forward(A,W,b):
    Z = np.dot(W,A)+b
    assert (Z.shape == (W.shape[0],A.shape[1]))
    cache = (A,W,b)
    return Z,cache

def sigmoid(Z):
    A = 1.0/(1+np.exp(-Z))
    cache = Z
    assert (A.shape == Z.shape)
    return A,cache

def relu(Z):
    A = np.maximum(0,Z)
    cache = Z
    assert (A.shape == Z.shape)
    return A, cache

def linear_activation_forward(A_prev,W,b,activation):
    if activation == 'sigmoid':
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = sigmoid(Z)
    elif activation =='relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (linear_cache,activation_cache)

    return A,cache

def L_model_forward(X,parameters):
    caches = []
    A = X
    L = len(parameters)//2
    for i in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters['W'+str(i)],parameters['b'+str(i)],'relu')
        caches.append(cache)

    AL,cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],'sigmoid')
    caches.append(cache)

    assert (AL.shape == (1,X.shape[1]))
    return AL,caches


def compute_cost(AL,Y):
    m = Y.shape[1]
    cost = -(np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL)))/m
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    return cost


def linear_backward(dZ,cache):
    A_prev,W,b=cache
    m = A_prev.shape[1]
    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ,axis =1,keepdims=True)/m
    dA_prev = np.dot(W.T,dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev,dW,db

def relu_backward(dA,cache):
    Z = cache
    #print(cache)
    dZ = np.array(dA,copy=True)
    dZ[Z<=0]=0
    assert (dZ.shape == Z.shape)
    #print(dZ)
    return dZ

def sigmoid_backward(dA,cache):
    Z = cache
    s = 1.0/(1+np.exp(-Z))
    dZ = dA*(1-s)*s
    assert (dZ.shape == Z.shape)
    return dZ

def linear_activation_backward(dA,cache,activation):
    linear_cache,activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA,activation_cache)
        dA_prev ,dW,db = linear_backward(dZ,linear_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev,dW,db

def L_model_backward(AL,Y,caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y,AL)-np.divide(1-Y,1-AL))

    current_cache = caches[L-1]
    grads["dA"+str(L)],grads["dW"+str(L)],grads['db'+str(L)]=linear_activation_backward(dAL,current_cache,'sigmoid')

    for i in reversed(range(L-1)):
        current_cache = caches[i]
        dA_prev_temp,dW_temp,db_temp = linear_activation_backward(grads['dA'+str(i+2)],current_cache,'relu')
        grads["dA" + str(i+1)] = dA_prev_temp
        grads["dW" + str(i + 1)] = dW_temp
        grads["db" + str(i + 1)] = db_temp

    return grads

def update_parameters(parameters,grads,learning_rate):
    L = len(parameters)//2
    for i in range(L):
        parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] - learning_rate * grads["dW" + str(i + 1)]
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * grads["db" + str(i + 1)]

    return parameters



def load_data():
    train_x_orig = []
    train_y_orig = []
    test_x_orig = []
    test_y_orig = []

    imgs = os.listdir('train/1')
    for img in imgs:
        imgpath =  os.path.join('train/1', img)
        img =  cv2.imread(imgpath)
        try:
            img = cv2.resize(img, (64, 64))
        except:
            continue
        train_x_orig.append(img)
        train_y_orig.append(1)

    imgs = os.listdir('train/0')
    for img in imgs:
        imgpath = os.path.join('train/0', img)
        img = cv2.imread(imgpath)
        try:
            img = cv2.resize(img, (64, 64))
        except:
            continue
        train_x_orig.append(img)
        train_y_orig.append(0)

    imgs = os.listdir('test/1')
    for img in imgs:
        imgpath = os.path.join('test/1', img)
        img = cv2.imread(imgpath)
        try:
            img = cv2.resize(img, (64, 64))
        except:
            continue
        test_x_orig.append(img)
        test_y_orig.append(1)

    imgs = os.listdir('test/0')
    for img in imgs:
        imgpath = os.path.join('test/0', img)
        img = cv2.imread(imgpath)
        try:
            img = cv2.resize(img, (64, 64))
        except:
            continue
        test_x_orig.append(img)
        test_y_orig.append(0)

    return np.array(train_y_orig),np.array(train_x_orig),np.array(test_x_orig),np.array(test_y_orig)


def L_layer_model(X,Y,layer_dims,learning_rate = 0.005,num_iterations = 5000,print_cost = False):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters(layer_dims)
    ans = []
    bestCost = np.inf
    for i in range(0,num_iterations):
        AL,caches = L_model_forward(X,parameters)
        cost = compute_cost(AL,Y)
        grads = L_model_backward(AL,Y,caches)
        parameters = update_parameters(parameters,grads,learning_rate)
        if cost < bestCost:
            ans = parameters
        if i%100 ==0:
            print("Cost after iteration %i:%f" %(i,cost))
            costs.append(cost)
    return ans

def predict(X,Y):
    parameters = grabParameters()
    AL, caches = L_model_forward(X, parameters)
    result = np.int64(AL>0.5)
    print(result)
    return np.sum((result == Y))*1.0/Y.shape[1]

def storeParmeters(parameters):
    import json
    parameters = change(parameters,1)
    with open('parameters.json','w') as json_file:
        json_file.write(json.dumps(parameters))

def grabParameters():
    import json
    with open('parameters.json') as json_file:
        data = json.load(json_file)
        data = change(data,0)
        return data

def change(data,k):
    if k:
        for i in range(1,len(data)//2+1):
            data['W'+str(i)] = data['W'+str(i)].tolist()
            data['b'+str(i)] = data['b'+str(i)].tolist()
    else:
        for i in range(1,len(data) // 2+1):
            data['W' + str(i)] = np.array(data['W' + str(i)])
            data['b' + str(i)] = np.array(data['b' + str(i)])
    return data

def train():
    train_y_orig, train_x_orig, test_x_orig, test_y_orig = load_data()
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    test_y = test_y_orig.reshape(1, -1)
    train_y = train_y_orig.reshape(1, -1)
    train_x = train_x_flatten / 255.0
    test_x = test_x_flatten / 255.0
    layers_dims = [train_x_flatten.shape[0], 20,7,5,1]
    parameters = L_layer_model(train_x, train_y, layers_dims)
    storeParmeters(parameters)
    ans = predict(train_x,train_y)
    print(u'训练集的正确率为：'+str(ans*100)+'%')
    ans = predict(test_x, test_y)
    print(u'测试集的正确率为：'+str(ans*100)+'%')

def test():
    while 1:
        str = input(u'输入文件名及路径：')
        img = cv2.imread(str)
        img = cv2.resize(img, (64, 64))

        img = img/255.0
        parameters = grabParameters()
        X = img.reshape(1,-1).T
        AL, caches = L_model_forward(X, parameters)
        print(AL)
        print('Yes' if AL>0.5 else 'No')

if __name__ == '__main__':
    test()

