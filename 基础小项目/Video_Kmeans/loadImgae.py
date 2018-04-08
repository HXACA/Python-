#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: loadImgae.py 
@time: 2018/04/04 
"""
import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
from PIL import Image

def Pca(data):
    #print data
    #data = data.reshape(1,-1)
    c = 10
    pca = PCA(n_components=c)
    pca.fit(data)
    PCA(copy=True, n_components=c, whiten=False)
    return pca.explained_variance_ratio_

def getPixel(imgDir,i):
    img = cv2.imread(imgDir + '/' + str(i) + '.jpg')
    img = cv2.resize(img,(9,8))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    difference = []
    for i in range(8):
        for j in range(8):
            difference.append(img[i][j]>img[i][j+1])
    return difference

def load_img(imgDir):
    imgs = os.listdir(imgDir)
    imgNum = len(imgs)
    print u"样本数量为："+str(imgNum)
    dataList = []
    for i in range(imgNum):
        img = cv2.imread(imgDir+'/'+str(i)+'.jpg')
        img = cv2.resize(img, (32, 32))
        HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        H,S,V = cv2.split(HSV)
        data=[]
        data.extend(H/360.0)
        data.extend(S/100.0)
        data.extend(V/100.0)
        #data = Pca(data)
        data = np.array(data).reshape(-1)
        dataList.append(data)
    return dataList

if __name__ == '__main__':
    data = load_img('images')