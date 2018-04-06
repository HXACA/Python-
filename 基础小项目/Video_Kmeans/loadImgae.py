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

def autoNorm(k):
    k.reshape(1,np.shape(k)[0]*np.shape(k)[1])
    m = k.shape[0]
    minVal = k.min()
    maxVal = k.max()
    ranges = maxVal - minVal
    nk = np.zeros(np.shape(k))
    nk = k - np.tile(minVal,(m,1))
    nk = k*1.0/np.tile(ranges,(m,1))
    nk = Pca(nk)
    #print nk
    return nk


def Pca(data):
    c = min(np.shape(data)[0],np.shape(data)[1])
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
        img = cv2.resize(img, (np.shape(img)[0]/10, np.shape(img)[1]/10))
        HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        H,S,V = cv2.split(HSV)
        data=[]
        data.append(autoNorm(H))
        data.append(autoNorm(S))
        data.append(autoNorm(V))
        data.append(autoNorm(gray))
        data = np.array(data).reshape(np.shape(data)[0]*np.shape(data)[1])
        dataList.append(data)
        #print np.shape(dataList)
    return dataList

if __name__ == '__main__':
    data = load_img('images')