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

def cosSimilarity(imgA,imgB):
    imgA = np.array(imgA)*1.0
    imgB = np.array(imgB)*1.0
    imgB = imgB.reshape(1,-1)
    imgA = imgA.reshape(-1, imgB.shape[1])
    try:
        u = np.sum(np.multiply(imgA,imgB),axis=1)
        d = np.sqrt(np.diag(np.dot(imgA,imgA.T)))*np.sqrt(np.diag(np.dot(imgB,imgB.T)))
        ans =np.divide(u,d)
    except Exception as e:
        print np.array(imgA).shape,imgB.shape
        print e
    assert (1-ans).shape[0] == imgA.shape[0]
    return ans

def Pca(data):
    c = 7
    pca = PCA(n_components=c)
    pca.fit(data)
    PCA(copy=True, n_components=c, whiten=False)
    return pca.transform(data)

def getPixel(imgDir,i):
    img = cv2.imread(imgDir + '/' + str(i) + '.jpg')
    img = cv2.resize(img,(9,8))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    difference = []
    for i in range(8):
        for j in range(8):
            difference.append(img[i][j]>img[i][j+1])
    return difference

def statmoments(p,n):
    L = len(p)
    G = L -1
    p = p*1.0/sum(p)
    z = [i for i in range(G+1)]
    z = np.array(z)/(G*1.0)
    m = np.dot(z,p)
    z = z-m
    v = []
    v.append(m)
    for j in range(2,n+1):
        v.append(np.dot((np.power(z,j)),p))
    unv = []
    unv.append(m*G)
    for j in range(2,n+1):
        unv.append(np.dot((np.power((z*G),j)),p))
    return v,unv

def getHSV(H,S,V):
    #print H,S,V
    H = (H/22.5).astype(int)
    S = (S/25).astype(int)
    V = (V/25).astype(int)
    data = []
    L = H * 16 + S * 4 + V
    # print(L[L==1])
    for i in range(256):
        # print i
        data.append(len(L[L == i]))
    data = np.array(data) * 1.0 / sum(data)
    return data

def getTexture(img):
    data = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imhist, bins = np.histogram(img, range(257))  # 直方图
    imhist = imhist / (img.size * 1.0)
    L = len(imhist)
    u, mu = statmoments(imhist, 3)
    data.append(mu[0])
    data.append(np.sqrt(mu[1]))
    varn = mu[1] / (np.power(L - 1, 2))
    data.append(1 - 1 / (1 + varn))
    data.append(mu[2] / (np.power(L - 1, 2)))
    data.append(np.sum(np.dot(imhist, imhist.T)))
    data.append(np.sum(imhist * np.log2(imhist + 0.0001)))
    return data

def check(img):
    pface = cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_profileface.xml')
    fface = cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faceRects = fface.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects)>0:
        return True
    faceRects = pface.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects)>0:
        return True
    gray = cv2.flip(gray,1)
    faceRects = pface.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects) > 0:
        return True
    return False

def load_img(imgDir):
    imgs = os.listdir(imgDir)
    imgNum = len(imgs)
    print u"样本数量为："+str(imgNum)
    dataList = []
    count = 0
    for Img in imgs:
        imgPath = os.path.join(imgDir,Img)
        img = cv2.imread(imgPath)
        try:
            HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        except:
            print(str(imgPath))
        H,S,V = cv2.split(HSV)
        data = getHSV(H*2,S/255.0*100,V/255.0*100)
        data = data.tolist()
        #data.extend(getTexture(img))
        dataList.append(data)
        count +=1
    dataList = np.array(dataList).reshape(count,-1)
    print(np.array(dataList).shape)
    dataList = Pca(dataList)
    return np.array(dataList)

if __name__ == '__main__':
    data = load_img('images')
    print(np.array(data).shape)