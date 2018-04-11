#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: algorithm.py
@time: 2018/04/04 
"""

import numpy as np
import cv2
np.random.seed()


def distEclud(imgA,imgB):
    imgA = np.array(imgA) * 1.0
    imgB = np.array(imgB) * 1.0
    return np.sqrt(np.abs(np.dot(imgA-imgB,(imgA-imgB).T)))

def cosSimilarity(imgA,imgB):
    imgA = np.array(imgA)*1.0
    imgB = np.array(imgB)*1.0
    imgB = imgB.reshape(1,-1)
    imgA = imgA.reshape(-1, imgB.shape[1])
    try:
        u = np.sum(np.multiply(imgA,imgB),axis=1)
        d = np.sqrt(np.diag(np.dot(imgA,imgA.T)))*np.sqrt(np.diag(np.dot(imgB,imgB.T)))
        ans =np.divide(u,d+0.00001)
    except Exception as e:
        print imgA.shape,imgB.shape
        print e
    assert (1-ans).shape[0] == imgA.shape[0]
    return 1-ans

def nearestPoint(data,point,distMeas = cosSimilarity):
    dist = distMeas(point,data)
    m = np.array(point).shape[0]
    return np.min(dist)

def ProbCent(dataSet, k):
    m,n = dataSet.shape
    index = np.random.randint(0,m)
    centroids = []
    centroids.append(dataSet[index,:])
    d = [0.0 for i in range(m)]
    for l in range(1,k):
        sum = 0.0
        for i in range(m):
            d[i] = nearestPoint(dataSet[i,],centroids)
            sum+=d[i]
        d = np.array(d)*np.array(d)
        d = d/(np.sum(d))
        for i in range(1,m):
            d[i] = d[i]+d[i-1]
        prob = np.random.rand()
        for i in range(m):
            if d[i]>=prob:
                centroids.append(dataSet[i, :])
                break;
    for i in range(k):
        centroids[i] = centroids[i].tolist()
    centroids = np.array(centroids)
    return centroids

def randCent(dataSet, k):
    m,n = np.shape(dataSet)
    #数据的列数
    minJ = dataSet.min(0)
    maxJ = dataSet.max(0)
    rangeJ = maxJ-minJ
    rangeJ = np.array(rangeJ).reshape(1,-1)
    minJ = minJ.reshape(1,n)
    centroids = minJ+np.dot(np.random.rand(k,1),rangeJ)
    assert centroids.shape == (k,n)
    return centroids

def kMeans(dataset, k, distMeas=cosSimilarity, createCent=randCent):
    m = np.shape(dataset)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    #簇分配结果，第一列为索引，第二列为误差
    centroids = createCent(dataset, k)
    print(centroids.shape)
    #随机创建k个质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                #计算第i个点与k个质心的分别的距离，求最小的距离
                distJI = distMeas(centroids[j, :], dataset[i, :])
                if (distJI - minDist) < -0.00001:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                #最小值的索引发生了变化
                 clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
            #取平方更为重视距离较远的点
        #print centroids
        for cent in range(k):
            #遍历所有的质心
            ptsInclust = dataset[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            #获得第cent簇中的所有点
            centroids[cent, :] = np.mean(ptsInclust, axis=0) if len(ptsInclust) else 0
            #按照列方向进行均值计算，求均值
    return centroids, clusterAssment

def kMeansP(dataset, k, distMeas=distEclud, createCent=ProbCent):
    m = np.shape(dataset)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    #簇分配结果，第一列为索引，第二列为误差
    centroids = createCent(dataset, k)
    #随机创建k个质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                #计算第i个点与k个质心的分别的距离，求最小的距离
                distJI = distMeas(centroids[j, :], dataset[i, :])
                if (distJI-minDist)< -0.00001:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                #最小值的索引发生了变化
                 clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
            #取平方更为重视距离较远的点
        #print centroids
        for cent in range(k):
            #遍历所有的质心
            ptsInclust = dataset[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            #获得第cent簇中的所有点
            centroids[cent, :] = np.mean(ptsInclust, axis=0) if len(ptsInclust) else 0
            #按照列方向进行均值计算，求均值
    return centroids, clusterAssment

def bikMeans(dataSet,k,distMeas = cosSimilarity):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    #0列存索引，1列存误差
    #print dataSet
    centroid0 = np.mean(dataSet,axis=0).tolist()
    centList = [centroid0]
    for i in range(m):
        clusterAssment[i,1]=distMeas(np.mat(centroid0),dataSet[i])**2

    while(len(centList)<k):
        lowestSSE = np.inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0]]
            if len(ptsInCurrCluster)==0:
                continue
            centroidMat,splitClusterAss = kMeans(ptsInCurrCluster,2,distMeas)
            sseSplit = np.sum(splitClusterAss[:,1])
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
            #print "sseSplit:, and notSplit: ", sseSplit, sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentTosplit = i
                bestNewsCent = centroidMat
                bestClustAss = splitClusterAss.copy()
                lowestSSE = sseNotSplit + sseSplit

        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentTosplit
         # 根据索引不同修改簇编号
        #print 'the bestCentToSplit is: ', bestCentTosplit
        #print 'the len of bestClustAss is ', len(bestClustAss)
        centList[bestCentTosplit] = bestNewsCent[0, :].tolist()
        centList.append(bestNewsCent[1, :].tolist())
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentTosplit)[0], :] = bestClustAss
    return np.mat(centList), clusterAssment

if __name__ == '__main__':
    imgA = cv2.imread('abstractVideo/Frame/2.jpg')
    imgB = cv2.imread('abstractVideo/Frame/3.jpg')
    imgC = cv2.imread('abstractVideo/Frame/4.jpg')
    data = []
    data.append(imgA/255.0)
    data.append(imgB/255.0)
    data.append(imgC/255.0)
    data = np.array(data).reshape(3,-1)
    print data
    data = Pca(data)
    print data.shape
    print data[0],data[1]
    print cosSimilarity(data[0],data[2])
