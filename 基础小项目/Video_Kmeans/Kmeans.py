#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: Kmeans.py 
@time: 2018/04/04 
"""

import numpy as np
from numba import jit

@jit
def distEclud(imgA,imgB):
    return np.sqrt(np.dot(imgA-imgB,(imgA-imgB).T))


def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    #数据的列数
    centroids = np.mat(np.zeros((k, n)))
    #构造k行n列的全零矩阵
    for j in range(n):
        minJ = min(dataSet[:, j]) if len(dataSet[:,j])!=0 else 0
        #当前列最小值
        #print "%d" %minJ
        rangeJ = float(max(dataSet[:, j])-minJ)
        #当前列数据范围
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
        #表示随机出一个k*1的array，这里用rangeJ与随机数相乘，保证结果在范围之内
    return centroids


def kMeans(dataset, k, distMeas=distEclud, createCent=randCent):
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
                if distJI<minDist:
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


def biKmeans(dataSet,k,distMeas = distEclud):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    #0列存索引，1列存误差
    #print dataSet
    centroid0 = np.mean(dataSet,axis=0).tolist()[0]
    #print centroid0
    centList = [centroid0]
    for i in range(m):
        clusterAssment[i,1]=distMeas(np.mat(centroid0),dataSet[i])**2

    while(len(centList)<k):
        lowestSSE = np.inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0]]
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
        centList[bestCentTosplit] = bestNewsCent[0, :].tolist()[0]
        centList.append(bestNewsCent[1, :].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentTosplit)[0], :] = bestClustAss
    return np.mat(centList), clusterAssment


