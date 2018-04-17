#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: Main.py
@time: 2018/04/04
"""
import videoWriter
import loadImgae
import readVideos
import algorithm
import numpy as np
import time
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def distEclud(imgA,imgB):
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
        print np.array(imgA).shape,imgB.shape
        print e
    assert (1-ans).shape[0] == imgA.shape[0]
    return 1-ans

def run(algorithm,imgList,k):
    myCentroids, clustAssing = algorithm(imgList, k)
    return myCentroids, clustAssing

def getA(imgList,index,b,distMeas = distEclud):#数据集，i向量下标，目标簇
    if len(b)==0:
        return 0
    imgList = np.array(imgList)
    dist = distMeas(imgList[b],imgList[index])
    if dist.shape[0]>1:
        sum = np.sum(np.diag(dist),axis=0)
    else:
        sum = np.sum(dist, axis=0)
    sum = sum*1.0/len(b)
    return sum

def getB(imgList,Cindex,index,ans):
    minDis = np.inf
    for i in range(len(ans)):
        if i == Cindex:
            continue
        minDis = min(minDis,getA(imgList,index,ans[i]))
    return minDis

def getCoefficient(imgList,ans):
    imgList = np.array(imgList)
    sum = 0.0
    for i in range(len(ans)):
        if len(ans[i])==1:
            continue
        for j in range(len(ans[i])):
            a = getA(imgList,ans[i][j],ans[i])
            b = getB(imgList,i,ans[i][j],ans)
            sum += (b-a)/(max(a,b))
    sum = sum/imgList.shape[0]
    return sum

def work(videoStr,tk=-1,det=0):
    videoStr = str(videoStr)
    print u"准备处理 '"+str(videoStr)+u"' 文件！"
    print u"视频读取中……"
    t1=time.clock()
    readVideos.readVideos(str(videoStr))
    t2=time.clock()
    print "readVideo's time:"+str(t2-t1)+"s"

    imgList = np.array(loadImgae.load_img('images'))
    t3 = time.clock()
    print "loadImage's time:" + str(t3-t2)+"s"
    bestans = []
    bestClust = []
    bestCentroids = []
    maxx = -1
    bestk = -1
    print u"关键帧提取中……"
    t4 = time.clock()
    l = 2
    r = imgList.shape[0]/40+1
    if tk!=-1:
        l=tk
        r=tk+1
    for k in range(l,r):
        kans = []
        kclust = []
        kmax = -1
        kCentroids = []
        ave = 0.0

        for j in range(10):
            #myCentroids, clustAssing= run(algorithm.kMeans, imgList, k)
            myCentroids, clustAssing= run(algorithm.kMeansP, imgList, k)
            #myCentroids,clustAssing= run(algorithm.bikMeans, imgList, k)
            ans = []
            for i in range(k):
                ptsInCurrCluster = np.nonzero(clustAssing[:,0].A==i)[0]
                ans.append(ptsInCurrCluster)
            Coefficient = getCoefficient(imgList,ans)
            ave += Coefficient

            if Coefficient>kmax:
                kmax = Coefficient
                kans = ans
                kCentroids = myCentroids
                kclust = clustAssing

        ave = ave/10.0
        print(u"k为"+str(k)+u"时的轮廓系数为："+str(ave))
        if ave>maxx:
            maxx = kmax
            bestans = kans
            bestClust = kclust
            bestk = k
            bestCentroids = kCentroids

    t5 = time.clock()
    print "Kmeans 's time:" + str(t5 - t4) + "s"
    print str(bestk)+u"段摘要生成中……"
    videoWriter.videoWriter(bestans,videoStr,bestClust,bestCentroids,imgList,det=det)
    t6 = time.clock()
    print "Video Write's time:" + str(t6 - t5)+"s"
    print "Total time:" + str(t6 - t1)+"s"
    print " "
    return bestk

def loadDataSet(filename):
    dataMat=[]
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #以tab进行分割
        fltLine = map(float, curLine)
        #对每个元素进行float强转
        dataMat.append(fltLine)
    return dataMat

def clusterClubs(numClust=5):
    import matplotlib
    import matplotlib.pyplot as plt
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    myCentroids, clustAssing = run(algorithm.bikMeans,datMat,numClust)
    #调用二分kMeans
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    #坐标 ，x方向的长度，y方向的长度
    scatterMarkers = ['s','o','^','8','p','d','v','h','>','<']
    #绘制的点的样式
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0',**axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon = False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:,0].A == i)[0],:]
        markerStyle = scatterMarkers[i%len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0],\
                    ptsInCurrCluster[:,1].flatten().A[0],\
                    marker=markerStyle,s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0],\
                myCentroids[:,1].flatten().A[0],marker='+',s=300)
    plt.show()

if __name__ == '__main__':
    videos = os.listdir(u'E:/Python的杂七杂八/基础小项目/Video_Kmeans/train/')
    for video in videos:
        videoPath = os.path.join(u'E:/Python的杂七杂八/基础小项目/Video_Kmeans/train/',video)
        work(videoPath,det=1)
        try:
            s = raw_input("处理下一条视频吗？（Y/N)")
        except Exception as e:
            s = ' '
        if s.upper == 'N':
            break
        else:
            continue