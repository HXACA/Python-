#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: Main.py 
@time: 2018/04/04 
"""
import videoWriter
import loadImgae
import readVideos
import pandas
import Kmeans
import numpy as np
from numba import jit
import time
import operator

def main():
    t1=time.clock()
    print u"视频读取中……"
    readVideos.readVideos('v03.avi')
    t2=time.clock()
    print "readVideo's time:"+str(t2-t1)
    imgList = np.mat(loadImgae.load_img('images'))
    t3 = time.clock()
    print "loadImage's time:" + str(t3-t2)
    print u"关键帧提取中……"
    myCentroids,clustAssing = Kmeans.biKmeans(imgList,int(np.shape(imgList)[0]/30)+1)
    t4 = time.clock()
    print "Kmeans's time:" + str(t4 - t3)
    ans = []
    for i in range(int(np.shape(imgList)[0]/30)+1):
        ptsInCurrCluster = np.nonzero(clustAssing[:,0].A==i)[0]
        ans.append(ptsInCurrCluster)
    for i in range(len(ans)):
        print ans[i]
    print u"摘要生成中……"
    videoWriter.videoWriter(ans,'v03.avi',clustAssing)
    t5 = time.clock()
    print "Video Write's time:" + str(t5 - t4)
    print "Total time:" + str(t5 - t1)

if __name__ == '__main__':
    main()