#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: videoWriter.py 
@time: 2018/04/05 
"""
import cv2
import os
import numpy as np

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

def delVideo(Dir='abstractVideo/Video'):
    videos = os.listdir(Dir)
    for video in videos:
        videoPath = os.path.join(Dir, video)
        os.remove(videoPath)

def delImg(imgDir='abstractVideo/Frame'):
    imgs = os.listdir(imgDir)
    for img in imgs:
        imgPath = os.path.join(imgDir, img)
        os.remove(imgPath)


def videoWriter(ans,dir,bestCluster,bestCentroids,imgList):
    delVideo()
    delImg()
    vc = cv2.VideoCapture(dir)
    fps = vc.get(cv2.cv.CV_CAP_PROP_FPS)
    size = (int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    videowriter = []
    for i in range(len(ans)):
        videowriter.append(cv2.VideoWriter("abstractVideo/Video/abstractVideo" + str(i+1) + '.avi', cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), fps, size))
    success = vc.isOpened()
    now = 0
    while success:
        success, frame = vc.read()
        if (now)%10 == 0 and (now)/10<len(bestCluster) and success:
            temp = bestCluster[(now)/10,0]
            videowriter[int(temp)].write(frame)
        now+=1
    vc.release()
    for i in range(len(bestCentroids)):
        dist = cosSimilarity(imgList,bestCentroids[i])
        print dist.shape
        dist = dist.tolist()
        index = dist.index(min(dist))
        img = cv2.imread('images/'+str(index).zfill(4)+'.jpg')
        cv2.imwrite('abstractVideo/Frame/'+str(i+1)+'.jpg', img)
