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

def reName(Dir='abstractVideo/newVideo'):
    videos = os.listdir(Dir)
    num = 1
    for video in videos:
        videoPath = os.path.join(Dir, video)
        os.rename(videoPath,os.path.join(Dir,'abstractVideo'+str(num)+'.avi'))
        num+=1

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

def videoWriter(ans,dir,bestCluster,bestCentroids,imgList,det=0):
    delVideo()
    delVideo('abstractVideo/newVideo')
    delImg()
    vc = cv2.VideoCapture(dir)
    fps = vc.get(cv2.cv.CV_CAP_PROP_FPS)
    size = (int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

    videowriter = []
    for i in range(len(ans)):
        videowriter.append(cv2.VideoWriter("abstractVideo/Video/abstractVideo" + str(i+1) + '.avi', cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), fps, size))
    if det!=0:
        for i in range(len(ans)):
            videowriter.append(cv2.VideoWriter("abstractVideo/newVideo/abstractVideo" + str(i+1) + '.avi', cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), fps, size))
    vc.release()

    imgs = os.listdir('images')
    count = 0
    files = []

    for Img in imgs:
        imgPath = os.path.join('images', Img)
        img = cv2.imread(imgPath)
        temp = bestCluster[count , 0]
        videowriter[int(temp)].write(img)
        if det!=0:
            videowriter[int(temp)+len(ans)].write(img)
        files.append(imgPath)
        count+=1

    for i in range(len(videowriter)):
        videowriter[i].release()

    num = 1
    for i in range(len(bestCentroids)):
        dist = cosSimilarity(imgList,bestCentroids[i])
        dist = dist.tolist()
        index = dist.index(min(dist))
        img = cv2.imread(files[index])
        if det!=0:
            res = 0
            for j in range(len(ans[i])):
                img2 = cv2.imread(files[np.nonzero(bestCluster[:, 0].A == i)[0][j]])
                if check(img2) == True:
                    res+=1
                    break
            if res == 0:
                os.remove("abstractVideo/newVideo/abstractVideo" + str(i+1) + '.avi')
        cv2.imwrite('abstractVideo/Frame/'+str(num)+'.jpg', img)
        num+=1

    reName()
