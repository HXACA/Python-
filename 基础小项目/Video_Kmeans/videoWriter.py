#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: videoWriter.py 
@time: 2018/04/05 
"""
import cv2
import os

def delVideo(Dir='abstractVideo/Video'):
    videos = os.listdir(Dir)
    for i in range(len(videos)):
        os.remove(Dir+'/abstractVideo'+str(i+1)+'.avi')

def delImg(imgDir='abstractVideo/Frame'):
    imgs = os.listdir(imgDir)
    for i in range(len(imgs)):
        os.remove(imgDir+'/abstractFrame'+str(i+1)+'.jpg')

def videoWriter(ans,dir,bestCluster):
    delVideo()
    delImg()
    vc = cv2.VideoCapture(dir)
    fps = vc.get(cv2.cv.CV_CAP_PROP_FPS)
    size = (int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    success,frame = vc.read()
    vis = {}
    t=0
    now = 0
    videowriter = []
    for i in range(len(ans)):
        videowriter.append(cv2.VideoWriter("abstractVideo/Video/abstractVideo" + str(i+1) + '.avi', cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), fps, size))
    while success:
        if (now+5)%20==0:
            temp = bestCluster[(now+5)/20-1,0]
            for i in range(10):
                if success:
                    videowriter[int(temp)].write(frame)
                now+=1
                success, frame = vc.read()
        now+=1
        success, frame = vc.read()
