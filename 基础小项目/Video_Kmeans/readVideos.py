#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: ReadVideos.py 
@time: 2018/04/04 
"""

import cv2
from numba import jit
import os
import shutil

def delImg(imgDir='images'):
    imgs = os.listdir(imgDir)
    for i in range(len(imgs)):
        os.remove(imgDir+'/'+str(i)+'.jpg')

def readVideos(dir):
    delImg()
    vc = cv2.VideoCapture(dir)
    c=1;
    if vc.isOpened():
        rval = True
    else:
        rval = False
    timeF = 20
    while rval:
        rval,frame = vc.read()
        if(c%timeF==0):
            cv2.imwrite('images/'+str((c/timeF)-1)+'.jpg',frame)
        c = c+1

    vc.release()

if __name__ == '__main__':
    readVideos()
