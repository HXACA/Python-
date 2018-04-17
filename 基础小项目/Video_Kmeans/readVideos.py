#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: ReadVideos.py 
@time: 2018/04/04 
"""

import cv2
import os


def delImg(imgDir='images'):
    imgs = os.listdir(imgDir)
    for img in imgs:
        imgPath = os.path.join(imgDir,img)
        os.remove(imgPath)

def readVideos(dir):
    delImg()
    print(dir)
    vc = cv2.VideoCapture(dir)
    print(type(vc))
    c=0;
    if vc.isOpened():
        rval = True
    else:
        rval = False
    timeF = 10
    while rval:
        rval,frame = vc.read()
        if(c%timeF==0 and rval):
            cv2.imwrite('images/'+str((c/timeF)).zfill(4)+'.jpg',frame)
        c = c+1

    vc.release()

if __name__ == '__main__':
    readVideos()
