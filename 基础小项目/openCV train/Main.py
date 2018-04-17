#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: Main.py 
@time: 2018/04/16 
"""

import cv2
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def Main():
    Imgs = os.listdir('poss')
    num = 912
    for Img in Imgs:
        imgPath = os.path.join('poss',Img)
        img = cv2.imread(imgPath)
        try:
            img = cv2.flip(img,2)
            cv2.imwrite('poss/'+str(num)+'.jpg',img)
            num += 1
        except Exception as e:
            print(e)
            os.remove(imgPath)
            continue

def check(img):
    pface = cv2.CascadeClassifier('E:\Python的杂七杂八\基础小项目\openCV train\dt\cascade.xml')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('test',gray)
    # cv2.waitKey()
    faceRects = pface.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects)>0:
        return True
    return False

def textEdit():
    fr = open('neg/neg.txt','w')
    for i in range(5400):
        fr.write('neg/'+str(i+1)+'.jpg'+'\n')

if __name__ == '__main__':
    Main()
    #textEdit()
