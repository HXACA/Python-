#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: main.py 
@time: 2018/06/30 
"""
import cv2
import numpy as np
global source, point1, point2,target,point
import test

import matplotlib.pyplot as plt
import time
import threading

def on_mouse(event, x, y, flags, param):
    global source, point1, point2,point
    global number
    img2 = source.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x,y)
        cv2.circle(img2, point1, 10, (0,255,0), 5)
        cv2.imshow('source', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               #按住左键拖曳
        cv2.rectangle(img2, point1, (x,y), (255,0,0), 5)
        cv2.imshow('source', img2)
    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        point2 = (x,y)
        if number<4:
            cv2.rectangle(img2, point1, point2, (0,0,255), 5)
        else:
            cv2.rectangle(img2, point1, point2, (255, 0, 0), 5)
        cv2.imshow('source', img2)
        min_x = min(point1[0],point2[0])
        min_y = min(point1[1],point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] -point2[1])
        if width<2 or height<2:
            return
        cut_img = img2[min_y:min_y+height, min_x:min_x+width]
        cv2.imwrite('result/result'+str(number)+'.jpg',cut_img)
        number+=1
        point.append([min_x,min_y,min_x+width,min_y+height])
        print(point)
        if number>4:
            surf()

def init():
    global source, number, target
    global point
    point = []
    number = 1
    source = cv2.imread('source.jpg')
    #target = cv2.imread('target.jpg')
    cv2.namedWindow('source')
    cv2.setMouseCallback('source', on_mouse)
    cv2.imshow('source', source)
    #cv2.imshow('target', target)
    cv2.waitKey(0)

def showimg(name,img,before):
    cv2.namedWindow(name)
    cv2.imshow(name, img)
    cv2.imshow('source', before)
    cv2.waitKey(0)

def same():
    global point
    img = cv2.cvtColor(cv2.imread("target.jpg"), cv2.COLOR_BGR2GRAY)
    before = cv2.imread("source.jpg")
    pre = []
    s = 0
    for i in range(3):
        cv2.rectangle(before, (point[i][0],point[i][1]), (point[i][2],point[i][3]), (0, 0, 255), 5)
        t1 = time.clock()
        img1 = cv2.cvtColor(cv2.imread("result/result"+str(i+1)+".jpg"),cv2.COLOR_BGR2GRAY)
        img2 = img*1.0
        h,w = img1.shape[:2]
        img1 = img1*1.0-1.0/(w*h)*np.sum(img1)
        sum = np.multiply(img1,img1)
        ans = 0
        result = []
        for x in range(0,img2.shape[0],5):
            if ans>0.8:
                break
            for y in range(0,img2.shape[1],5):
                if x+h>=img2.shape[0] or y+w>=img2.shape[1]:
                    continue
                if ans>0.8:
                    break
                cut_img = img2[x:x+h, y:y+w]
                cut_img = cut_img-1.0/(w*h)*np.sum(cut_img)
                a = np.sum(np.multiply(img1,cut_img))
                b = np.sum(sum)*np.sum(np.multiply(cut_img,cut_img))
                b = np.sqrt(b)
                a = a/b
                if a>ans:
                    ans=a
                    result=[x,y]
        t2 = time.clock()
        print(t2-t1)
        print(ans)
        s+=ans
        x,y = result
        bottom_right = (y + w, x + h)
        cv2.rectangle(img, (y,x), bottom_right, 255, 2)
        pre.append([y, x, y + w, x + h,ans])
    cv2.rectangle(before, (point[3][0], point[3][1]), (point[3][2], point[3][3]), (0, 0, 255), 5)
    rx=0
    ry=0
    for i in range(3):
        dx = point[i][0]-point[3][0]
        dy = point[i][1]-point[3][1]
        rx += (pre[i][0]-dx)*pre[i][4]/s
        ry += (pre[i][1]-dy)*pre[i][4]/s
    rx = (int)(rx)
    ry = (int)(ry)
    cv2.rectangle(img, (rx, ry), (rx+point[3][2]-point[3][0], ry+point[3][3]-point[3][1]), (0, 0, 255), 5)
    showimg('result',img,before)

def main():
    init()

def surf():
    t1 = time.clock()
    global point,source
    img2 = source.copy()
    img2_gray = cv2.imread("0002.jpg")
    img2_gray = test.surf(source,img2_gray)
    pre = []
    for i in range(0,3):
        print(i)
        cv2.rectangle(img2, (point[i][0], point[i][1]), (point[i][2], point[i][3]), (0, 0, 255), 5)
        img1_gray = cv2.imread("result/result"+str(i+1)+".jpg")
        h, w = img1_gray.shape[:2]
        suft = cv2.xfeatures2d.SURF_create()
        kp1, des1 = suft.detectAndCompute(img1_gray,None)
        kp2, des2 = suft.detectAndCompute(img2_gray, None)

        # BFmatcher with default parms
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)

        goodMatch = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                goodMatch.append(m)
        #drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20])
        p1 = [kpp.queryIdx for kpp in goodMatch]
        p2 = [kpp.trainIdx for kpp in goodMatch]

        post1 = np.int32([kp1[pp].pt for pp in p1])
        post2 = np.int32([kp2[pp].pt for pp in p2])
        dx = 0.0
        dy = 0.0
        for (x1, y1), (x2, y2) in zip(post1, post2):
            dx += (x2 - x1)
            dy += (y2 - y1)

        dx = (int)(dx / len(post1))
        dy = (int)(dy / len(post1))

        cv2.rectangle(img2_gray, (dx, dy), (dx + w, dy + h), (0, 0, 255), 5)
        pre.append([dx, dy, dx + w, dy + h])

    cv2.rectangle(img2, (point[3][0], point[3][1]), (point[3][2], point[3][3]), (255, 0, 0), 5)
    rx = 0
    ry = 0
    for i in range(3):
        dx = point[i][0] - point[3][0]
        dy = point[i][1] - point[3][1]
        rx += (pre[i][0] - dx)
        ry += (pre[i][1] - dy)
    rx = (int)(rx/3)
    ry = (int)(ry/3)
    cv2.rectangle(img2_gray, (rx, ry), (rx + point[3][2] - point[3][0], ry + point[3][3] - point[3][1]), (255, 0, 0), 5)
    print(time.clock()-t1)
    showimg('result', img2_gray, img2)

def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2])
    dx = 0.0
    dy = 0.0
    for (x1, y1), (x2, y2) in zip(post1, post2):
        dx +=(x2-x1)
        dy +=(y2-y1)
        print(x2-x1)
        print(y2-y1)
        cv2.line(vis, (x1, y1), (x2+w1, y2), (0, 0, 255))
    dx = (int)(dx/len(post1))+w1
    dy = (int)(dy/len(post1))
    print(dx,dy,len(post1))
    cv2.rectangle(vis, (dx, dy), (dx+w1, dy+h1), (0, 0, 255), 5)
    cv2.namedWindow("match")
    cv2.imshow("match", vis)
    cv2.waitKey()

if __name__ == '__main__':
    main()

