#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: test.py 
@time: 2018/07/05 
"""
import sys
sys.setrecursionlimit(10000)  # set the maximum depth as 10000
import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import os


def surf(img1_gray=None,img2_gray=None):

    #img1_gray = cv2.cvtColor(cv2.imread("source4.jpg"),cv2.COLOR_RGB2GRAY)

    img2 = cv2.imread("test.jpg")
    img2_gray = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)

    t1 = time.clock()
    suft = cv2.xfeatures2d.SURF_create()
    kp1, des1 = suft.detectAndCompute(img1_gray,None)
    kp2, des2 = suft.detectAndCompute(img2_gray, None)
    # BFmatcher with default parms
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    print("第一次检测：" + str(time.clock() - t1))

    t1 = time.clock()
    a = -1
    b = -1
    minn = 0x3f3f3f3f
    goodMatch = []
    count = 0
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            goodMatch.append(m)
            if m.distance<minn:
                b = a
                a = count
                minn = m.distance
            count+=1


    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2])

    y1 = (post1[a][1] - post1[b][1])
    x1 = (post1[a][0] - post1[b][0])

    y2 = (post2[a][1] - post2[b][1])
    x2 =  (post2[a][0] - post2[b][0])
    cos1 = min((x1*x2+y1*y2)/(np.sqrt(x1*x1+y1*y1)*np.sqrt(x2*x2+y2*y2)),1.0)
    cos1 = max(-1.0,cos1)
    cos1 = math.acos(cos1)*180.0/math.pi
    #drawline(img1_gray, kp1, img2_gray, kp2, goodMatch,post1,post2)
    temp = x1*y2-x2*y1
    if temp>=0:
        img2_gray,M= rotate_bound_white_bg(img2_gray, 360-cos1)
    else:
        img2_gray,M= rotate_bound_white_bg(img2_gray, cos1)


    for i in range(len(post2)):
        post2[i, :] = np.dot(M[0:2, 0:2], post2[i, :]) + M[:, 2]
    print("旋转：" + str(time.clock() - t1))
    t1 = time.clock()
    img2_gray = drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20],post1,post2)
    print("精确裁切：" + str(time.clock() - t1))
    # cv2.imshow("test",img2_gray)
    # cv2.waitKey()
    return img2_gray

def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch,post1,post2):
    h, w = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]
    #
    # p1 = [kpp.queryIdx for kpp in goodMatch]
    # p2 = [kpp.trainIdx for kpp in goodMatch]
    #
    # post1 = np.int32([kp1[pp].pt for pp in p1])
    # post2 = np.int32([kp2[pp].pt for pp in p2])
    # print(post2)
    g1 = np.zeros((len(post1), len(post1)))
    g2 = np.zeros((len(post1), len(post1)))
    for i in range(len(post1)):
        for j in range(i, len(post1)):
            if i == j:
                continue
            x1, y1 = post1[i, :]
            x2, y2 = post1[j, :]
            g1[i, j] = g1[j, i] = np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))

    for i in range(len(post1)):
        for j in range(i, len(post1)):
            if i == j:
                g2[i, j] = 0x3f3f3f3f
                continue
            x1, y1 = post2[i, :]
            x2, y2 = post2[j, :]
            g2[i, j] = g2[j, i] = np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))

    g = np.abs(g2 - g1)
    #print(np.argwhere(g == np.min(g)))
    a, b = np.argwhere(g == np.min(g))[0]
    x1,y1 = post1[a,:]
    x2,y2 = post2[a,:]
    img = img2_gray[max(y2-y1-25,0):min(y2-y1+h+25,h2),max(x2-x1-25,0):min(x2-x1+w+25,w2)]
    # cv2.imshow("match2", img)
    # cv2.waitKey()
    return img

def drawline(img1_gray, kp1, img2_gray, kp2, goodMatch,post1,post2):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    count = 0
    for (x1, y1), (x2, y2) in zip(post1, post2):
        print(x1,y1)
        print(x2, y2)
        cv2.line(vis, (x1, y1), (x2+w1, y2), (count*255, 0, 255*(1-count)))
    cv2.namedWindow("match")
    cv2.imshow("match", vis)
    cv2.waitKey()

def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    print(M.shape)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255)),M
    # borderValue 缺省，默认是黑色（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))

def sum(img=None):
    # img = cv2.imread("EMS/0002.jpg")
    # img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    h,w = img.shape
    l = 0
    r = w-1
    u = 0
    p = h-1
    print(h,w)
    # img[0:600,:]=0
    # cv2.imwrite("test2.jpg", img)
    sum = np.sum(img,axis=0)
    for i in range(w):
        if sum[i]>330000 :
            l=i
            break
    for i in range(w-1,l+1,-1):
        if sum[i]>330000:
            r=i
            break
    print("------")
    sum = np.sum(img, axis=1)
    for i in range(h):
        if (i>200 and sum[i] > 330000):
            u = i
            print(sum[i])
            break
    for i in range(h-1, u + 1, -1):
        if (i<h-200 and sum[i] > 330000):
            p = i
            break
    print(u,p)
    img = img[u:p,l:r]
    cv2.imwrite("test.jpg",img)
    # cv2.namedWindow("match")
    # cv2.imshow("match",img)
    # cv2.waitKey()

def test(gray=None):
    # img = cv2.imread("EMS/0016.jpg")
    # cv2.namedWindow("source", cv2.WINDOW_NORMAL)
    # cv2.imshow("source", img)
    # gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
    # cv2.imshow("gray", gray)
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
    # cv2.imshow("binary", thresh1)
    cv2.imwrite("binary.jpg",thresh1)
    h,w = gray.shape[:2]
    sum = np.sum(thresh1,axis=0)

    l = []
    sum = sum>51000
    l = np.argwhere(sum==True)

    temp = np.zeros((h,len(l)))
    for i in range(len(l)):
        temp[:, i] = gray[:, l[i,0]]

    l = []
    sum = np.sum(thresh1, axis=1)
    sum = sum > 51000
    l = np.argwhere(sum == True)
    new = np.zeros((len(l),temp.shape[1]))
    for i in range(len(l)):
        new[i] = temp[l[i]]
    cv2.imwrite("test.jpg",new)
    # cv2.namedWindow("new", cv2.WINDOW_NORMAL)
    # cv2.imshow("new", cv2.imread("test.jpg"))
    # cv2.waitKey()

def test2(gray=None):
    imgDir = 'EMS/'
    imgs = os.listdir(imgDir)
    for Img in imgs:
        t1 = time.clock()
        imgPath = os.path.join(imgDir,Img)
        img = cv2.imread(imgPath)
        h,w = img.shape[:2]
        img = img[600:h-600,:]
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite("binary/binary+"+str(Img)+".jpg",thresh1)
        h,w = gray.shape[:2]
        sum = np.sum(thresh1,axis=0)
        l = []
        for i in range(w):
            if sum[i]>25500 :
                l.append(i)
        temp = np.zeros((h,len(l)))
        for i in range(len(l)):
            temp[:, i] = gray[:, l[i]]

        l = []
        sum = np.sum(thresh1, axis=1)
        for i in range(h):
            if sum[i]>25500:
                l.append(i)
        new = np.zeros((len(l),temp.shape[1]))
        for i in range(len(l)):
            new[i] = temp[l[i]]
        cv2.imwrite("test/test"+str(Img)+".jpg",new)
        print(time.clock()-t1)

def getdis(x1,y1,x2,y2):
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))

def test3():
    imgDir = 'EMS/'
    imgs = os.listdir(imgDir)
    img = cv2.imread('source/source.jpg')
    for Img in imgs:
        t1 = time.clock()
        imgPath = os.path.join(imgDir, Img)
        img2_gray = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_RGB2GRAY)
        # 适当裁切
        test(img2_gray)
        # 旋转至标准图
        img2_gray = surf(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        print(imgPath)
        cv2.imwrite("change/"+str(Img),img2_gray)
        print(time.clock() - t1)


if __name__ == '__main__':
    test3()
