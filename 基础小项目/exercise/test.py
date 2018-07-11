#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: test.py 
@time: 2018/07/05 
"""

import cv2
import numpy as np
import math

def surf(img1_gray=None,img2_gray=None):

    # img1_gray = cv2.imread("source2.jpg")
    # img2_gray = cv2.imread("0002.jpg")

    h, w = cv2.imread("source.jpg").shape[:2]
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



    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2])

    y1 = (post1[1][1] - post1[0][1])
    x1 = (post1[1][0] - post1[0][0])

    y2 = (post2[1][1] - post2[0][1])
    x2 =  (post2[1][0] - post2[0][0])

    cos1 = (x1*x2+y1*y2)/(np.sqrt(x1*x1+y1*y1)*np.sqrt(x2*x2+y2*y2))
    cos1 = math.acos(cos1)*180/math.pi
    print(cos1)
    #drawline(img1_gray, kp1, img2_gray, kp2, goodMatch,post1,post2)
    temp = x1*y2-x2*y1
    if temp>=0:
        img2_gray = rotate_bound_white_bg(img2_gray, 360-cos1)
    else:
        img2_gray = rotate_bound_white_bg(img2_gray, cos1)
    kp2, des2 = suft.detectAndCompute(img2_gray, None)
    matches = bf.knnMatch(des1, des2, k=2)

    goodMatch = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            goodMatch.append(m)

    return drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20])

def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]
    h, w = cv2.imread("source.jpg").shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2])

    x1 = int(np.sum(post1[:,0])/len(post1))
    x2 = int(np.sum(post2[:,0])/len(post2))
    y1 = int(np.sum(post1[:, 1]) / len(post1))
    y2 = int(np.sum(post2[:, 1]) / len(post2))

    img = img2_gray[max(y2-y1-50,0):min(y2-y1+h+50,h2),max(x2-x1-50,0):min(x2-x1+w+50,w2)]
    # cv2.imshow("match2", img)
    # cv2.waitKey()
    return img

def drawline(img1_gray, kp1, img2_gray, kp2, goodMatch,post1,post2):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    count = 0
    for (x1, y1), (x2, y2) in zip(post1, post2):
        if count>1:
            break
        print(x1,y1)
        print(x2, y2)
        cv2.line(vis, (x1, y1), (x2+w1, y2), (count*255, 0, 255*(1-count)))
        count+=1
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
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))
    # borderValue 缺省，默认是黑色（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))


if __name__ == '__main__':
    surf()
