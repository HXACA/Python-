#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: test.py 
@time: 2018/07/05 
"""

import cv2
import numpy as np
import math

def surf(img1_gray,img2_gray):

    # img1_gray = cv2.imread("source.jpg")
    # img2_gray = cv2.imread("0002.jpg")

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

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2])

    tan1 = (post1[1][1] - post1[0][1]) / (post1[1][0] - post1[0][0])
    tan1 = math.atan(tan1) * 180 / math.pi
    print(tan1)

    tan2 = (post2[1][1] - post2[0][1]) / (post2[1][0] - post2[0][0])
    tan2 = math.atan(tan2) * 180 / math.pi
    print(tan2)
    img2_gray = rotate_bound_white_bg(img2_gray, tan1 - tan2)
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

    img = img2_gray[max(y2-y1-50,0):min(y2-y1+h1+50,h2),max(x2-x1-50,0):min(x2-x1+w1+50,w2)]
    return img


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
