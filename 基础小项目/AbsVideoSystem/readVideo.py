# -*- coding: utf-8 -*-
import cv2

# 获得视频的格式
videoCapture = cv2.VideoCapture('D:/testVideo/v04.avi')

# 获得码率及尺寸
fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
nFrame = int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
# 读帧
success, frame = videoCapture.read()
count = 1
for k in range(0,nFrame):
    if k % 50 == 0 :
        filename = str(count)
        print filename
        cv2.imwrite("D:/images/"+filename +".jpg", frame)
        count +=1
    success, frame = videoCapture.read()  # 获取下一帧

