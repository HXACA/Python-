import cv2.cv as cv
from PyQt4.QtGui import *
from PyQt4 import QtCore
import numpy as np
import threading
import openFile
import alert
global filename
global capture
global nbFrames
global wait
global prog
# global arg1,fname1
prog = 0
def showVideo(arg,fname):
        global capture
        global nbFrames
        global wait
        global prog
        prog = 0
        try:
            filename = fname
            openFile.showMes(arg,filename)
            capture = cv.CaptureFromFile(str(filename))
            nbFrames = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT))
            fps = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FPS)
            wait = int(1 / fps * 1000 / 1)
            duration = (nbFrames * fps) / 1000
            arg.Timer1 = QtCore.QTimer()
            arg.Timer1.timeout.connect(lambda: review(arg))
            arg.Timer1.start(wait)
            review(arg)
        except:
            alert.alert(arg,1)
def readFilenameAndPlay(arg):
        global filename
        mes1 = arg.label_mes1.text()
        index1 = str(mes1).find('\n')
        text = mes1[0:index1]
        filename = text
        # setArg(arg,filename)
        showVideo(arg,filename)
        # showVideo(arg, "D:/v92.avi")

# def setArg(arg,filename):
#     global arg1
#     global filename1
#     arg1 = arg
#     filename1 = filename

def pause(arg):
    arg.Timer1.start(10000000000000)
def conti(arg):
        arg.Timer1.start(wait)
def review(arg):
        global prog
        frame = cv.QueryFrame(capture)
        thumb = resize(frame)
        nimage = QImage

        if thumb:
            image = QImage(thumb.tostring(), thumb.width, thumb.height, QImage.Format_RGB888).rgbSwapped()
            nimage = image.scaled(321, 251)
        # painter = QPainter(arg.label)
        # painter.drawImage(QtCore.QPoint(5, 5),nimage)
        pixmap = QPixmap.fromImage(nimage)
        arg.label.setPixmap(pixmap)
        arg.progressBar.setProperty("value", (prog+1)*100/nbFrames)
        prog=prog+1

        if (prog>=nbFrames-1):
                arg.Timer1.stop()
                arg.label.clear()
                arg.label.setText("                  END")
                arg.progressBar.setProperty("value", 100)

def resize(frame):
        thumb = cv.CreateImage((frame.width,frame.height), 8, 3)
        cv.Resize(frame, thumb)
        return thumb
def resizeImg(src, dstsize):
        if src.ndim == 3:
            dstsize.append(3)
        dst = np.array(np.zeros(dstsize), src.dtype)
        factory = float(np.size(src, 0)) / dstsize[0]
        factorx = float(np.size(src, 1)) / dstsize[1]
        print 'factory', factory, 'factorx', factorx
        srcheight = np.size(src, 0)
        srcwidth = np.size(src, 1)
        print 'srcwidth', srcwidth, 'srcheight', srcheight
        for i in range(dstsize[0]):
            for j in range(dstsize[1]):
                y = float(i) * factory
                x = float(j) * factorx
                if y + 1 > srcheight:
                    y -= 1
                if x + 1 > srcwidth:
                    x -= 1
                cy = np.ceil(y)
                fy = cy - 1
                cx = np.ceil(x)
                fx = cx - 1
                w1 = (cx - x) * (cy - y)
                w2 = (x - fx) * (cy - y)
                w3 = (cx - x) * (y - fy)
                w4 = (x - fx) * (y - fy)
                if (x - np.floor(x) > 1e-6 or y - np.floor(y) > 1e-6):
                    t = src[fy, fx] * w1 + src[fy, cx] * w2 + src[cy, fx] * w3 + src[cy, cx] * w4
                    t = np.ubyte(np.floor(t))
                    dst[i, j] = t
                else:
                    dst[i, j] = (src[y, x])
        return dst















