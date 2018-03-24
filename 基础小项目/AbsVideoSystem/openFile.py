#-- coding:utf-8 --
import cv2.cv as cv
from PyQt4 import QtGui
from PyQt4.QtGui import *
from PyQt4 import QtCore
import numpy as np
import time
from threading import Timer
import sys
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)
def openFile(arg):
        global capture

        filename = QFileDialog.getOpenFileName(None , "Open file dialog", "/", "AVI files(*.avi)")
        # elif fname == 2:
        #     row = arg.listWidget.currentRow()
        #     text = arg.listWidget.takeItem(row).text()
        #     filename = text
        #     arg.listWidget.setCurrentRow(row)


        if filename == "":
            return
        arg.listWidget.addItem(str(filename))
        print filename
        # showVideo(arg)
def showMes(arg,fname):
        filename = fname
        capture = cv.CaptureFromFile(str(filename))
        nbFrames = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT))
        fps = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FPS)
        mes1 = filename
        mes2 = str(nbFrames)
        arg.label_mes1.setText(mes1 + "\n" + mes2)











