#-- coding:utf-8 --
import cv2.cv as cv
from PyQt4 import QtGui
from PyQt4.QtGui import *
from PyQt4 import QtCore
from PyQt4.phonon import *
import threading
import numpy as np
import time
from threading import Timer
import Abstract
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

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
        filename = QFileDialog.getOpenFileName(None , "Open file dialog", u"E:\Python的杂七杂八\基础小项目\Video_Kmeans", "AVI files(*.avi)")
        if filename == "":
            return
        arg.videoPlayer.load(Phonon.MediaSource(filename))
        arg.videoPlayer.play()
        return filename


def abstract(arg):
    # videostr = openFile(arg)
    # thread1 = threading.Thread(target=Abstract.work(str(videostr)))
    # thread1.setDaemon(True)
    # thread1.start()
    # arg.videoPlayer.load(Phonon.MediaSource(videostr))
    # arg.videoPlayer.play()
    filename = u'E:\Python的杂七杂八\基础小项目\Video_Kmeans\\abstractVideo\Video\\abstractVideo'
    while arg.result_1.state() != Phonon.PlayingState:
        arg.result_1.load(Phonon.MediaSource(filename+str(1)+'.avi'))
        arg.result_1.play()
    arg.result_2.load(Phonon.MediaSource(filename + str(2) + '.avi'))
    arg.result_2.play()
    arg.result_3.load(Phonon.MediaSource(filename + str(3) + '.avi'))
    arg.result_3.play()
