# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'init.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

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

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(960, 640)
        MainWindow.setMinimumSize(QtCore.QSize(960, 640))
        MainWindow.setMaximumSize(QtCore.QSize(960, 640))
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.videowidget = QtGui.QWidget(self.centralwidget)
        self.videowidget.setGeometry(QtCore.QRect(0, 0, 400, 300))
        self.videowidget.setStyleSheet(_fromUtf8(""))
        self.videowidget.setObjectName(_fromUtf8("videowidget"))
        self.videoPlayer = phonon.Phonon.VideoPlayer(self.videowidget)
        self.videoPlayer.setGeometry(QtCore.QRect(0, 0, 400, 300))
        self.videoPlayer.setObjectName(_fromUtf8("videoPlayer"))
        self.widget_2 = QtGui.QWidget(self.centralwidget)
        self.widget_2.setGeometry(QtCore.QRect(480, 0, 500, 380))
        self.widget_2.setObjectName(_fromUtf8("widget_2"))
        self.result_1 = phonon.Phonon.VideoPlayer(self.widget_2)
        self.result_1.setGeometry(QtCore.QRect(0, 0, 160, 120))
        self.result_1.setObjectName(_fromUtf8("result_1"))
        self.result_2 = phonon.Phonon.VideoPlayer(self.widget_2)
        self.result_2.setGeometry(QtCore.QRect(170, 0, 160, 120))
        self.result_2.setObjectName(_fromUtf8("result_2"))
        self.result_3 = phonon.Phonon.VideoPlayer(self.widget_2)
        self.result_3.setGeometry(QtCore.QRect(340, 0, 160, 120))
        self.result_3.setObjectName(_fromUtf8("result_3"))
        self.result_4 = phonon.Phonon.VideoPlayer(self.widget_2)
        self.result_4.setGeometry(QtCore.QRect(0, 130, 160, 120))
        self.result_4.setObjectName(_fromUtf8("result_4"))
        self.result_5 = phonon.Phonon.VideoPlayer(self.widget_2)
        self.result_5.setGeometry(QtCore.QRect(170, 130, 160, 120))
        self.result_5.setObjectName(_fromUtf8("result_5"))
        self.result_6 = phonon.Phonon.VideoPlayer(self.widget_2)
        self.result_6.setGeometry(QtCore.QRect(340, 130, 160, 120))
        self.result_6.setObjectName(_fromUtf8("result_6"))
        self.result_7 = phonon.Phonon.VideoPlayer(self.widget_2)
        self.result_7.setGeometry(QtCore.QRect(0, 260, 160, 120))
        self.result_7.setObjectName(_fromUtf8("result_7"))
        self.result_8 = phonon.Phonon.VideoPlayer(self.widget_2)
        self.result_8.setGeometry(QtCore.QRect(170, 260, 160, 120))
        self.result_8.setObjectName(_fromUtf8("result_8"))
        self.result_9 = phonon.Phonon.VideoPlayer(self.widget_2)
        self.result_9.setGeometry(QtCore.QRect(340, 260, 160, 120))
        self.result_9.setObjectName(_fromUtf8("result_9"))
        self.play = QtGui.QPushButton(self.centralwidget)
        self.play.setGeometry(QtCore.QRect(10, 320, 80, 40))
        self.play.setObjectName(_fromUtf8("play"))
        self.pause = QtGui.QPushButton(self.centralwidget)
        self.pause.setGeometry(QtCore.QRect(120, 320, 80, 40))
        self.pause.setObjectName(_fromUtf8("pause"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(230, 320, 80, 40))
        self.label.setObjectName(_fromUtf8("label"))
        self.information = QtGui.QLabel(self.centralwidget)
        self.information.setGeometry(QtCore.QRect(310, 320, 80, 40))
        self.information.setText(_fromUtf8(""))
        self.information.setObjectName(_fromUtf8("information"))
        self.file = QtGui.QPushButton(self.centralwidget)
        self.file.setGeometry(QtCore.QRect(10, 400, 80, 40))
        self.file.setStyleSheet(_fromUtf8(""))
        self.file.setObjectName(_fromUtf8("file"))
        self.work = QtGui.QPushButton(self.centralwidget)
        self.work.setGeometry(QtCore.QRect(120, 400, 80, 40))
        self.work.setStyleSheet(_fromUtf8(""))
        self.work.setObjectName(_fromUtf8("work"))
        self.num = QtGui.QTextEdit(self.centralwidget)
        self.num.setGeometry(QtCore.QRect(310, 400, 40, 40))
        self.num.setObjectName(_fromUtf8("num"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(230, 400, 80, 40))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.play.setText(_translate("MainWindow", "播放", None))
        self.pause.setText(_translate("MainWindow", "暂停", None))
        self.label.setText(_translate("MainWindow", "文件信息：", None))
        self.file.setText(_translate("MainWindow", "打开文件", None))
        self.work.setText(_translate("MainWindow", "提取摘要", None))
        self.label_2.setText(_translate("MainWindow", "摘要数量：", None))

from PyQt4 import phonon
