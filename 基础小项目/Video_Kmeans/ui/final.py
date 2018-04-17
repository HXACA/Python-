# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'final.ui'
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

class Ui_videofrom(object):
    def setupUi(self, videofrom):
        videofrom.setObjectName(_fromUtf8("videofrom"))
        videofrom.resize(1000, 610)
        self.verticalLayoutWidget_2 = QtGui.QWidget(videofrom)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(0, 10, 1001, 551))
        self.verticalLayoutWidget_2.setObjectName(_fromUtf8("verticalLayoutWidget_2"))
        self.verticalLayout_main = QtGui.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_main.setObjectName(_fromUtf8("verticalLayout_main"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.verticalLayout_player = QtGui.QVBoxLayout()
        self.verticalLayout_player.setSpacing(6)
        self.verticalLayout_player.setObjectName(_fromUtf8("verticalLayout_player"))
        self.verticalLayout.addLayout(self.verticalLayout_player)
        self.seekSlider = phonon.Phonon.SeekSlider(self.verticalLayoutWidget_2)
        self.seekSlider.setObjectName(_fromUtf8("seekSlider"))
        self.verticalLayout.addWidget(self.seekSlider)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.BtnOpen = QtGui.QPushButton(self.verticalLayoutWidget_2)
        self.BtnOpen.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.BtnOpen.setObjectName(_fromUtf8("BtnOpen"))
        self.horizontalLayout.addWidget(self.BtnOpen)
        self.line = QtGui.QFrame(self.verticalLayoutWidget_2)
        self.line.setFrameShape(QtGui.QFrame.VLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.horizontalLayout.addWidget(self.line)
        self.horizontalLayout_btn = QtGui.QHBoxLayout()
        self.horizontalLayout_btn.setObjectName(_fromUtf8("horizontalLayout_btn"))
        self.horizontalLayout.addLayout(self.horizontalLayout_btn)
        self.line_2 = QtGui.QFrame(self.verticalLayoutWidget_2)
        self.line_2.setFrameShape(QtGui.QFrame.VLine)
        self.line_2.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_2.setObjectName(_fromUtf8("line_2"))
        self.horizontalLayout.addWidget(self.line_2)
        self.volumeSlider = phonon.Phonon.VolumeSlider(self.verticalLayoutWidget_2)
        self.volumeSlider.setObjectName(_fromUtf8("volumeSlider"))
        self.horizontalLayout.addWidget(self.volumeSlider)
        self.line_3 = QtGui.QFrame(self.verticalLayoutWidget_2)
        self.line_3.setFrameShape(QtGui.QFrame.VLine)
        self.line_3.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_3.setObjectName(_fromUtf8("line_3"))
        self.horizontalLayout.addWidget(self.line_3)
        self.lcdNumber = QtGui.QLCDNumber(self.verticalLayoutWidget_2)
        self.lcdNumber.setObjectName(_fromUtf8("lcdNumber"))
        self.horizontalLayout.addWidget(self.lcdNumber)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(2, 2)
        self.horizontalLayout.setStretch(4, 5)
        self.horizontalLayout.setStretch(6, 2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout.setStretch(0, 8)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)
        self.verticalLayout_main.addLayout(self.verticalLayout)
        self.verticalLayout_main.setStretch(0, 2)
        self.pushButton = QtGui.QPushButton(videofrom)
        self.pushButton.setGeometry(QtCore.QRect(180, 570, 60, 30))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.label = QtGui.QLabel(videofrom)
        self.label.setGeometry(QtCore.QRect(10, 570, 60, 30))
        self.label.setObjectName(_fromUtf8("label"))
        self.textBrowser = QtGui.QTextBrowser(videofrom)
        self.textBrowser.setGeometry(QtCore.QRect(70, 570, 60, 30))
        self.textBrowser.setObjectName(_fromUtf8("textBrowser"))
        self.pushButton_2 = QtGui.QPushButton(videofrom)
        self.pushButton_2.setGeometry(QtCore.QRect(350, 570, 100, 30))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.label_2 = QtGui.QLabel(videofrom)
        self.label_2.setGeometry(QtCore.QRect(850, 570, 150, 30))
        self.label_2.setObjectName(_fromUtf8("label_2"))

        self.retranslateUi(videofrom)
        QtCore.QMetaObject.connectSlotsByName(videofrom)

    def retranslateUi(self, videofrom):
        videofrom.setWindowTitle(_translate("videofrom", "Form", None))
        self.BtnOpen.setToolTip(_translate("videofrom", "<html><head/><body><p>选择文件，右键选择音频or 视频</p></body></html>", None))
        self.BtnOpen.setText(_translate("videofrom", "选择文件", None))
        self.pushButton.setText(_translate("videofrom", "提取摘要", None))
        self.label.setText(_translate("videofrom", "摘要数量：", None))
        self.pushButton_2.setText(_translate("videofrom", "打开摘要目录", None))
        self.label_2.setText(_translate("videofrom", "视频摘要系统  HXACA", None))

from PyQt4 import phonon
