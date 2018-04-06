# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Main.ui'
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

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        Dialog.resize(900, 600)
        Dialog.setMouseTracking(False)
        self.openFile = QtGui.QPushButton(Dialog)
        self.openFile.setGeometry(QtCore.QRect(10, 10, 100, 50))
        self.openFile.setObjectName(_fromUtf8("openFile"))
        self.videoPlayer = phonon.Phonon.VideoPlayer(Dialog)
        self.videoPlayer.setGeometry(QtCore.QRect(10, 70, 400, 225))
        self.videoPlayer.setObjectName(_fromUtf8("videoPlayer"))
        self.seekSlider = phonon.Phonon.SeekSlider(Dialog)
        self.seekSlider.setGeometry(QtCore.QRect(10, 310, 311, 22))
        self.seekSlider.setObjectName(_fromUtf8("seekSlider"))
        self.start = QtGui.QPushButton(Dialog)
        self.start.setGeometry(QtCore.QRect(160, 10, 100, 50))
        self.start.setObjectName(_fromUtf8("start"))
        self.pause = QtGui.QPushButton(Dialog)
        self.pause.setGeometry(QtCore.QRect(310, 10, 100, 50))
        self.pause.setObjectName(_fromUtf8("pause"))
        self.lcdNumber = QtGui.QLCDNumber(Dialog)
        self.lcdNumber.setGeometry(QtCore.QRect(330, 302, 81, 31))
        self.lcdNumber.setObjectName(_fromUtf8("lcdNumber"))
        self.line = QtGui.QFrame(Dialog)
        self.line.setGeometry(QtCore.QRect(453, 0, 20, 591))
        self.line.setFrameShape(QtGui.QFrame.VLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "视频摘要系统", None))
        self.openFile.setText(_translate("Dialog", "打开文件", None))
        self.start.setText(_translate("Dialog", "播放", None))
        self.pause.setText(_translate("Dialog", "暂停", None))

from PyQt4 import phonon

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    Dialog = QtGui.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

