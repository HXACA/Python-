
from PyQt4 import QtGui,QtCore,Qt
import sys
global count1


class MainWnd(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.ui = Abs_3.Ui_MainWindow()
        self.ui.setupUi(self)
        #self.setWindowFlags(Qt.Qt.CustomizeWindowHint)
        self.setStyleSheet("border:1px solid;border-color:rgb(99, 99, 99);border-radius:2px;")



        self.setWindowFlags(Qt.Qt.FramelessWindowHint)
        self.ui.pushButton_2.clicked.connect(lambda :self.min())
        self.ui.pushButton_3.clicked.connect(lambda :self.clo())
        self.SHADOW_WIDTH = 8;
        # QtCore.QObject.connect(self.ui.pushButton,QtCore.SIGNAL("clicked()"),show.openFile(self))
    def clo(self):
        self.close()

    def min(self):
        self.showMinimized()

    def trayClick(self, reason):
        if reason == QtGui.QSystemTrayIcon.DoubleClick:
            self.showNormal()
        else:
            pass
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.dragPosition = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            self.move(event.globalPos() - self.dragPosition)
            event.accept()
    #


if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    my = MainWnd()
    my.show()
    #QtCore.QObject.connect(my.ui.pushButton, QtCore.SIGNAL("clicked()"), showVideo.show())
    #QtCore.QObject.connect(my.ui.pushButton,QtCore.SIGNAL("clicked()"),my.openFile())
    sys.exit(app.exec_())