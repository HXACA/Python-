# -- coding: utf-8 --
from PyQt4 import QtGui,QtCore,Qt
import matlab
import absImage
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
def selectAlert(arg):
        arg.dia2 = QtGui.QMessageBox()
        arg.dia2.setStandardButtons(QtGui.QMessageBox.Cancel)
        arg.dia2.setWindowTitle(_translate("MainWindow", "选择算法：", None))
        arg.dia2.setText(_translate("MainWindow", "                                                                   ", None))
        #arg.dia2.setWindowFlags(Qt.Qt.FramelessWindowHint)
        arg.combobox = QtGui.QComboBox(arg.dia2)
        arg.combobox.setGeometry(QtCore.QRect(5, 10, 200, 30))
        #添加算法名

        arg.combobox.addItem("Kmeans")
        arg.combobox.addItem("...")

        arg.pushButton_99 = QtGui.QPushButton(arg.dia2)
        arg.pushButton_99.setGeometry(QtCore.QRect(220, 10, 90, 30))
        arg.pushButton_99.setText(_translate("MainWindow", "确定运行", None))

        arg.dia2.show()

        arg.pushButton_99.clicked.connect(lambda: selRusult(arg))
def selRusult(arg):
        funcname = arg.combobox.currentText()
        print funcname
        if(funcname == "Kmeans"):
            matlab.selectFunc("Kmeans1")
            absImage.nameAbs("Kmeans")
            matlab.run(arg, "clu")
        elif(funcname == "Kmeans1"):
            matlab.selectFunc("Kmeans1")
            absImage.nameAbs("Kmeans1")
            matlab.run(arg, "clu")





        arg.dia2.close()
        #matlab.run(arg, "2")



def alert(arg,type):
    arg.dia = QtGui.QMessageBox()
    arg.dia.setWindowTitle("error")
    if(type==1):
        alert1(arg)
    elif (type == 2):
        alert2(arg)
    elif (type == 3):
        alert3(arg)
    elif (type == 4):
        alert4(arg)
    elif (type == 5):
        alert5(arg)
    elif (type == 6):
        alert6(arg)

    arg.dia.setWindowFlags(Qt.Qt.FramelessWindowHint)
    arg.dia.show()

def alert1(arg):
    arg.dia.setText(_translate("MainWindow","打开失败，可能未选中文件！",None))
    # arg.dia.resize(500,500)
    # arg.dia.setGeometry(QtCore.QRect(800, 400, 200, 1))
    # layout1 = QtGui.QGridLayout()
    # label1 = QtGui.QLabel()
    # label1.setText("12")
    # layout1.addWidget(label1)
    # arg.dia.setLayout(layout1)
def alert2(arg):
    arg.dia.setText(_translate("MainWindow","清空完成",None))

def alert3(arg):
    arg.dia.setText(_translate("MainWindow","完成！",None))

def alert4(arg):
    arg.dia.setText(_translate("MainWindow","当前不为空！",None))

def alert5(arg):
    arg.dia.setText(_translate("MainWindow","当前为空！",None))

def alert6(arg):
    arg.dia.setText(_translate("MainWindow","缺少参数！",None))

