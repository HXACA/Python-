#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: MainWindows.py 
@time: 2018/03/29 
"""
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import sys
import main

class ui(QMainWindow,main.Ui_Dialog):
    def __init__(self,parent=None):
        super(ui, self).__init__(parent)
        self.setupUi(self)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    my = ui()
    my.show()
    sys.exit(app.exec_())
