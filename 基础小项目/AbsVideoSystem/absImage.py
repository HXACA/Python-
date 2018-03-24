import cv2
from PyQt4 import  QtCore, QtGui
from PyQt4.QtGui import QPixmap
from PyQt4.QtGui import QImage
import glob
import os
import Image
import alert
import matlab_mesg
global src1
global absName
path1=os.path.abspath('.')
absName = "undefined"
src = path1 + r"/images"
src1 =path1+ r"/ResImages"


def showImage1(arg):
    global src1
    os.chdir(src1)
    listWidget_showImage1 = QtGui.QListWidget()
    for file_name in glob.glob("*.jpg"):
            print file_name
            filenames = src1+"/"+file_name
            m = QImage(filenames)
            m = QPixmap.fromImage(m)
            item = QtGui.QListWidgetItem()
            icon = QtGui.QIcon(m)
            item.setIcon(icon)

            listWidget_showImage1.setIconSize(QtCore.QSize(100,100))
            listWidget_showImage1.setViewMode(QtGui.QListView.IconMode)
            listWidget_showImage1.addItem(item)

    arg.tabWidget_2.addTab(listWidget_showImage1, absName)
    arg.tabWidget_2.setCurrentWidget(listWidget_showImage1);
    arg.pushButton_13.setEnabled(False)
    matlab_mesg.showMesg(arg,5)

def nameAbs(absname):
    global absName
    absName = absname

def checkImage(arg):
    exist = 0
    os.chdir(src)
    for file_name in glob.glob("*.jpg"):
        print file_name
        exist = 1
    if(exist==1):
        alert.alert(arg,4)
        matlab_mesg.showMesg(arg,2)
    elif(exist==0):
        alert.alert(arg,5)
        matlab_mesg.showMesg(arg,3)


def deleteImage(arg,f):
    os.chdir(src)
    for file_name in glob.glob("*.jpg"):
            #arg.listWidget_showImage.takeItem(0)# delete a item,the next item always 0
            os.remove(src+"/"+file_name)
    if (f == "1"):
        1
    else:
        alert.alert(arg, 2)
    matlab_mesg.showMesg(arg,7)


def clearAbsImage(arg):
    arg.pushButton_13.setEnabled(True)
    arg.tabWidget_2.removeTab(arg.tabWidget_2.indexOf(arg.tabWidget_2.currentWidget()))  # delete a item,the next item always 0
    # for file_name in glob.glob("*.jpg"):
    #         os.remove(src1+"/"+file_name)
def deleteAbsImage(arg):
    os.chdir(src1)
    for file_name in glob.glob("*.jpg"):
        os.remove(src1+"/"+file_name)
    matlab_mesg.showMesg(arg,4)
    arg.pushButton_13.setEnabled(True)




