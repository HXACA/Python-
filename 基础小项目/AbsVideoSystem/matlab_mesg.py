# -- coding: utf-8 --
from PyQt4 import QtGui,QtCore,Qt
import  time
global count
count = 0
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



def showMesg(arg,c):
    global count
    count +=1
    cutime = time.strftime('%Y-%m-%d(%H:%M:%S)',time.localtime(time.time()))
    cutime_text =str(cutime)
    if (c == 1):
        mesg1(arg,cutime_text)
    elif (c == 2):
        mesg2(arg,cutime_text)
    elif (c == 3):
        mesg3(arg,cutime_text)
    elif (c == 4):
        mesg4(arg,cutime_text)
    elif (c == 5):
        mesg5(arg,cutime_text)
    elif (c == 6):
        mesg6(arg,cutime_text)
    elif (c == 7):
        mesg7(arg,cutime_text)
    elif (c == 8):
        mesg8(arg,cutime_text)
    elif (c == 9):
        mesg9(arg,cutime_text)

    #arg.listWidget_2.setCurrentRow(arg.listWidget_2.count()-1)
    #QtGui.QAbstractItemView.scrollToBottom(arg.listWidget_2)
def mesg1(arg,cutime_text):
    arg.listWidget_2.insertItem(0,_translate("", "简单提取结束！" + cutime_text + "", None))

def mesg2(arg,cutime_text):
    arg.listWidget_2.insertItem(0,_translate("","初步摘要已生成，可进行精确提取！"+cutime_text+"",None))

def mesg3(arg,cutime_text):
    arg.listWidget_2.insertItem(0,_translate("","初步摘要未生成！"+cutime_text+"",None))

def mesg4(arg,cutime_text):
    arg.listWidget_2.insertItem(0,_translate("","摘要删除！"+cutime_text+"",None))

def mesg5(arg,cutime_text):
    arg.listWidget_2.insertItem(0,_translate("","摘要显示！"+cutime_text+"",None))

def mesg6(arg,cutime_text):
    arg.listWidget_2.insertItem(0,_translate("","精确提取结束！"+cutime_text+"",None))

def mesg7(arg,cutime_text):
    arg.listWidget_2.insertItem(0,_translate("","清空完成！"+cutime_text+"",None))

def mesg8(arg,cutime_text):
    arg.listWidget_2.insertItem(0,_translate("","摘要生成！"+cutime_text+"",None))

def mesg9(arg,cutime_text):
    arg.listWidget_2.insertItem(0,_translate("","执行过程可能出现问题！"+cutime_text+"",None))

def Mesg99(arg,fname):
    cutime = time.strftime('%Y-%m-%d(%H:%M:%S)',time.localtime(time.time()))
    cutime_text =str(cutime)
    arg.listWidget_2.insertItem(0,_translate("","开始执行"+fname + cutime_text+"",None))
    #arg.listWidget_2.setCurrentRow(arg.listWidget_2.count()-1)
    #QtGui.QAbstractItemView.scrollToBottom(arg.listWidget_2)
