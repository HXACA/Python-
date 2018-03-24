import pythoncom
from PyQt4 import QtCore
from win32com.client import Dispatch
import threading,time,thread
import alert
import absImage
import matlab_mesg
import os
import glob
global step
global arg1
step = 0
global func
func = "null"

def selectFunc(Func):#in this give the function name
    global func
    func = Func

    func = str("'"+Func+"'")

def run(arg,funcName):
    global step
    global arg1
    arg1 = arg
    step = 0

    if(funcName=="readvideo"):
        ReadVideo(arg)
    elif(funcName=="clu"):
        Cul(arg)


    

def runCul():
    global func
    print func
    src = r"ResImages"
    exist = 0
    pythoncom.CoInitialize()  # Solve a bug
    args = arg1.textEdit_arg1.toPlainText()
    fname = "keyframe_video2("+args+","+func+")"
    rfname = str(fname)
    print rfname
    matlab_mesg.Mesg99(arg1, rfname)
    # print threading.activeCount()
    # print threading.enumerate()
    h = Dispatch("Matlab.application")
    res = h.execute(rfname)
    arg1.Timer3.stop()
    #arg1.progressBar_2.setProperty("value", 100)
    #matlab_mesg.showMesg(arg1, 6)
    arg1.pushButton_6.setEnabled(True)
    arg1.pushButton_7.setEnabled(True)
    arg1.pushButton_13.setEnabled(True)
    arg1.pushButton_14.setEnabled(True)
    arg1.pushButton_16.setEnabled(True)

    os.chdir(src)
    for file_name in glob.glob("*.jpg"):
        print file_name
        exist = 1
    if (exist == 1):
        matlab_mesg.showMesg(arg1, 8)
    elif (exist == 0):
        matlab_mesg.showMesg(arg1, 9)


def runReadVideo():
    # lock = threading.Lock()
    # lock.acquire()
    pythoncom.CoInitialize()#Solve a bug
    mes1 = arg1.label_mes1.text()
    index1 = str(mes1).find('\n')
    text = mes1[0:index1]
    #analysis the filename
    fname = "readVideo1('"+text+"')"
    rfname = str(fname)
    print rfname
    #matlab_mesg.Mesg99(arg1, rfname)
    #time.sleep(1)
    h = Dispatch("Matlab.application")
    res = h.execute(rfname)

    # print threading.activeCount()
    # print threading.enumerate()

    #time.sleep(1)
    # for i in range(1,100000):
    #     print threading.activeCount()

    arg1.Timer2.stop()
    #arg1.progressBar_2.setProperty("value", 100)
    matlab_mesg.showMesg(arg1, 1)
    arg1.pushButton_6.setEnabled(True)
    arg1.pushButton_7.setEnabled(True)
    arg1.pushButton_16.setEnabled(True)
    # lock.release()



def pro(arg):
    global step
    arg.progressBar_2.setProperty("value", step)
    if(step == 100):
        step = 0
    else:
        step = 100

def ReadVideo(arg):
    arg.Timer2 = QtCore.QTimer()
    arg.Timer2.start(200)
    arg.Timer2.timeout.connect(lambda: pro(arg))
    mes1 = arg1.label_mes1.text()
    absImage.deleteImage(arg,"1")
    arg1.pushButton_6.setEnabled(False)
    arg1.pushButton_7.setEnabled(False)
    arg1.pushButton_16.setEnabled(False)
    index1 = str(mes1).find('\n')
    text = mes1[0:index1]
    if(text==""):
        alert.alert(arg,1)
        arg1.Timer2.stop()
        arg1.pushButton_6.setEnabled(True)
        arg1.pushButton_7.setEnabled(True)
        arg1.pushButton_16.setEnabled(True)
    else:
        thread1 = threading.Thread(target=runReadVideo, args=())
        thread1.setDaemon(True)
        thread1.start()

def Cul(arg):
    arg.Timer3 = QtCore.QTimer()
    arg.Timer3.start(200)
    arg.Timer3.timeout.connect(lambda: pro(arg))
    absImage.deleteAbsImage(arg)
    arg.pushButton_6.setEnabled(False)
    arg.pushButton_7.setEnabled(False)
    arg1.pushButton_13.setEnabled(False)
    arg1.pushButton_14.setEnabled(False)
    arg1.pushButton_16.setEnabled(False)
    args = arg1.textEdit_arg1.toPlainText()
    if(args==""):
        alert.alert(arg,6)
        arg1.Timer3.stop()
        arg1.pushButton_6.setEnabled(True)
        arg1.pushButton_7.setEnabled(True)
        arg1.pushButton_13.setEnabled(True)
        arg1.pushButton_14.setEnabled(True)
        arg1.pushButton_16.setEnabled(True)
    else:
        thread2 = threading.Thread(target=runCul, args=())
        thread2.setDaemon(True)
        thread2.start()

