import openFile,Aplayer,absImage,alert
import matlab

def buttonConnect(arg):
    arg.pushButton.clicked.connect(lambda: openFile.openFile(arg))
   # arg.pushButton.clicked.connect(lambda: pressStyle.button(arg))
    arg.pushButton_9.clicked.connect(lambda: Aplayer.readFilenameAndPlay(arg))
    arg.pushButton_4.clicked.connect(lambda :Aplayer.pause(arg))
    arg.pushButton_5.clicked.connect(lambda: Aplayer.conti(arg))
    arg.pushButton_6.clicked.connect(lambda: alert.selectAlert(arg))
    arg.pushButton_7.clicked.connect(lambda: matlab.run(arg, "readvideo"))
    arg.pushButton_8.clicked.connect(lambda: absImage.checkImage(arg))
    arg.pushButton_13.clicked.connect(lambda :absImage.showImage1(arg))
    arg.pushButton_16.clicked.connect(lambda: absImage.deleteImage(arg,"2"))
    arg.pushButton_14.clicked.connect(lambda: absImage.deleteAbsImage(arg))
    arg.pushButton_10.clicked.connect(lambda: absImage.clearAbsImage(arg))




def listWidgetConnect(arg):
    arg.listWidget.itemDoubleClicked.connect(lambda: openFile.showMes(arg,arg.listWidget.item(arg.listWidget.currentRow()).text()))


