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
# def button(arg):
#     arg.pushButton.pressed.setStyleSheet(_fromUtf8("border:3px solid;\n"
#                                             "color: rgb(117, 117, 117);\n"
#                                             "background-color: rgb(133, 133, 133);\n"
#                                             "border-radius:11px;\n"
#                                             "border-color: rgb(19, 11, 255);"))