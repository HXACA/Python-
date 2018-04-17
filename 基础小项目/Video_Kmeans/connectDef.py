#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: connectDef.py 
@time: 2018/04/12 
"""
import openFile

videostr = []

def buttonConnect(arg):
    arg.file.clicked.connect(lambda :openFile.openFile(arg))
    arg.work.clicked.connect(lambda :openFile.abstract(arg))
