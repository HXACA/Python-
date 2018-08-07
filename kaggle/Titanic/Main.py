#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: Main.py 
@time: 2018/07/19 
"""
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

def data():
    data_train = pd.read_csv("all/train.csv")

    return data_train

if __name__ == '__main__':
    data()