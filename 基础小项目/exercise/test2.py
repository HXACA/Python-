#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: test2.py 
@time: 2018/07/28 
"""
from aip import AipOcr
import json

""" 读取图片 """
def get_file_content(filePath):
     with open(filePath, 'rb') as fp:
        return fp.read()

def work(name='result.jpg'):
    APP_ID = '11602238'
    API_KEY = 'WaRSCpHw6aM5rcpX5mHyCt8c'
    SECRET_KEY = '5Xj70sCcgz8VErrmgISAbAcSDA1LUCXY'

    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

    image = get_file_content(name)

    """ 如果有可选参数 """
    options = {}
    options["language_type"] = "CHN_ENG"
    options["detect_direction"] = "true"
    options["detect_language"] = "true"
    options["probability"] = "true"

    """ 带参数调用通用文字识别, 图片参数为本地图片 """
    result = client.basicGeneral(image, options)
    result = result['words_result']
    result = result[0]['words']
    print(result)

if __name__ == '__main__':
    work()

