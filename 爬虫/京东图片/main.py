#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: main.py 
@time: 2018/08/07 
"""
import re
import urllib.request

def craw(url,page):
    html1 = urllib.request.urlopen(url).read()
    html1 = str(html1)
    #正则选择内容部分
    pat1 = '<div id="plist" .+? <div class="page clearfix">'
    result1 = re.compile(pat1).findall(html1)
    result1 = result1[0]
    #正则提取图片链接
    pat2 = '<img width="220" height="220" data-img="1" src="//(.+?\.(jpg|png))">'
    imagelist = re.compile(pat2).findall(result1)
    x = 1
    for imageurl in imagelist:
        imagename = "img/"+str(page)+str(x)+"."+imageurl[1]
        print(imageurl)
        imageurl = "http://"+imageurl[0]
        try:
            urllib.request.urlretrieve(imageurl,filename=imagename)
        except urllib.error.URLError as e:
            if hasattr(e,"code"):
                x+=1
            if hasattr(e,"reason"):
                x+=1
            print(e,imageurl)
        x+=1

if __name__ == '__main__':
    # page控制页数
    for i in range(1,79):
        url = "https://list.jd.com/list.html?cat=9987,653,655&page="+str(i)
        craw(url,i)