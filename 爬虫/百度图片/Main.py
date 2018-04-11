#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: Main.py 
@time: 2018/04/08 
"""
import os
import re
import sys
import urllib
import requests
import threading
threadLock = threading.Lock()

class myThread(threading.Thread):
    def __init__(self,active,img=[],path=""):
        threading.Thread.__init__(self)
        self.active = active
        self.img = img
        self.path = path
    def run(self):
        if self.active == "del":
            print("开启删除图片线程")
            threadLock.acquire()
            delImg(self.path)
            threadLock.release()
        elif self.active == "save":
            print("开启保存图片线程")
            threadLock.acquire()
            down_pic(self.img,self.path)
            threadLock.release()


def delImg(imgDir):
    imgs = os.listdir(imgDir)
    for img in imgs:
        os.remove(imgDir+'/'+img)

def get_onepage_urls(onepageurl):
    if not onepageurl:
        print('最后一页，结束')
        return []
    try:
        type = sys.getfilesystemencoding()
        html = requests.get(onepageurl).content.decode('utf-8')
    except Exception as e:
        print(e)
        pic_urls = []
        fanye_url = ''
        return pic_urls,fanye_url
    pic_urls = re.findall('"objURL":"(.*?)",', html, re.S)
    fanye_urls = re.findall(re.compile('<a href="(.*)" class="n">下一页</a>'), html, flags=0)
    fanye_url = 'http://image.baidu.com' + fanye_urls[0] if fanye_urls else ''
    return pic_urls,fanye_url

def down_pic(pic_urls,path):
    print("一共有%d张图片准备保存" %(len(pic_urls)))
    for i,pic_url in enumerate(pic_urls):
        try:
            pic = requests.get(pic_url,timeout=10)
            string = path+'/'+str(i+1)+'.jpg'
            with open(string,'wb') as f:
                f.write(pic.content)
                print('成功下载第%s张图片' % (str(i+1)))
        except Exception as e:
            print('下载%s张图片时失败：%s' % (str(i+1),str(pic_url)))
            print(e)
            continue


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

if __name__ == '__main__':
    keyword = '典邪视频'
    url_init_first = r'http://image.baidu.com/search/flip?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1497491098685_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&ctd=1497491098685%5E00_1519X735&word='
    url_init = url_init_first + urllib.parse.quote(keyword, safe='/')
    print(str(url_init))
    all_pic_urls = []
    fanye_count = 0
    mkdir(keyword)
    thread1 = myThread('del', [], keyword)
    thread1.start()
    onepage_urls,fanye_url = get_onepage_urls(url_init)
    all_pic_urls.extend(onepage_urls)
    while 1:
        if fanye_count>=10:
            break
        onepage_urls,fanye_url = get_onepage_urls(fanye_url)
        fanye_count+=1
        print('第%s页' %fanye_count)
        if fanye_url == '' and onepage_urls==[]:
            break
        all_pic_urls.extend(onepage_urls)
    thread2 = myThread('save',all_pic_urls,keyword)
    thread2.start()