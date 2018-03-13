#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: spyder.py
@time: 2018/03/13 
"""

import sys,urllib2
from bs4 import BeautifulSoup
import xlwt

reload(sys)
sys.setdefaultencoding('utf-8')


def getHtml(url):
    try:
        #请求抓取对象
        request = urllib2.Request(url)
        #响应对象
        response = urllib2.urlopen(request)
        #读取网页
        html = response.read()
    except:
        return False
    return html



def getData(baseurl):
    print u'******正在获取当前南京热映的电影*****'
    html = getHtml(baseurl)
    soup = BeautifulSoup(html)
    nowplaying = soup.find_all('div', id = 'nowplaying')
    dataList = []
    movie_list = nowplaying[0].find_all('li', class_='list-item')
    print u'电影数量'+str(len(movie_list))
    for movie in movie_list:
        data = {}
        now_url = 'https://movie.douban.com/subject/' + movie['data-subject'] + '/comments'
        html_comments = getHtml(now_url)
        if html_comments == False:
            continue
        soup_comments = BeautifulSoup(html_comments)
        data['name'] = movie['data-title']
        data['score'] = movie['data-score']
        comments = [];
        for item in soup_comments.find_all('div' ,class_ ='comment'):
            comments.append(item.find_all('p')[0].string)
        data['comments'] = comments
        dataList.append(data)
        print '电影名： '+data['name']+' 评分：'+ data['score']+' 点评数量：'+ str(len(data['comments']))
    return dataList

def saveData(datalist):
    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    sheet=book.add_sheet('豆瓣南京在映影评',cell_overwrite_ok=True)
    col = ('电影名','评分','评论')
    for i in range(0,3):
        sheet.write(0,i,col[i])
    num = 1
    for item in datalist:
        for i in range(0,len(item['comments'])):
            sheet.write(num,0,item['name'])
            sheet.write(num,1,item['score'])
            sheet.write(num,2,item['comments'][i])
            #print item['comments'][i]
            num+=1
    book.save(u'豆瓣南京在映影评.xls')


def spyder():
    baseurl = 'https://movie.douban.com/cinema/nowplaying/nanjing/'
    datalist = getData(baseurl)
    saveData(datalist)
spyder()
