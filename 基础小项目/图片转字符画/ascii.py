#-*- coding:utf-8 _*-  
""" 
@author:KING 
@file: ascii.py 
@time: 2018/03/15 
"""

from PIL import Image
import argparse
import cv2

ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='timg.jpg')#输入文件
parser.add_argument('-o','--output')#输出文件
parser.add_argument('--width', type=int, default=80)
parser.add_argument('--height', type=int, default=80)

#获取参数
args = parser.parse_args()
IMG = args.file
WIDTH = args.width
HEIGHT = args.height
OUTPUT = args.output

fout = open(u'输出文件.txt','w')
length = len(ascii_char)
unit = (256.0+1)/length

#根据灰度值映射至自定义ascii表中
def get_char(r,g,b,alpha = 256):
    #gray ＝ 0.2126 * r + 0.7152 * g + 0.0722 * b
    gray = int(0.2126*r+0.7152*g+0.0722*b)
    return ascii_char[int(gray/unit)]

def main():
    im = Image.open(IMG)
    im = im.resize((WIDTH,HEIGHT),Image.NEAREST)
    txt = ""
    for i in range(HEIGHT):
        for j in range(WIDTH):
            txt+=get_char(*im.getpixel((j,i)))
        txt+='\n'
    #print txt
    fout.write(txt)

def main2():
    image = cv2.imread(IMG)
    image = cv2.resize(image,(HEIGHT,WIDTH),interpolation=cv2.INTER_CUBIC)
    image_gray = cv2.cvtColor(image ,cv2.COLOR_BGR2GRAY)
    txt = ""
    (B,G,R) = cv2.split(image)
    #取出RGB矩阵
    for i in range(HEIGHT):
        for j in range(WIDTH):
            #txt+=get_char(R[i,j],G[i,j],B[i,j])
            #txt += get_char(image[i,j,2],image[i,j,1],image[i,j,0])
            txt += ascii_char[int(image_gray[i,j]/unit)]
        txt+='\n'
    fout.write(txt)


main2()



