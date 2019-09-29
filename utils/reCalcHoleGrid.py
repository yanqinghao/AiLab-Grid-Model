# coding: utf-8

from utils import *
import sys
import json

'''
def reCalcHoleGrid(png_path, json_path, flag, rectangle=[], points=[]):

    :param png_path: 原来的png图片位置
    :param json_path: json文件位置
    :param rectangle: [左上height, 左上width, 右下height, 右下width]
    :param flag: true 为添加， false 为删除
    :parma points: 折线区域

    :return: 把png,img,grid,vis都重新覆盖
'''


if __name__ == '__main__':
    param = []
    for i in range(1, len(sys.argv)):
        print(sys.argv[i])
        param.append(sys.argv[i])

    reCalcHoleGrid(param[0],param[1],json.loads(param[2]),json.loads(param[3]))