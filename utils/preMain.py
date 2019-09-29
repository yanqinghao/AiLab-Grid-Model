# coding: utf-8

from utils import *
import sys

'''
preMain(imgpath, saved_path, model_path, height_real, width_real):


    :param imgpath: 可以是单张图片也可以是图片文件夹
    :param saved_path: 保存路径
    :param model_path: 模型的文件夹目录
    :param height_real: 真实高度
    :param width_real: 真实宽度

    :return:
    在保存路径中会生成四个文件夹如下
    │ 
    ├── GridVisualization: 带局部自适应网格的孔隙识别图
    │  
    ├── JPEGImages: 原图
    │  
    ├── SegmentationClassPNG：png图
    │  
    ├── SegmentationClassVisualization： 分割图
    │  
    └── response.json

'''

if __name__ == '__main__':
    param = []
    for i in range(1, len(sys.argv)):
        print(sys.argv[i])
        param.append(sys.argv[i])

    preMain(param[0],param[1],param[2],float(param[3]),float(param[4]))