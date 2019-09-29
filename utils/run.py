# coding: utf-8

from utils import *
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
    imgpath = './different-img/'
    saved_get = './saved_img_valid10'

    points = [[563, 801],
              [475, 801],
              [451, 800],
              [435, 798],
              [434, 797],
              [434, 792],
              [436, 784],
              [437, 783],
              [442, 779],
              [450, 776],
              [465, 777],
              [477, 778],
              [529, 775],
              [555, 776],
              [558, 777],
              [564, 783]]

    # preMain(imgpath=imgpath, saved_path=saved_get, model_path='./model', height_real=1061.8, width_real=1416.5)
    reCalcHoleGrid(png_path='./saved_img_valid10/SegmentationClassPNG/附着物4.png', json_path='./saved_img_valid10/response.json',
                    rectangle=[], flag=True, points=points)