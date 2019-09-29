# coding: utf-8

import PIL.Image
import PIL.ImageDraw
from PIL.Image import fromarray

from fastai.vision import *
from fastai.callbacks.hooks import *

import os
import cv2
import numpy as np
import itertools
from tqdm import tqdm
import time



def build_dir(target):
    # 对输出生成文件夹
    if not os.path.exists(target):
        os.mkdir(target)
    if not os.path.exists('%s/SegmentationClassPNG' % target):
        os.mkdir('%s/SegmentationClassPNG' % target)
    if not os.path.exists('%s/JPEGImages' % target):
        os.mkdir('%s/JPEGImages' % target)
    if not os.path.exists('%s/SegmentationClassVisualization' % target):
        os.mkdir('%s/SegmentationClassVisualization' % target)
    if not os.path.exists('%s/GridVisualization' % target):
        os.mkdir('%s/GridVisualization' % target)

def save_json(res_json, saved_path):
    json_path = '{}/response.json'.format(saved_path)
    if os.path.exists(json_path):
        dj = readJson(json_path)
        for key in res_json.keys():
            dj[key] = res_json[key]
        res2 = json.dumps(dj, indent=4, ensure_ascii=False)
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(res2)
        return 

    if os.path.splitext(saved_path)[1] == '.json':
        json_path = saved_path
        os.remove(saved_path)
    res2 = json.dumps(res_json, indent=4, ensure_ascii=False)
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(res2)

def readJson(JsPath):
    f = open(JsPath, encoding='utf-8')
    user_dic = json.load(f)
    return user_dic

def get_files(file_dir, extension):
    files = []
    for dirpath, dirnames, filenames in os.walk(file_dir):
        for file in filenames:
            if os.path.splitext(file)[1] == extension:
                files.append(os.path.join(dirpath, file))
    files.sort(key=str.lower)
    return files

def find8rope(img, h_i, w_i):
    height, width = img.shape[:2]
    res = []
    tt1 = [h_i]
    tt2 = [w_i]

    if min(max(h_i - 1, 0), height-1) != h_i:
        tt1.append(min(max(h_i - 1, 0), height-1))
    if min(max(w_i - 1, 0), width - 1) != w_i:
        # res.append([h_i, min(max(w_i - 1, 0), width - 1)])
        tt2.append(min(max(w_i - 1, 0), width - 1))
    if min(max(w_i + 1, 0), width - 1) != w_i:
        # res.append([h_i, min(max(w_i + 1, 0), width - 1)])
        tt2.append(min(max(w_i + 1, 0), width - 1))
    if min(max(h_i + 1, 0), height-1) != 0:
        # res.append([min(max(h_i + 1, 0), height-1), w_i])
        tt1.append(min(max(h_i + 1, 0), height-1))

    for i in itertools.product(tt1, tt2):
        res.append(i)
    res = list(set(res))
    res.remove((h_i, w_i))
    return res


def findMrope(img, msk, v):
    img2 = img
    if len(msk.shape) == 3:
        img2 = img[:, :, 0]
        msk = msk[:, :, 0]
    height, width = img2.shape[:2]
    count = 2
    maps = {}
    for h in range(height):
        for w in range(width):
            if img2[h,w] in v and (msk[h,w] ==0):
                maps[count] = [h, w, h, w]
                dsf(img2, msk, v, count, [(h,w)], maps)
                count += 1
    return maps

def dsf(img, msk, v, tmp, points, maps):
    if len(points) == 0:
        return None
    for point in points:
        if img[point[0], point[1]] in v and msk[point[0], point[1]] ==0:
            msk[point[0], point[1]] = tmp
            if point[0] < maps[tmp][0]:
                maps[tmp][0] = point[0]

            if point[0] > maps[tmp][2]:
                maps[tmp][2] = point[0]

            if point[1] < maps[tmp][1]:
                maps[tmp][1] = point[1]

            if point[1] > maps[tmp][3]:
                maps[tmp][3] = point[1]

            res = find8rope(img, point[0], point[1])
            points.extend(res)

def findRetangle(img, msk):
        msk_li = np.zeros_like(msk)
        maps = findMrope(msk, msk_li, v=[1])
        return maps

def compute_square(sq, square_full):
    if sq == 0:
        return 0
    if sq <= square_full*0.25 and sq >0:
        return 0.25
    if sq <= square_full*0.5 and sq > square_full*0.25:
        return 0.5
    if sq <= square_full*0.75 and sq > square_full*0.5:
        return 0.75
    if sq <= square_full and sq > square_full*0.75:
        return 1


def square_grid(height, width, heightBatchSize, widthBatchSize):
    # 纵向的数量
    he_te = int(height // heightBatchSize)
    # 横向的数量
    wi_te = int(width // widthBatchSize)
    # 计算网格数量
    full_grid = he_te * wi_te
    he_grid = he_te * compute_square(heightBatchSize * (width % widthBatchSize), heightBatchSize * widthBatchSize)
    wi_grid = wi_te * compute_square(widthBatchSize * (height % heightBatchSize), heightBatchSize * widthBatchSize)
    return full_grid + he_grid + wi_grid

def changeColor(img_raw, msk, rectangle):
    msk[rectangle[0]:rectangle[2],rectangle[1]:rectangle[3],:] = img_raw[rectangle[0]:rectangle[2],rectangle[1]:rectangle[3],:]*0.8
    return msk

def changeImgInfo(jpg_path, grid_path, vis_path, rectangle):
    jpg_info = cv2.imread(jpg_path)

    grid_info = changeColor(jpg_info, cv2.imread(grid_path), rectangle)
    vis_info = changeColor(jpg_info, cv2.imread(vis_path), rectangle)
    cv2.imwrite(grid_path, grid_info)
    cv2.imwrite(vis_path, vis_info)

def shape_to_mask(path, points, shape_type=None):
    mask = PIL.Image.open(path)
    draw = PIL.ImageDraw.Draw(mask)
    for points_pair in points:
        xy = [tuple(point) for point in points_pair]

        if shape_type == 'rectangle':
            assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
            draw.rectangle(xy, outline=0, fill=0)
        if shape_type == 'polygon':
            assert len(xy) > 2, 'Polygon must have points more than 2'
            draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=np.uint8)
    mask[:,:,1] = mask[:,:,0]
    mask[:,:,2] = mask[:,:,0]
    return mask

def getNewHoleDelete(height_real, width_real, rectangle, png_info):
    height, width = png_info.shape[:2]
    rec_width = 1000 / 40
    # 纵向网格的长度
    heightBatchSize = int(height * rec_width / height_real)
    # 横向网格的长度
    widthBatchSize = int(width * rec_width / width_real)

    hole_grid = 0.0
    height_rec = rectangle[0]
    width_rec = rectangle[1]
    height_rec1 = rectangle[2] - rectangle[0]
    width_rec1 = rectangle[3] - rectangle[1]
    # 纵向的数量
    he_te = int(height_rec1 // heightBatchSize)

    # 横向的数量
    wi_te = int(width_rec1 // widthBatchSize)

    t_index = []

    for i in itertools.product(range(he_te + 1), range(wi_te + 1)):
        t_index.append(i)

    for h_i, w_i in t_index:
        hs = (h_i + 1) * heightBatchSize
        ws = (w_i + 1) * widthBatchSize
        if h_i == he_te and hs > height:
            hs = height
        if w_i == wi_te and w_i > width:
            ws = width

        cropPng = png_info[h_i * heightBatchSize + height_rec:hs + height_rec,
                  w_i * widthBatchSize + width_rec:ws + width_rec, :]
        sq = np.sum(cropPng[:, :, 0] == 1)
        # 对该区域png设置为0
        png_info[h_i * heightBatchSize + height_rec:hs + height_rec,
        w_i * widthBatchSize + width_rec:ws + width_rec, :] = 0
        square = cropPng[:, :, 0].shape[0] * cropPng[:, :, 0].shape[1]

        if compute_square(sq, square) == 0.25:
            tmp_grid = 0.25 * compute_square(square, heightBatchSize * widthBatchSize)
            hole_grid += tmp_grid
            continue
        if compute_square(sq, square) == 0.5:
            tmp_grid = 0.5 * compute_square(square, heightBatchSize * widthBatchSize)
            hole_grid += tmp_grid
            continue
        if compute_square(sq, square) == 0.75:
            tmp_grid = 0.75 * compute_square(square, heightBatchSize * widthBatchSize)
            hole_grid += tmp_grid
            continue
        if compute_square(sq, square) == 1:
            tmp_grid = 1 * compute_square(square, heightBatchSize * widthBatchSize)
            hole_grid += tmp_grid
            continue

    return hole_grid,png_info

def splitimg(path):
    files = []
    target = ''
    if os.path.isdir(path):
        files = get_files(path, '.jpg')
        target = os.path.join(path, 'img_tmp')
    if os.path.splitext(path)[1] == '.jpg':
        files.append(path)
        target = os.path.join(os.path.dirname(path), 'img_tmp')
    if len(files) == 0:
        return '', None

    build_dir(target)
    shape_dict = {}
    for imgpath in files:
        img = cv2.imread(str(imgpath))
        file = os.path.basename(imgpath).replace('.jpg', '')
        shape_dict[file] = img.shape
        height, width = img.shape[:2]
        heightBatchSize = 258
        widthBatchSize = 344

        he_te = height // heightBatchSize
        wi_te = width // widthBatchSize
        #   获得索引
        t_index = []
        for i in itertools.product(range(he_te + 1), range(wi_te + 1)):
            t_index.append(i)
        count = 0
        for h_i, w_i in t_index:
            cropImg = np.ones((heightBatchSize, widthBatchSize, 3)) * 255

            if h_i == he_te and w_i <= wi_te - 1:
                cropImg[:height % heightBatchSize, :] = img[h_i * heightBatchSize:height,
                                                        w_i * widthBatchSize:(w_i + 1) * widthBatchSize]
            if h_i <= he_te - 1 and w_i == wi_te:
                cropImg[:, :width % widthBatchSize] = img[h_i * heightBatchSize:(h_i + 1) * heightBatchSize,
                                                      w_i * widthBatchSize:width]
            if h_i == he_te and w_i == wi_te:
                cropImg[:height % heightBatchSize, :width % widthBatchSize] = img[h_i * heightBatchSize:height,
                                                                              w_i * widthBatchSize:width]
            if h_i <= he_te - 1 and w_i <= wi_te - 1:
                cropImg = img[h_i * heightBatchSize:(h_i + 1) * heightBatchSize,
                          w_i * widthBatchSize:(w_i + 1) * widthBatchSize]

            img_target_file = '%s/JPEGImages/%s_%d.jpg' % (target, file, count,)
            count += 1
            cv2.imwrite(img_target_file, cropImg)
    return target, shape_dict


def predictImg(foldpath, target, learn):
    build_dir(target)
    files = get_files(foldpath, '.jpg')
    pbar_files = tqdm(files, desc='Predict: ')
    for f in pbar_files:
        file = os.path.basename(f).replace('.jpg', '')
        p = learn.predict(open_image(f))[0].data.numpy()
        p = p.astype(np.uint8)
        p.resize(258,344)
        png_target_file = '%s/SegmentationClassPNG/%s.png' % (target,
                                                                 file,)

        PIL.Image.fromarray(p).save(png_target_file)


def gridImg(img_info, png_info, vis_mask, height, width, height_real, width_real):
    rec_width = 1000 / 40
    # 纵向网格的长度
    heightBatchSize = int(height * rec_width / height_real)
    # 横向网格的长度
    widthBatchSize = int(width * rec_width / width_real)

    maps = findRetangle(img_info, png_info)
    all_grid = square_grid(height, width, heightBatchSize, widthBatchSize)
    hole_grid = 0.0
    for v in maps.values():

        height_rec = v[0]
        width_rec = v[1]
        # (v[1], v[0]), (v[3], v[2])
        height_rec1 = v[2] - v[0]
        width_rec1 = v[3] - v[1]
        # 纵向的数量
        he_te = int(height_rec1 // heightBatchSize)

        # 横向的数量
        wi_te = int(width_rec1 // widthBatchSize)

        t_index = []

        for i in itertools.product(range(he_te + 1), range(wi_te + 1)):
            t_index.append(i)

        for h_i, w_i in t_index:
            hs = (h_i + 1) * heightBatchSize
            ws = (w_i + 1) * widthBatchSize
            if h_i == he_te and hs > height:
                hs = height
            if w_i == wi_te and w_i > width:
                ws = width

            cropPng = png_info[h_i * heightBatchSize+height_rec:hs+height_rec, w_i * widthBatchSize+width_rec:ws+width_rec, :]
            sq = np.sum(cropPng[:, :, 0] == 1)
            square = cropPng[:, :, 0].shape[0] * cropPng[:, :, 0].shape[1]
            tmp = np.where(cropPng[:, :, 0] == 1)
            if sq < square*0.08:
                continue

            cv2.line(img_info, (w_i * widthBatchSize+width_rec, h_i * heightBatchSize+height_rec), (w_i * widthBatchSize+width_rec, hs+height_rec), (125, 255, 51), 1)
            cv2.line(img_info, (w_i * widthBatchSize+width_rec, h_i * heightBatchSize+height_rec), (ws+width_rec, h_i * heightBatchSize+height_rec), (125, 255, 51),
                     1)
            cv2.line(img_info, (ws+width_rec, h_i * heightBatchSize+height_rec), (ws+width_rec, hs+height_rec), (125, 255, 51), 1)
            cv2.line(img_info, (w_i * widthBatchSize+width_rec, hs+height_rec), (ws+width_rec, hs+height_rec), (125, 255, 51), 1)


            if compute_square(sq, square) == 0.25:
                vis_mask[h_i * heightBatchSize+height_rec:hs+height_rec,
                         w_i * widthBatchSize+width_rec:ws+width_rec, :][tmp] = (0, 0, 255)
                tmp_grid = 0.25 * compute_square(square, heightBatchSize*widthBatchSize)
                hole_grid += tmp_grid
                continue
            if compute_square(sq, square) == 0.5:
                vis_mask[h_i * heightBatchSize + height_rec:hs + height_rec,
                         w_i * widthBatchSize + width_rec:ws + width_rec, :][tmp] = (0, 255, 0)
                tmp_grid = 0.5 * compute_square(square, heightBatchSize*widthBatchSize)
                hole_grid += tmp_grid
                continue
            if compute_square(sq, square) == 0.75:
                vis_mask[h_i * heightBatchSize + height_rec:hs + height_rec,
                         w_i * widthBatchSize + width_rec:ws + width_rec, :][tmp] = (0, 255, 255)
                tmp_grid = 0.75 * compute_square(square, heightBatchSize*widthBatchSize)
                hole_grid += tmp_grid
                continue
            if compute_square(sq, square) == 1:
                vis_mask[h_i * heightBatchSize + height_rec:hs + height_rec,
                w_i * widthBatchSize + width_rec:ws + width_rec, :][tmp] = (255, 0, 0)
                tmp_grid = 1 * compute_square(square, heightBatchSize*widthBatchSize)
                hole_grid += tmp_grid
                continue

    vis_img = cv2.addWeighted(img_info, 0.8, vis_mask, 0.2, 0)
    return vis_img, maps, all_grid, hole_grid

def mergeImg_shape(source, target, shape_dict, height_real, width_real):
    res_json = {}
    build_dir(target)
    filenames = get_files(os.path.join(source, 'SegmentationClassPNG'), '.png')
    imgFile_dict = {}
    pngFile_dict = {}
    for files in filenames:
        file = os.path.basename(files).replace('.png', '')
        img_file = '%s/JPEGImages/%s.jpg' % (source, file)
        png_file = '%s/SegmentationClassPNG/%s.png' % (source, file)
        if not os.path.exists(png_file) or not os.path.exists(img_file):
            print('图片路径不存在%s!!!!!' % file)
            continue
        if os.path.basename(file).split('_')[0] not in pngFile_dict.keys():
            pngFile_dict[os.path.basename(file).split('_')[0]] = 1
            imgFile_dict[os.path.basename(file).split('_')[0]] = 1

    pbar_files = tqdm(imgFile_dict.keys(), desc='Merge: ')
    for k in pbar_files:
        res_json[str(k)] = {
            'full_grid': 0.0,
            'hole_grid': 0.0,
            'height_real': height_real*shape_dict[k][0],
            'width_real': width_real*shape_dict[k][1],
            'height_pixel': shape_dict[k][0],
            'width_pixel': shape_dict[k][1],
            'rect': []
        }
        heightBatchSize, widthBatchSize = 258, 344
        height_tmp, width_tmp = shape_dict[k][:2]
        height, width = ((height_tmp // heightBatchSize)+1) * heightBatchSize, ((width_tmp // widthBatchSize)+1) * widthBatchSize
        img_mask = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        png_mask = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        vis_mask = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        vis_mask2 = np.zeros(shape=(height_tmp, width_tmp, 3), dtype=np.uint8)


        he_te = height//heightBatchSize
        wi_te = width//widthBatchSize
        #  获得索引
        t_index = []
        for i in itertools.product(range(he_te), range(wi_te)):
            t_index.append(i)
        count = 0
        for h_i, w_i in t_index:        
            if count == len(t_index):
                break
            img_file = '%s/JPEGImages/%s_%s.jpg' % (source, k, str(count))
            png_file = '%s/SegmentationClassPNG/%s_%s.png' % (source, k, str(count))

            img_tmp = cv2.imread(str(img_file))
            png_tmp = cv2.imread(str(png_file))
            img_mask[h_i * heightBatchSize:(h_i + 1) * heightBatchSize, w_i * widthBatchSize:(w_i + 1) * widthBatchSize,:] = img_tmp[:,:,:]
            png_mask[h_i * heightBatchSize:(h_i + 1) * heightBatchSize, w_i * widthBatchSize:(w_i + 1) * widthBatchSize,:] = png_tmp[:,:,:]
            os.remove(img_file)
            os.remove(png_file)
            count += 1


        tt = png_mask[:,:]
        vis_mask[png_mask[:,:,0] == 1] = (255, 0, 0)
        vis_img = cv2.addWeighted(img_mask, 0.8, vis_mask, 0.2, 0)
    
        img_target_file = '%s/JPEGImages/%s.jpg' % (target,
                                                    k,)
        png_target_file = '%s/SegmentationClassPNG/%s.png' % (target,
                                                              k,)

        vis_target_file = '%s/SegmentationClassVisualization/%s.jpg' % (target,
                                                              k,)
        
        vis2_target_file = '%s/GridVisualization/%s_0.jpg' % (target,
                                                              k,)
        cv2.imwrite(img_target_file, img_mask[:shape_dict[k][0],:shape_dict[k][1],:])
        cv2.imwrite(png_target_file, png_mask[:shape_dict[k][0],:shape_dict[k][1],:])
        cv2.imwrite(vis_target_file, vis_img[:shape_dict[k][0],:shape_dict[k][1],:])
        
        vis_img2, maps, full_grid, hole_grid = gridImg(img_info=img_mask[:shape_dict[k][0],:shape_dict[k][1],:],
                           png_info=png_mask[:shape_dict[k][0],:shape_dict[k][1],:],
                           vis_mask=vis_mask2,
                           height=height_tmp, width=width_tmp,
                           height_real=height_real*shape_dict[k][0], width_real=width_real*shape_dict[k][1])
        cv2.imwrite(vis2_target_file, vis_img2)
        res_json[str(k)]['rect'] = list(maps.values())
        res_json[str(k)]['full_grid'] = full_grid
        res_json[str(k)]['hole_grid'] = hole_grid
    save_json(res_json,target)


def reCalcHoleGrid(png_path, json_path, rectangle=[], points=[]):
    '''

    :param png_path: 原来的png图片位置
    :param json_path: json文件位置
    :param rectangle: [左上height, 左上width, 右下height, 右下width]
    :param flag: true 为添加， false 为删除
    :parma points: 折线区域

    :return: 把png,img,grid,vis都重新覆盖
    '''
    png_files = tqdm([png_path], desc='Predict: ')
    for png_path in png_files:
        f_name = os.path.basename(png_path).replace(os.path.splitext(png_path)[1] , '')
        root_dir = os.path.dirname(json_path)
        file_dict = readJson(json_path)

        png_info = cv2.imread(png_path)

        jpg_path = os.path.join(os.path.join(root_dir, 'JPEGImages'),
                                f_name + '.jpg')
        grid_path = os.path.join(os.path.join(root_dir, 'GridVisualization'),
                                 f_name + '_0.jpg')
        vis_path = os.path.join(os.path.join(root_dir, 'SegmentationClassVisualization'),
                                f_name + '.jpg')
        if len(rectangle) != 0:
            for rect in rectangle:
                hole_grid, png_info = getNewHoleDelete(file_dict[f_name]['height_real'], file_dict[f_name]['width_real'],
                                                       rect, png_info)
                changeImgInfo(jpg_path, str(grid_path), str(vis_path), rect)
                cv2.imwrite(png_path, png_info)

                file_dict[f_name]['hole_grid'] = file_dict[f_name]['hole_grid'] - hole_grid
                if rect in file_dict[f_name]['rect']:
                    file_dict[f_name]['rect'].remove(rect)
            save_json(file_dict, json_path)
        if len(points) != 0:
            # 操作points
            png_info = shape_to_mask(png_path, points, shape_type='polygon')
            img_info = cv2.imread(jpg_path)
            vis_mask = np.zeros_like(img_info, dtype=np.uint8)
            height, width = img_info.shape[:2]
            height_real, width_real = file_dict[f_name]['height_real'], file_dict[f_name]['width_real']

            vis_mask[png_info[:, :, 0] == 1] = (255, 0, 0)
            vis_img2 = cv2.addWeighted(img_info, 0.8, vis_mask, 0.2, 0)
            cv2.imwrite(vis_path, vis_img2)

            vis_img, maps, full_grid, hole_grid = gridImg(img_info, png_info, vis_mask,
                                                          height, width, height_real, width_real)
            file_dict[f_name]['rect'] = list(maps.values())
            file_dict[f_name]['full_grid'] = full_grid
            file_dict[f_name]['hole_grid'] = hole_grid

            cv2.imwrite(png_path, png_info)
            cv2.imwrite(grid_path, vis_img)
            save_json(file_dict, json_path)

def preMain(imgpath, saved_path, model_path, height_real, width_real):
    '''

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

    l2 = load_learner(path=model_path, fname='trained_model-522.pkl')
    
    target, shape_dict = splitimg(imgpath)
    tt = os.path.join(target, 'JPEGImages')
    predictImg(tt, target, l2)
    mergeImg_shape(target, saved_path, shape_dict, height_real, width_real)
    
    shutil.rmtree(target, True)
