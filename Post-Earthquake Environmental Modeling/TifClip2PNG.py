import os
import rasterio
import math
import numpy as np
import cv2 as cv

def TifClip2PNG(dir, file_name, width, height, save_path):
    """

    :param dir: 原影像路径
    :param file_name: 原影像文件名，应带.tif
    :param width: 裁剪后影像的宽
    :param height: 裁剪后影像的高
    :param save_path: 裁剪后影像的保存路径
    :return: 无

    """

    os.chdir(dir)
    data = rasterio.open(file_name)
    image = np.array(data.read()).transpose((1, 2, 0))
    print(image.shape)
    image_height = image.shape[0]
    image_width = image.shape[1]

    new_image_height = int(math.ceil(image_height / height) * height)
    new_image_width = int(math.ceil(image_width / width) * width)

    # # cv2.copyMakeBorder(src,top,bottom,left,right,borderType,value)
    # # 对四周进行填充
    image = cv.copyMakeBorder(image, height//4, new_image_height - image_height + height//4, width//4, new_image_width - image_width + width//4, cv.BORDER_CONSTANT, 0)
    # cv.imwrite('/data1/gjc23/' + os.path.splitext(file_name)[0] + '.png', image)

    _id = 1
    h = 0
    while h < new_image_height:
        w = 0
        while w < new_image_width:
            cropped = image[h:h + height, w:w + width]
            cv.imwrite(save_path + '/' + os.path.splitext(file_name)[0] + "_" + str(_id) + ".png", cropped)  # 保存为png格式
            _id += 1
            w += 512
        h += 512

if __name__ == '__main__':
    TifClip2PNG('/data1/zwy/jingjiang/', 'jingjiang.tif', 1024, 1024, "/data1/gjc23/jingjiang_clip_png")


