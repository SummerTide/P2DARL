# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 2022
@author: suiyingy
"""
import cv2
import numpy as np
import os
import json
from collections import defaultdict


def cocojson2png(coco_dir, json_path='mianzhu_post_disaster_image_RGB_34.json', cls_type='train2017', save_dir='G:\毕业论文\影像裁剪\label'):
    save_path = os.path.join(save_dir, cls_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    annotation_file = os.path.join(coco_dir, 'annotations', json_path)
    with open(annotation_file, 'r', encoding='utf-8') as annf:
        annotations = json.load(annf)
        images = [i['id'] for i in annotations['images']]

    img_anno = defaultdict(list)
    for anno in annotations['annotations']:
        for img_id in images:
            if anno['image_id'] == img_id:
                img_anno[img_id].append(anno)
    imgid_file = {}
    for im in annotations['images']:
        imgid_file[im['id']] = im['file_name']

    for img_idx in img_anno:
        image = cv2.imread(coco_dir + 'images/' + cls_type + '/' + imgid_file[img_idx])
        h, w, _ = image.shape
        instance_png = np.zeros((h, w), dtype=np.uint8)
        for idx, ann in enumerate(img_anno[img_idx]):
            im_mask = np.zeros((h, w), dtype=np.uint8)
            mask = []
            for an in ann['segmentation']:
                ct = np.expand_dims(np.array(an), 0).astype(int)
                contour = np.stack((ct[:, ::2], ct[:, 1::2])).T
                mask.append(contour)
            imm = cv2.drawContours(im_mask, mask, -1, 1, -1)
            imm = imm * (1000 * anno['category_id'] + idx)
            instance_png = instance_png + imm
            instance_png = np.clip(instance_png, 0, 255)
        instance_png = np.expand_dims(instance_png, axis=2).repeat(3, axis=2).astype(np.uint8)

        print(instance_png.shape)
        cv2.imwrite(os.path.join(save_path, imgid_file[img_idx].split('.')[0] + ".png"), instance_png)


if __name__ == '__main__':
    coco_dir = "G:\label"
    cocojson2png(coco_dir, json_path='mianzhu_post_disaster_image_RGB_34.json')