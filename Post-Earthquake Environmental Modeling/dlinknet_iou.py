from PIL import Image
import numpy as np

def dlinknet_iou(filename_label, filename_dlink):
    image_label = np.array(Image.open(filename_label))
    image_dlink = np.array(Image.open(filename_dlink))
    height = image_label.shape[0]
    width = image_label.shape[1]
    intersection = 0.0
    union = 0.0
    for i in range(height):
        for j in range(width):
            if image_label[i][j] == 1 or (image_dlink[i][j] == [255, 255, 255]).all():
                union += 1
            if image_label[i][j] == 1 and (image_dlink[i][j] == [255, 255, 255]).all():
                intersection += 1

    iou = intersection / union
    print('Intersection = ', intersection)
    print('Union = ', union)
    print('IOU = ', iou)

if __name__ == '__main__':
    filename_label = 'G:\D-LinkNet\label_514.png'
    filename_dlink = 'G:\D-LinkNet\mianzhu_514.png'
    dlinknet_iou(filename_label, filename_dlink)