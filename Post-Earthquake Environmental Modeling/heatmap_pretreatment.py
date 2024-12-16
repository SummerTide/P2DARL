import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 200000000

"""
    Background: [0 0 0] 0
    Non-Damage:  [255 255 255]  0
    Minor Damage: [0 255 0] 1
    Major Damage: [248 179 101] 2
    Destroyed:  [255 0 0]   3
"""
def PNG_to_heatmap(filename):
    image = Image.open(filename)
    image = np.array(image)
    height = image.shape[0]
    width = image.shape[1]

    heatmap = np.zeros((height, width), dtype=np.int8)

    # non_damage = np.array([255, 255, 255])
    minor_damage = np.array([0, 255, 0])
    major_damage = np.array([248, 179, 101])
    destroyed = np.array([255, 0, 0])

    for i in range(0, height):
        for j in range(0, width):
            # if (image[i][j] == non_damage).all():
            #     heatmap[i][j] = 1
            if (image[i][j] == minor_damage).all():
                heatmap[i][j] = 1
            elif (image[i][j] == major_damage).all():
                heatmap[i][j] = 2
            elif (image[i][j] == destroyed).all():
                heatmap[i][j] = 3
            else:
                heatmap[i][j] = 0
    new_image = Image.fromarray(heatmap)
    new_image.save('yushu_dam_处理.png')


if __name__ == '__main__':
    filename = 'G:\data\yushu_dam_.png'
    PNG_to_heatmap(filename)

