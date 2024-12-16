from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = 200000000


def heatmap_points(filename):
    image = np.array(Image.open(filename))
    height = image.shape[0]
    width = image.shape[1]
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # if i == 0 and j == 0 and image[i][j] > image[i + 1][j] \
            #         and image[i][j] > image[i][j + 1] and image[i][j] > image[i + 1][j + 1]:
            #     print('height, width, value = ', i, j, image[i][j])
            #     continue
            # if i == 0 and image[i][j] > image[i][j - 1] and image[i][j] > image[i][j + 1] \
            #         and image[i][j] > image[i + 1][j - 1] and image[i][j] > image[i + 1][j] \
            #         and image[i][j] > image[i + 1][j + 1]:
            #     print('height, width, value = ', i, j, image[i][j])
            #     continue
            # if j == 0 and image[i][j] > image[i - 1][j] and image[i][j] > image[i + 1][j] \
            #         and image[i][j] > image[i - 1][j + 1] and image[i][j] > image[i][j + 1] \
            #         and image[i][j] > image[i + 1][j + 1]:
            #     print('height, width, value = ', i, j, image[i][j])
            #     continue
            # if i == height - 1 and j == width - 1 and image[i][j] > image[i][j - 1] \
            #         and image[i][j] > image[i - 1][j] and image[i][j] > image[i - 1][j - 1]:
            #     print('height, width, value = ', i, j, image[i][j])
            #     continue
            # if i == height - 1 and image[i][j] > image[i][j - 1] and image[i][j] > image[i][j + 1] \
            #         and image[i][j] > image[i - 1][j - 1] and image[i][j] > image[i - 1][j] \
            #         and image[i][j] > image[i - 1][j + 1]:
            #     print('height, width, value = ', i, j, image[i][j])
            #     continue
            # if j == width - 1 and image[i][j] > image[i - 1][j] and image[i][j] > image[i + 1][j] \
            #         and image[i][j] > image[i - 1][j - 1] and image[i][j] > image[i][j - 1] \
            #         and image[i][j] > image[i + 1][j - 1]:
            #     print('height, width, value = ', i, j, image[i][j])
            #     continue
            if image[i][j] > image[i - 1][j - 1] and image[i][j] > image[i - 1][j] \
                    and image[i][j] > image[i - 1][j + 1] and image[i][j] > image[i][j - 1] \
                    and image[i][j] > image[i][j + 1] and image[i][j] > image[i + 1][j - 1] \
                    and image[i][j] > image[i + 1][j] and image[i][j] > image[i + 1][j + 1]:
                print('height, width, value = ', i, j, image[i][j])
                # continue

if __name__ == '__main__':
    filename = 'G:\data\mianzhu_heatmap_normalization.tif'
    heatmap_points(filename)
