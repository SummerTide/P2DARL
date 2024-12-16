from PIL import Image
import numpy as np
from skimage import morphology

Image.MAX_IMAGE_PIXELS = 200000000

def zhang_method(filename):
    image = Image.open(filename).convert('L')   # [255 255 255] -> [-1] [0 0 0] -> [0]
    # print(image.size)
    image = np.array(image, dtype=np.int8) * (-1)
    # print(image.shape)
    # height = thinned_image.shape[0]
    # width = thinned_image.shape[1]
    # dilation erosion
    # image = morphology.binary_erosion(image, morphology.square(5)).astype(np.int8)
    # Image.fromarray(image.astype(np.int8)).save('data/mianzhu_road_erosion5.png')
    # dilation_image = morphology.binary_erosion(image, morphology.square(7)).astype(np.uint8)
    # Image.fromarray(dilation_image.astype(np.uint8)).save('data/mianzhu_road_erosion7.png')

    thinned_image = morphology.medial_axis(image).astype(np.uint8)

    # thinned_image_labeled = thinned_image * 1
    new_image = Image.fromarray(np.uint8(thinned_image))
    new_image.save('data/yushu_road_thinned.png')
    # new_image.save('data/mianzhu_road_erosion7_thinned.png')
    # print(thinned_image.shape)

    # print(height, width)
    # changing1 = changing2 = 1
    # while changing1 or changing2:
    #     # step 1:
    #     changing1 = []
    #     for i in range(1, height-1):
    #         for j in range(1, width-1):
    #             P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(i, j, thinned_image)
    #             B = sum(n)
    #             A = transitions(n)
    #             if (thinned_image[i][j] == 1 and
    #                     2 <= B <= 6 and
    #                     A == 1 and
    #                     P2 * P4 * P6 == 0 and
    #                     P4 * P6 * P8 == 0):
    #                 changing1.append((i, j))
    #     for i, j in changing1:
    #         thinned_image[i][j] = 0
    #
    #     # step 2:
    #     changing2 = []
    #     for i in range(1, height - 1):
    #         for j in range(1, width - 1):
    #             P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(i, j, thinned_image)
    #             B = sum(n)
    #             A = transitions(n)
    #             if (thinned_image[i][j] == 1 and
    #                     2 <= B <= 6 and
    #                     A == 1 and
    #                     P2 * P4 * P8 == 0 and
    #                     P2 * P6 * P8 == 0):
    #                 changing2.append((i, j))
    #     for i, j in changing2:
    #         thinned_image[i][j] = 0
    #
    # thinned_image_labeled = thinned_image * 2
    # Image.fromarray(thinned_image_labeled.astype(np.int8)).save('data/mianzhu_road_erosion10_thinned.png')

    # image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # n = neighbours(1, 1, image)
    # n = n + n[0:1]
    # for n1, n2 in zip(n, n[1:]):
    #     print(n1, n2)

# 定义像素点周围的8邻域
#   P9 P2 P3
#   P8 P1 P4
#   P7 P6 P5

# def neighbours(i, j, image):
#     return [image[i-1][j], image[i-1][j+1], image[i][j+1], image[i+1][j+1],  # P2,P3,P4,P5
#             image[i+1][j], image[i+1][j-1], image[i][j-1], image[i-1][j-1]]  # P6,P7,P8,P9
#
# # 计算邻域像素从0变化到1的次数
# def transitions(neighbours):
#     n = neighbours + neighbours[0:1]      # P2,P3,...,P8,P9,P2
#     return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))  # (P2,P3),(P3,P4),...,(P8,P9),(P9,P2)

if __name__ == '__main__':
    filename = 'G:\data\yushu_road.png'
    # filename = 'G:\data\yushu_post_disaster_image_116mask.png'
    zhang_method(filename)