from PIL import Image
import glob
import os

def image_stitching(dir, filename):
    width = 28160
    height = 38400
    # height = 14336
    new_image = Image.new('RGB', (width, height))
    width_index = 0
    height_index = 0
    image_path_list = glob.glob(os.path.join(dir, '*.png'))
    num = len(image_path_list)
    print(num)
    for i in range(1, num + 1): #561 401
        img = Image.open(dir + filename + str(i) + '.png')
        img_clipped = img.crop([255, 255, 255+512, 255+512])
        new_image.paste(img_clipped, (width_index * 512, height_index * 512))
        width_index += 1
        if(width_index == (width // 512)):
            width_index = 0
            height_index += 1
    # new_image.crop([0, 0, 10096, 14134])
    new_image.save('/data1/gjc23/loc_jinjiang.png')

if __name__ == '__main__':
    dir = '/data1/zwy/jinjiang_tif/jinjiang_building/'
    filename = 'loc_jinjiang_'
    image_stitching(dir, filename)


