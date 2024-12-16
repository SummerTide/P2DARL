import os
from osgeo import gdal

os.chdir(r'F:\RSIDEA_Data\Tool\Image')
image_name = 'yushu_post_disaster_image.tif'

image = gdal.Open(image_name)
band1 = image.GetRasterBand(1)  # Blue
band2 = image.GetRasterBand(2)  # Green
band3 = image.GetRasterBand(3)  # Red

gtiff_driver = gdal.GetDriverByName('GTIFF')
new_image = gtiff_driver.Create('yushu_post_disaster_image_RGB.tif', band1.XSize, band1.YSize, 3, band1.DataType)
new_image.SetProjection(image.GetProjection())
new_image.SetGeoTransform(image.GetGeoTransform())

in_data = band1.ReadAsArray()
out_data = new_image.GetRasterBand(3)
out_data.WriteArray(in_data)

in_data = band2.ReadAsArray()
out_data = new_image.GetRasterBand(2)
out_data.WriteArray(in_data)

in_data = band3.ReadAsArray()
out_data = new_image.GetRasterBand(1)
out_data.WriteArray(in_data)

# new_image band: RGB
new_image.FlushCache()
for i in range(1, 4):
    new_image.GetRasterBand(i).ComputeStatistics(False)

new_image.BuildOverviews('average', [2, 4, 8, 16, 32])

del new_image