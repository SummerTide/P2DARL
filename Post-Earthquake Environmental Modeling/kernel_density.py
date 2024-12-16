import arcpy
from arcpy import env
from arcpy.sa import *
import os


def kernel_density_analysis(input_tif, output_folder, population_field=None,
                            cell_size=30, search_radius=None, area_unit="SQUARE_KILOMETERS"):
    try:
        if arcpy.CheckExtension("Spatial") == "Available":
            arcpy.CheckOutExtension("Spatial")
        else:
            raise arcpy.ExecuteError("Spatial Analyst许可不可用")

        env.workspace = output_folder
        env.overwriteOutput = True

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_density = os.path.join(output_folder, "yushu_kernel.tif")

        if population_field:
            outKernelDensity = KernelDensity(input_tif, population_field,
                                             cell_size, search_radius, area_unit)
        else:
            outKernelDensity = KernelDensity(input_tif,
                                             cell_size=cell_size,
                                             search_radius=search_radius,
                                             area_unit_scale_factor=area_unit)

        outKernelDensity.save(output_density)

        print(f"Save: {output_density}")

        # 计算基本统计量
        arcpy.CalculateStatistics_management(output_density)

        # 获取并打印结果统计信息
        mean = arcpy.GetRasterProperties_management(output_density, "MEAN")
        std = arcpy.GetRasterProperties_management(output_density, "STD")
        minimum = arcpy.GetRasterProperties_management(output_density, "MINIMUM")
        maximum = arcpy.GetRasterProperties_management(output_density, "MAXIMUM")

        print("\nResult:")
        print(f"mean: {mean.getOutput(0)}")
        print(f"std: {std.getOutput(0)}")
        print(f"minimum: {minimum.getOutput(0)}")
        print(f"maximum: {maximum.getOutput(0)}")

    except arcpy.ExecuteError:
        print(arcpy.GetMessages(2))
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        arcpy.CheckInExtension("Spatial")


def reclassify_density(density_raster, output_raster, remap_range,
                       reclass_field="Value"):
    try:
        print("Reclass...")

        remap = arcpy.sa.RemapRange(remap_range)

        outReclass = Reclassify(density_raster, reclass_field, remap, "NODATA")

        outReclass.save(output_raster)

        print(f"Save: {output_raster}")

    except arcpy.ExecuteError:
        print(arcpy.GetMessages(2))
    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    input_tif = r"data/mianzhu.tif"
    output_folder = r"result/"

    # 执行核密度分析
    kernel_density_analysis(
        input_tif=input_tif,
        output_folder=output_folder,
        population_field="POP",
        cell_size=30,
        search_radius=1000,
        area_unit="SQUARE_KILOMETERS"
    )

    density_result = os.path.join(output_folder, "mianzhu_kernel_density.tif")
    reclass_output = os.path.join(output_folder, "mianzhu_density_classified.tif")

    remap_range = [
        [0, 100, 1],  # 0-100 -> 1
        [100, 500, 2],  # 100-500 -> 2
        [500, 1000, 3],  # 500-1000 -> 3
        [1000, 5000, 4]  # 1000-5000 -> 4
    ]

    reclassify_density(density_result, reclass_output, remap_range)


if __name__ == "__main__":
    main()
