import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
import math
import pandas as pd


def calculate_angle(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    angle = math.degrees(math.atan2(dx, dy))
    if angle < 0:
        angle += 360
    return angle


def get_direction_index(angle):
    direction_ranges = [
        (337.5, 360), (0, 22.5),  # North
        (22.5, 67.5),  # Northeast
        (67.5, 112.5),  # East
        (112.5, 157.5),  # Southeast
        (157.5, 202.5),  # South
        (202.5, 247.5),  # Southwest
        (247.5, 292.5),  # West
        (292.5, 337.5)  # Northwest
    ]

    for i, (start, end) in enumerate(direction_ranges):
        if i == 0 and (angle >= start or angle <= end):
            return 0
        elif start <= angle < end:
            return i if i > 0 else 1


def create_neighborhood_table(shp_path, output_path=None):
    gdf = gpd.read_file(shp_path)

    nodes = set()
    node_coords = {}

    for idx, row in gdf.iterrows():
        if isinstance(row.geometry, LineString):
            coords = list(row.geometry.coords)
            for coord in [coords[0], coords[-1]]:
                nodes.add(coord)
                node_coords[coord] = Point(coord)

    nodes = list(nodes)
    node_indices = {node: i for i, node in enumerate(nodes)}

    n_nodes = len(nodes)
    neighborhood = np.full((n_nodes, 9), -1, dtype=int)  # 第一列存储节点ID

    neighborhood[:, 0] = range(n_nodes)

    for idx, row in gdf.iterrows():
        if isinstance(row.geometry, LineString):
            coords = list(row.geometry.coords)

            for i in range(len(coords) - 1):
                start_coord = coords[i]
                end_coord = coords[i + 1]

                if start_coord in node_indices and end_coord in node_indices:
                    start_idx = node_indices[start_coord]
                    end_idx = node_indices[end_coord]

                    angle = calculate_angle(start_coord, end_coord)
                    direction_idx = get_direction_index(angle)

                    neighborhood[start_idx, direction_idx + 1] = end_idx

                    reverse_direction = (direction_idx + 4) % 8
                    neighborhood[end_idx, reverse_direction + 1] = start_idx

    columns = ['NodeID', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    df = pd.DataFrame(neighborhood, columns=columns)

    if output_path:
        df.to_csv(output_path, index=False)
        np.save(output_path.replace('.csv', '.npy'), neighborhood)

        node_coords_array = np.array([[i, coord[0], coord[1]] for coord, i in node_indices.items()])
        np.save(output_path.replace('.csv', '_coords.npy'), node_coords_array)

    return df, node_coords_array


if __name__ == "__main__":
    shp_path = "mianzhu_road.shp"
    output_path = "mianzhu_road.csv"

    df, node_coords = create_neighborhood_table(shp_path, output_path)
    print("Done！")