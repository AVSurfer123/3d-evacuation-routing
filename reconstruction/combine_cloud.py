import numpy as np
import open3d as o3d
import os

files = [f for f in os.listdir(".") if "cloud_color" in f]

combined = o3d.geometry.PointCloud()

for f in files:
    print(f)
    cloud = o3d.io.read_point_cloud(f)
    combined += cloud

downsampled = combined.voxel_down_sample(voxel_size=0.01)
print(combined, downsampled)
o3d.io.write_point_cloud("complete_color_pointcloud.ply", downsampled)

