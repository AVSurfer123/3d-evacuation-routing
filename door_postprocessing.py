import open3d as o3d
import numpy as np

print("Load a ply point cloud, print it, and render it")
ply_point_cloud = o3d.data.PLYPointCloud()
all = o3d.io.read_point_cloud("complete_color_pointcloud.ply")
# print(all)
downsampled = all.voxel_down_sample(voxel_size=0.05)
# downsampled.paint_uniform_color([.9, .9, .9])
# print(downsampled)
doors = o3d.io.read_point_cloud("door_pointcloud.ply")
doors.paint_uniform_color([0.8, 0.8, 0.8])
# print(doors)
# print(np.asarray(doors.points))
for point in np.asarray(doors.points):
    point[2] = 0

print("Radius oulier removal")
cl, ind = doors.remove_radius_outlier(nb_points=10, radius=0.5)
doors_radius = doors.select_by_index(ind)

print("Statistical oulier removal")
cl, ind = doors_radius.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=1)

inlier_cloud = doors_radius.select_by_index(ind)
inlier_cloud.paint_uniform_color([1, 0, 0])
print(inlier_cloud)
inliers = np.asarray(inlier_cloud.points)
print(inliers.shape)
np.save("door_points.npy", inliers)
# o3d.visualization.draw_geometries([downsampled, inlier_cloud])