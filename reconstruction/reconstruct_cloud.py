import json
import os
import imageio as iio
import numpy as np
import matplotlib.pyplot as plt
import time
from pprint import pprint
import open3d as o3d

start = time.time()

DATA_DIR = '/storage/pointclouds/area_3/data'
POSE_DIR = os.path.join(DATA_DIR, 'pose')
DEPTH_DIR = os.path.join(DATA_DIR, 'depth')

pose = sorted(os.listdir(POSE_DIR))
depth = sorted(os.listdir(DEPTH_DIR))
print(len(pose), len(depth))

NUM_IMAGES = 10
all_points = []

MAX_DEPTH = 20 # meters

for i in range(1, NUM_IMAGES + 1): #change this 3

    print("Img name:", depth[i])

    # load pose json
    with open(os.path.join(POSE_DIR, pose[i])) as f:
        data_pose = json.load(f)

    # camera parameters
    K = np.array(data_pose["camera_k_matrix"])

    # extrinsic parameters
    RT = np.array(data_pose["camera_rt_matrix"])
    R = RT[:3, :3]
    T = RT[:, 3]

    print(K)
    print(RT)
    print(np.array(data_pose['camera_location']))

    # load depth png
    img_depth = iio.imread(os.path.join(DEPTH_DIR, depth[i]))/512.0 #in meters
    print("Max depth:", img_depth.max())

    # plt.hist(img_depth)
    # plt.show()

    # create list of pixels on image plane
    # pixels = np.zeros((img_depth.shape[0], img_depth.shape[1], 3))
    # count = 0
    # for i in range(0,img_depth.shape[0]):
    #     for j in range(0, img_depth.shape[1]):
    #         pixels[count] = [i * img_depth[i,j], j * img_depth[i,j], img_depth[i,j]]
    #         count += 1
    xs = np.arange(img_depth.shape[0])
    ys = np.arange(img_depth.shape[1])
    yS, xs = np.meshgrid(ys, xs, sparse=True)
    pixels = np.stack([img_depth * ys, img_depth * xs, img_depth], axis=2)
    flattened = np.reshape(pixels, (-1, 3))

    mask = flattened[:, 2] < MAX_DEPTH
    print(mask.shape)
    filtered = flattened[mask]
    print("Filtered shape", filtered.shape)
    
    # compute 3D points in world system
    # print("RT", RT) # RT is 3x4 but need to make it a square matrix to invert
    # points_3D = (np.linalg.inv(K) @ pixels.T).T
    # ones = np.ones((1, points_3D.shape[0]))
    # homo = np.array([0, 0, 0, 1])
    # points_3D_world = (np.linalg.inv(np.vstack((RT, homo))) @ np.vstack((points_3D.T, ones))).T
    # print(points_3D_world.shape)

    points_world = ((filtered @ np.linalg.inv(K).T) - T) @ R
    print(points_world.shape)
    
    all_points.append(points_world)

clouds = []
for points in all_points:
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(points)
    downsampled = point_cloud_o3d.voxel_down_sample(voxel_size=0.01)
    downsampled.paint_uniform_color(np.random.uniform(size=3))
    clouds.append(downsampled)

o3d.visualization.draw_geometries(clouds)

all_points = np.concatenate(all_points, axis=0)
print(all_points.shape)

end = time.time()
print(end - start, " seconds")

point_cloud_o3d = o3d.geometry.PointCloud()
point_cloud_o3d.points = o3d.utility.Vector3dVector(all_points)
downsampled = point_cloud_o3d.voxel_down_sample(voxel_size=0.01)
print(point_cloud_o3d, downsampled)
o3d.visualization.draw_geometries([downsampled])



# print(points_3D.shape)
# print(points_3D)

