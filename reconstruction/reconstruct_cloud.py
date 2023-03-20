import json
import os
import imageio as iio
import numpy as np
import matplotlib.pyplot as plt
import time
from pprint import pprint
import open3d as o3d


DATA_DIR = '/storage/pointclouds/area_3/data'
POSE_DIR = os.path.join(DATA_DIR, 'pose')
DEPTH_DIR = os.path.join(DATA_DIR, 'depth')
RGB_DIR = os.path.join(DATA_DIR, 'rgb')

pose = sorted(os.listdir(POSE_DIR))
depth = sorted(os.listdir(DEPTH_DIR))
rgb = sorted(os.listdir(RGB_DIR))

NUM_IMAGES = len(pose)

MAX_DEPTH = 20 # meters

def reconstruct(start, end):
    start_time = time.time()
    combined_cloud = o3d.geometry.PointCloud()
    for i in range(start, end):
        print(f"Img {i} name: {depth[i]}")

        # load pose json
        with open(os.path.join(POSE_DIR, pose[i])) as f:
            data_pose = json.load(f)

        # camera parameters
        K = np.array(data_pose["camera_k_matrix"])

        # extrinsic parameters
        RT = np.array(data_pose["camera_rt_matrix"])
        R = RT[:3, :3]
        T = RT[:, 3]

        # load depth png
        img_depth = iio.imread(os.path.join(DEPTH_DIR, depth[i]))/512.0 #in meters
        # plt.hist(img_depth)
        # plt.show()

        # Create lambda x in vectorized format
        xs = np.arange(img_depth.shape[0])
        ys = np.arange(img_depth.shape[1])
        xs, ys = np.meshgrid(xs, ys, sparse=True)
        pixels = np.stack([img_depth * xs, img_depth * ys, img_depth], axis=2)
        flattened = np.reshape(pixels, (-1, 3))

        # Remove far away points as they are outliers
        mask = flattened[:, 2] < MAX_DEPTH
        filtered = flattened[mask]
        # print("Filtered shape", filtered.shape)
        # print("Max depth:", filtered[:, 2].max())
        
        # compute 3D points in world frame
        # lambda x = K(RX + T)
        points_world = ((filtered @ np.linalg.inv(K).T) - T) @ R
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_world)
        point_cloud.colors = o3d.utility.Vector3dVector()
        downsampled = point_cloud.voxel_down_sample(voxel_size=0.01)
        
        # Project 3D points back into image plane to get color. Do this after downsampling to save memory
        sampled_points = (np.asarray(downsampled.points) @ R.T + T) @ K.T
        sampled_pixels = sampled_points[:, :2] / sampled_points[:, 2:]
        sampled_pixels = sampled_pixels.astype(int)
        rgb_img = iio.imread(os.path.join(RGB_DIR, rgb[i]))
        colors = rgb_img[sampled_pixels[:, 1], sampled_pixels[:, 0]] / 255.0
        downsampled.colors = o3d.utility.Vector3dVector(colors)

        combined_cloud += downsampled


    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    downsampled = combined_cloud.voxel_down_sample(voxel_size=0.01)
    print(combined_cloud, downsampled)

    o3d.io.write_point_cloud(f"cloud_color_{start}_{end}.ply", downsampled)

    print(time.time() - start_time, " seconds")

    # o3d.visualization.draw_geometries(clouds)
    o3d.visualization.draw_geometries([downsampled, mesh_frame])


if __name__ == '__main__':
    cutoffs = np.arange(1, NUM_IMAGES, 750)
    cutoffs = np.append(cutoffs, NUM_IMAGES)
    for i in range(len(cutoffs) - 1):
        print(f"Reconstructing {cutoffs[i]} to {cutoffs[i+1]}")
        reconstruct(cutoffs[i], cutoffs[i+1])

