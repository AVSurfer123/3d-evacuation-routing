import open3d as o3d
import json
import os
import imageio as iio
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from doorPoints import getDoorPoints 

start = time.time()

pose = sorted(os.listdir('pose'))
depth = sorted(os.listdir('depth'))
rgb = sorted(os.listdir('rgb'))

NUM_IMAGES = len(pose)
MAX_DEPTH = 20 # meters

all_points = []
all_door_points = []
for x in range(1, NUM_IMAGES):  
    print(x)
    # load pose json
    file_pose = open('pose/' + pose[x])
    data_pose = json.load(file_pose)

    # camera parameters
    K = np.array(data_pose["camera_k_matrix"])

    # rotation matrix
    RT = np.array(data_pose["camera_rt_matrix"])
    R = RT[0:3,0:3]
    T = RT[:,3]
    camera_location = np.array(data_pose["camera_location"])

    # load depth png
    img_depth = iio.imread('depth/' + depth[x])/512.0 #in meters
    # print(img_depth)

    # create list of pixels on image plane
    xs = np.arange(img_depth.shape[0])
    ys = np.arange(img_depth.shape[1])
    ys, xs = np.meshgrid(ys, xs, sparse=True)
    pixels = np.stack([img_depth * ys, img_depth * xs, img_depth], axis=2)
    flattened_pixels = np.reshape(pixels, (-1, 3))
    
    # remove outlier points with large depth
    mask = flattened_pixels[:, 2] < MAX_DEPTH
    filtered_pixels = flattened_pixels[mask]

    # compute 3D points in 
    points_3D = (np.linalg.inv(K) @ filtered_pixels.T).T
    ones = np.ones((1, points_3D.shape[0]))
    homo = np.array([0, 0, 0, 1])

    # load RBG image
    image = cv2.imread("rgb/" + rgb[x])
    doorPointsInImage = getDoorPoints(image)
    # print("door points: ", doorPointsInImage) 
    if doorPointsInImage.size > 0:
        doorPixels = np.zeros((doorPointsInImage.shape[0], 3))
        for p in range(0,doorPointsInImage.shape[0]-1):
            doorPoint_x = doorPointsInImage[p,0]
            doorPoint_y = doorPointsInImage[p,1]
            pixelDepth = img_depth[doorPoint_y, doorPoint_x] #flipped order to account for x being horizontal
            doorPixels[p,:] = np.array([doorPoint_x * pixelDepth, doorPoint_y * pixelDepth, pixelDepth])
        # print("doorpixels", doorPixels)
        doormask = doorPixels[:,2] < MAX_DEPTH
        filtered_door_pixels = doorPixels[doormask]
        door_points_world = ((filtered_door_pixels @ np.linalg.inv(K).T) - T) @ R
        all_door_points.append(door_points_world)

    # transform
    points_world = ((filtered_pixels @ np.linalg.inv(K).T) - T) @ R
    # print(points_world.shape)
    
    all_points.append(points_world)

all_door_points = np.concatenate(all_door_points, axis=0)
# spheres = []
# for point in all_door_points:
#     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
#     sphere.paint_uniform_color([0.9, 0.1, 0.1])
#     sphere.translate(point)
#     spheres.append(sphere)
# print("spheres",spheres)
# print(all_door_points.shape)
door_point_cloud_o3d = o3d.geometry.PointCloud()
door_point_cloud_o3d.points = o3d.utility.Vector3dVector(all_door_points)
door_point_cloud_o3d.paint_uniform_color([0.9, 0.1, 0.1])

o3d.io.write_point_cloud('door_pointcloud.ply', door_point_cloud_o3d)

all_points = np.concatenate(all_points, axis=0)
# print(all_points.shape)

end = time.time()
print(end - start, " seconds")

# point_cloud_o3d = o3d.geometry.PointCloud()
# point_cloud_o3d.points = o3d.utility.Vector3dVector(all_points)
# downsampled = point_cloud_o3d.voxel_down_sample(voxel_size=0.01)
# downsampled.paint_uniform_color([0.1, 0.1, 0.9])
# print(point_cloud_o3d, downsampled)
# o3d.visualization.draw_geometries([downsampled, door_point_cloud_o3d])




