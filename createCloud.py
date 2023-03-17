import json
import os
import imageio as iio
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

pose = sorted(os.listdir('pose'))
depth = sorted(os.listdir('depth'))
all_points = np.zeros((100,1080*1080,4)) #change this 3
for x in range(1, 100): #change this 3
    # load pose json
    file_pose = open('pose/' + pose[x])
    data_pose = json.load(file_pose)
    # print(type(data_pose))
    # print(depth[x])
    # camera parameters
    K = np.array(data_pose["camera_k_matrix"])
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    # rotation matrix
    RT = np.array(data_pose["camera_rt_matrix"])
    R = RT[0:3,0:3]
    T = RT[:,3]
    # print("RT", RT)
    # print(R)
    # print("T", T.shape)
    # load depth png
    img_depth = iio.imread('depth/' + depth[x])/512.0 #in meters
    # print(img_depth)

    # create list of pixels on image plane
    pixels = np.zeros((img_depth.shape[0]*img_depth.shape[1], 3))
    count = 0
    for i in range(0,img_depth.shape[0]):
        for j in range(0, img_depth.shape[1]):
            pixels[count] = [i * img_depth[i,j], j * img_depth[i,j], img_depth[i,j]]
            count += 1
    
    # compute 3D points in world system
    # print("RT", RT) # RT is 3x4 but need to make it a square matrix to invert
    points_3D = (np.linalg.inv(K) @ pixels.T).T
    ones = np.ones((1, points_3D.shape[0]))
    homo = np.array([0, 0, 0, 1])
    # print(points_3D.shape)
    points_3D_world = (np.linalg.inv(np.vstack((RT, homo))) @ np.vstack((points_3D.T, ones))).T
    # print(points_3D_world.shape)
    
    all_points[x] = points_3D_world

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')


# ax.scatter(pixels[:,0], pixels[:,1], pixels[:,2])
# ax.scatter(points_3D[:,0], points_3D[:,1], points_3D[:,2], s=.1, alpha=0.5)
# ax.scatter(points_3D_world[:,0], points_3D_world[:,1], points_3D_world[:,2], s=.1, alpha=0.5)


# print(all_points.shape)
plt.scatter(all_points[:,:,0], all_points[:,:,1], all_points[:,:,2], color="g")
# plt.scatter(all_points[2,:,0], all_points[2,:,1], all_points[2,:,2], color="b")

plt.show()


# print(points_3D.shape)
# print(points_3D)

    

end = time.time()
print(end - start, " seconds")