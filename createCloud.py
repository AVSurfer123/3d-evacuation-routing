import json
import os
import imageio as iio

pose = sorted(os.listdir('pose'))
depth = sorted(os.listdir('depth'))

for i in range(1, 2):
    # load pose json
    file_pose = open('pose/' + pose[i])
    data_pose = json.load(file_pose)
    print(data_pose)

    # load depth png
    img_depth = iio.imread('depth/' + depth[i])
    print(img_depth.shape)
    # iio.imwrite("g4g.png", img_depth)

 