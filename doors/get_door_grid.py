import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from egress_generator.plotter import plot

grid_min = np.array([ -1.48131465, -16.25003485])
grid_max = np.array([27.47468535, 9.65796515])
res = .3048
grid_shape = np.array([95, 85])

print(grid_min)
print(grid_min + res * grid_shape)

print(grid_max)
door_points = np.load("door_points.npy")[:, :2]

doors = np.zeros(grid_shape, dtype=bool)
for p in door_points:
    idx = (p - grid_min) / res
    idx = idx.astype(int)
    print(idx)
    doors[idx[0], idx[1]] = True

np.save("door_grid.npy", doors)

grid = np.load("../plane_segmentation/obstacle_grid.npy")

plt.figure()
plt.imshow(grid.T)
plt.gca().invert_yaxis()

plt.figure()
plt.imshow(doors.T)
plt.gca().invert_yaxis()
plt.show()

combined = np.zeros(grid_shape, dtype=int)
for i in range(combined.shape[0]):
    for j in range(combined.shape[1]):
        if grid[i, j]:
            if doors[i, j]:
                combined[i, j] = 2
            else:
                combined[i, j] = 1
plot(combined)
