import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

RESOLUTION = 0.025 # meters

def view_pc():
    print("Load a ply point cloud, print it, and render it")
    # ply_point_cloud = o3d.data.PLYPointCloud()
    # path = ply_point_cloud.path
    path = '/storage/pointclouds/ptcloud_0.1grid_colored.ply'
    # path = '/storage/pointclouds/complete_color_pointcloud.ply'
    pcd = o3d.io.read_point_cloud(path)
    # bbox = o3d.geometry.AxisAlignedBoundingBox([-float('inf'), -float('inf'), -.1], [float('inf'), float('inf'), 3.2])
    # pcd = pcd.crop(bbox)
    print(pcd)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, mesh_frame],
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])

def plane_seg():
    # pcd_point_cloud = o3d.data.PLYPointCloud()
    # pcd = o3d.io.read_point_cloud(pcd_point_cloud.path)
    path = '/storage/pointclouds/ptcloud_0.1grid_colored.ply'
    pcd = o3d.io.read_point_cloud(path)

    outliers = o3d.geometry.PointCloud(pcd)
    print("Total points:", len(outliers.points))

    clouds = []
    planes = []

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    while True:
        plane_model, inliers = outliers.segment_plane(distance_threshold=0.015,
                                                ransac_n=10,
                                                num_iterations=1000)

        if len(inliers) < 7000:
            break

        [a, b, c, d] = plane_model
        planes.append(plane_model)
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0, Num inliers: {len(inliers)}")

        inlier_cloud = outliers.select_by_index(inliers)
        inlier_cloud.paint_uniform_color(np.random.uniform(size=3))
        clouds.append(inlier_cloud)

        outliers = outliers.select_by_index(inliers, invert=True)

        o3d.visualization.draw_geometries(clouds + [mesh_frame],
                            zoom=0.8,
                            front=[-0.4999, -0.1659, -0.8499],
                            lookat=[2.1813, 2.0619, 2.0999],
                            up=[0.1204, -0.9852, 0.1215])

    # mesh_sphere = o3d.create_mesh_sphere(radius = 1.0)
    # mesh_sphere.compute_vertex_normals()
    # mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
    

    print(f"Found {len(clouds)} planes in scene")

    ground = None
    ground_cloud = None
    mag = 0
    points = 0
    for i, plane in enumerate(planes):
        z_dir = np.abs(plane[2]) / np.linalg.norm(plane[:3])
        if z_dir > .9 and len(clouds[i].points) > points:
            points = len(clouds[i].points)
            ground = plane
            ground_cloud = clouds[i]
    
    print("Ground plane:", ground)
    
    ground_points = np.asarray(ground_cloud.points)
    min_bound = np.min(ground_points, axis=0)
    max_bound = np.max(ground_points, axis=0)
    print("Bounds:")
    print(min_bound, max_bound)

    ground_height = np.mean(ground_points[:, 2])
    print("Mean ground height:", ground_height)

    # num_cells = 1000
    # res = (max_bound - min_bound) / num_cells
    res = .025
    num_cells = ((max_bound - min_bound) / res).astype(int) + 1 # Add 1 to include upper limit
    print("Num cells:", num_cells)

    grid = np.ones((num_cells[0], num_cells[2]), dtype=np.bool8) # Default is everything is an obstacle
    grid_min = np.array([min_bound[0], min_bound[2]])
    grid_max = grid_min + res * np.array(grid.shape)
    print(grid_min, grid_max)

    # labels = np.array(pcd.cluster_dbscan(eps=0.01, min_points=5, print_progress=True))
    # max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd],
    #                                 zoom=0.455,
    #                                 front=[-0.4999, -0.1659, -0.8499],
    #                                 lookat=[2.1813, 2.0619, 2.0999],
    #                                 up=[0.1204, -0.9852, 0.1215])

    ground_eps = (max_bound - min_bound)[1]
    print("Ground eps:", ground_eps)

    dists = pcd.compute_point_cloud_distance(ground_cloud)
    ind = np.where(np.asarray(dists) > 0.001)[0]
    pcd_obstacles = pcd.select_by_index(ind)

    total_points = 0

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            cell_min = min_bound + [i * res, j * res, -.1]
            cell_max = min_bound + [(i+1) * res, (j+1) * res, 3]
            bbox = o3d.geometry.AxisAlignedBoundingBox(cell_min[:, None], cell_max[:, None])
            # print(bbox)

            cell_ground = ground_cloud.crop(bbox)
            if len(cell_ground.points) == 0:
                continue

            grid[i, j] = 0 # Ground exists here

            cell_cloud = pcd_obstacles.crop(bbox)
            total_points += len(cell_cloud.points)
            if len(cell_cloud.points) != 0:
                middle = np.median(cell_cloud.points, axis=0)
                if (abs(ground_height - middle[2]) > ground_eps):
                    # Obstacle is here
                    grid[i, j] = 1
                    # print(f"Obstacle at ({i}, {j})")

    print("Totals:")
    print(total_points)
    print(len(pcd_obstacles.points))

    print(f"Found {np.count_nonzero(grid)} obstacles")
    plt.ion()
    plt.imshow(grid.T)
    plt.gca().invert_yaxis()
    plt.savefig("obstacle_grid.png")
    plt.show()
    np.save("obstacle_grid.npy", grid)

    o3d.visualization.draw_geometries([ground_cloud, pcd_obstacles, mesh_frame],
                                    front=[ -0.13628523412698096, -0.99051719584896003, -0.017378713602195017 ],
                                    lookat=[ 2.6172, 2.0474999999999999, 1.532 ],
                                    up=[ -0.024578094295810211, -0.014156332132812324, 0.99959767683870282 ],
                                    zoom=0.78120000000000034)
                                #   zoom=0.3412,
                                #   front=[0.4257, -0.2125, -0.8795],
                                #   lookat=[2.6172, 2.0475, 1.532],
                                #   up=[-0.0694, -0.9768, 0.2024])

if __name__ == '__main__':
    # plane_seg()
    view_pc()
