import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

RESOLUTION = .1524 # 6 inches in meters

def view_pc():
    print("Load a ply point cloud, print it, and render it")
    # ply_point_cloud = o3d.data.PLYPointCloud()
    # path = ply_point_cloud.path
    # path = '/storage/pointclouds/ptcloud_0.1grid_colored.ply'
    path = '/storage/pointclouds/complete_color_pointcloud.ply'
    # path = "/storage/pointclouds/downsampled_color_pointcloud.ply"
    pcd = o3d.io.read_point_cloud(path)
    bbox = o3d.geometry.AxisAlignedBoundingBox([-float('inf'), -float('inf'), -.1], [float('inf'), float('inf'), 3.1])
    pcd = pcd.crop(bbox)
    # o3d.io.write_point_cloud(path, pcd)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # print(pcd.normals)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, mesh_frame])

def plane_seg():
    # pcd_point_cloud = o3d.data.PLYPointCloud()
    # pcd = o3d.io.read_point_cloud(pcd_point_cloud.path)
    # path = '/storage/pointclouds/ptcloud_0.1grid_colored.ply'
    path = '/storage/pointclouds/complete_color_pointcloud.ply'
    # path = '/storage/pointclouds/downsampled_color_pointcloud.ply'
    pcd = o3d.io.read_point_cloud(path)
    print(pcd)
    pcd = pcd.voxel_down_sample(voxel_size=0.025)
    
    outliers = o3d.geometry.PointCloud(pcd)
    print("Total points:", len(outliers.points))

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([outliers, mesh_frame])

    bbox = o3d.geometry.AxisAlignedBoundingBox([-float('inf'), -float('inf'), -.1], [float('inf'), float('inf'), .1])
    outliers = outliers.crop(bbox)
    print("Ground points:", len(outliers.points))

    o3d.visualization.draw_geometries([outliers, mesh_frame])

    clouds = []
    planes = []

    while True:
        plane_model, inliers = outliers.segment_plane(distance_threshold=0.025,
                                                ransac_n=10,
                                                num_iterations=1000)

        if len(inliers) < 10000:
            break

        [a, b, c, d] = plane_model
        planes.append(plane_model)
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0, Num inliers: {len(inliers)}")

        inlier_cloud = outliers.select_by_index(inliers)
        inlier_cloud.paint_uniform_color(np.random.uniform(size=3))
        clouds.append(inlier_cloud)

        outliers = outliers.select_by_index(inliers, invert=True)

    print(f"Found {len(clouds)} planes in scene")

    o3d.visualization.draw_geometries(clouds + [mesh_frame])

    ground = None
    ground_cloud = None
    points = 0
    for i, plane in enumerate(planes):
        z_dir = np.abs(plane[2]) / np.linalg.norm(plane[:3])
        if z_dir > .9 and len(clouds[i].points) > points:
            points = len(clouds[i].points)
            ground = plane
            ground_cloud = clouds[i]
    
    print("Ground plane:", ground)

    ground_cloud = o3d.geometry.PointCloud()
    for c in clouds:
        ground_cloud += c
    print("Ground size:", ground_cloud)
    
    ground_points = np.asarray(ground_cloud.points)
    min_bound = np.min(ground_points, axis=0)
    max_bound = np.max(ground_points, axis=0)
    print("Bounds:")
    print(min_bound, max_bound)

    ground_height = np.mean(ground_points[:, 2])
    print("Mean ground height:", ground_height)

    num_cells = ((max_bound - min_bound) / RESOLUTION).astype(int) + 1 # Add 1 to include upper limit
    print("Num cells:", num_cells)

    grid = np.ones(num_cells[:2], dtype=np.bool8) # Default is everything is an obstacle
    grid_min = min_bound[:2]
    grid_max = grid_min + RESOLUTION * np.array(grid.shape)
    print(grid_min, grid_max)

    ground_eps = (max_bound - min_bound)[2]
    print("Ground eps:", ground_eps)

    dists = pcd.compute_point_cloud_distance(ground_cloud)
    ind = np.where(np.asarray(dists) > 0.001)[0]
    pcd_obstacles = pcd.select_by_index(ind)

    o3d.visualization.draw_geometries([ground_cloud, pcd_obstacles, mesh_frame])

    total_points = 0

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            cell_min = min_bound + [i * RESOLUTION, j * RESOLUTION, -.1]
            cell_max = min_bound + [(i+1) * RESOLUTION, (j+1) * RESOLUTION, 2]
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

    print("Totals:")
    print(total_points)

    print(f"Found {np.count_nonzero(grid)} blocked cells")
    plt.imshow(grid.T)
    plt.gca().invert_yaxis()
    plt.savefig("obstacle_grid.png")
    plt.show()
    np.save("obstacle_grid.npy", grid)


if __name__ == '__main__':
    plane_seg()
    # view_pc()
