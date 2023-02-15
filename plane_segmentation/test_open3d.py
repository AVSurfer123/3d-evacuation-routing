import numpy as np
import open3d as o3d

def view_pc():

    print("Load a ply point cloud, print it, and render it")
    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])

def plane_seg():
    pcd_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(pcd_point_cloud.path)

    outliers = o3d.geometry.PointCloud(pcd)
    print("Total points:", len(outliers.points))

    clouds = []

    while True:
        plane_model, inliers = outliers.segment_plane(distance_threshold=0.02,
                                                ransac_n=3,
                                                num_iterations=1000)

        if len(inliers) < 7000:
            break

        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0, Num inliers: {len(inliers)}")

        inlier_cloud = outliers.select_by_index(inliers)
        inlier_cloud.paint_uniform_color(np.random.uniform(size=3))
        clouds.append(inlier_cloud)

        outliers = outliers.select_by_index(inliers, invert=True)

        o3d.visualization.draw_geometries(clouds,
                                zoom=0.8,
                                front=[-0.4999, -0.1659, -0.8499],
                                lookat=[2.1813, 2.0619, 2.0999],
                                up=[0.1204, -0.9852, 0.1215])



    print(f"Found {len(clouds)} planes in scene")

   

if __name__ == '__main__':
    plane_seg()
