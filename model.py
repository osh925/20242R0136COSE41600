import open3d as o3d
import numpy as np
import os
import time


voxel_size = 0.2  
eps = 0.3  
min_points_in_cluster = 5   
max_points_in_cluster = 40  
min_z_value = -1.5         
max_z_value = 2.5          
min_height = 0.5            
max_height = 2.0            
max_distance = 30.0        

def process_pcd_file(file_path):
   
    original_pcd = o3d.io.read_point_cloud(file_path)

    downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

    cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    ror_pcd = downsample_pcd.select_by_index(ind)

    plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                                 ransac_n=3,
                                                 num_iterations=2000)

    final_point = ror_pcd.select_by_index(inliers, invert=True)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(final_point.cluster_dbscan(eps=eps, min_points=10, print_progress=False))

    colors = np.zeros((len(labels), 3)) 
    colors[labels >= 0] = [0, 0, 1] 
    final_point.colors = o3d.utility.Vector3dVector(colors)

    bboxes = []
    for i in range(labels.max() + 1):
        cluster_indices = np.where(labels == i)[0]
        if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
            cluster_pcd = final_point.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points)
            z_values = points[:, 2]
            z_min = z_values.min()
            z_max = z_values.max()
            if min_z_value <= z_min and z_max <= max_z_value:
                height_diff = z_max - z_min
                if min_height <= height_diff <= max_height:
                    distances = np.linalg.norm(points, axis=1)
                    if distances.max() <= max_distance:
                        bbox = cluster_pcd.get_axis_aligned_bounding_box()
                        bbox.color = (1, 0, 0) 
                        bboxes.append(bbox)

    return final_point, bboxes

def visualize_pcd_sequence(folder_path):
    file_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pcd')])
    if not file_paths:
        print("No PCD files found in the specified folder.")
        return

    print(f"Found {len(file_paths)} PCD files.")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PCD Sequence Viewer")

    for i, file_path in enumerate(file_paths):
        print(f"Processing {file_path} ({i + 1}/{len(file_paths)})")

        final_point, bboxes = process_pcd_file(file_path)

        vis.clear_geometries()
        vis.add_geometry(final_point)
        for bbox in bboxes:
            vis.add_geometry(bbox)

        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.3)

    vis.destroy_window()
# testcode
# folder_path = "data/01/pcd"
# visualize_pcd_sequence(folder_path)
