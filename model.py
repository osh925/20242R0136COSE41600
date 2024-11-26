import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN  # HDBSCAN 라이브러리 추가
from sklearn.preprocessing import MinMaxScaler

# PCD 파일 불러오기 및 다운샘플링
file_path = "data/01/pcd/pcd_000001.pcd"
original_pcd = o3d.io.read_point_cloud(file_path)
voxel_size = 0.2
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

# Radius Outlier Removal (ROR) 적용
cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
ror_pcd = downsample_pcd.select_by_index(ind)

# RANSAC을 사용하여 평면 추정
plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                             ransac_n=3,
                                             num_iterations=2000)
final_point = ror_pcd.select_by_index(inliers, invert=True)

# HDBSCAN 클러스터링 적용
points = np.asarray(final_point.points)
scaler = MinMaxScaler()
points_scaled = scaler.fit_transform(points)  # HDBSCAN에 적합한 스케일링 수행
hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=6, cluster_selection_epsilon=0.3)
labels = hdbscan_model.fit_predict(points_scaled)

# 클러스터별 색상 설정
num_clusters = labels.max() + 1
colors = plt.cm.get_cmap("tab20", num_clusters)(labels)[:, :3]  # 클러스터별 색상
colors[labels == -1] = [0, 0, 0]  # 노이즈는 검정색
final_point.colors = o3d.utility.Vector3dVector(colors)

# 필터링 조건 정의
def filter_clusters_by_conditions(pcd, labels, conditions):
    filtered_bboxes = []
    for i in range(labels.max() + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = pcd.select_by_index(cluster_indices)
        points = np.asarray(cluster_pcd.points)
        
        if len(cluster_indices) < conditions["min_points"] or len(cluster_indices) > conditions["max_points"]:
            continue

        z_values = points[:, 2]
        z_min, z_max = z_values.min(), z_values.max()
        if not (conditions["min_z"] <= z_min <= z_max <= conditions["max_z"]):
            continue

        height_diff = z_max - z_min
        if not (conditions["min_height"] <= height_diff <= conditions["max_height"]):
            continue

        distances = np.linalg.norm(points, axis=1)
        if distances.max() > conditions["max_distance"]:
            continue

        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        bbox.color = (1, 0, 0)  # Bounding box 색상
        filtered_bboxes.append(bbox)

    return filtered_bboxes

# 필터링 조건 설정
filter_conditions = {
    "min_points": 5,
    "max_points": 40,
    "min_z": -1.5,
    "max_z": 2.5,
    "min_height": 0.5,
    "max_height": 2.0,
    "max_distance": 30.0
}

# 클러스터 필터링
bboxes_1234 = filter_clusters_by_conditions(final_point, labels, filter_conditions)

# 시각화 함수
def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Filtered Clusters and Bounding Boxes", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

# 시각화
visualize_with_bounding_boxes(final_point, bboxes_1234, point_size=2.0)
