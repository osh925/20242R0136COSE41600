import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import time

# 초기 설정
sequence_dir = "data/01/pcd"
voxel_size = 0.2
eps = 0.3
min_samples = 10
movement_threshold = 0.5  # 이동 거리 임계값 (m)
time_window = 3  # 이동 분석을 위한 프레임 윈도우 크기

# 이전 프레임의 클러스터 정보 저장
previous_clusters = {}
global_cluster_map = {}

# Voxel Downsampling 및 필터링
def preprocess_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    downsample_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    filtered_pcd = downsample_pcd.select_by_index(ind)
    plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.1,
                                                      ransac_n=3,
                                                      num_iterations=2000)
    return filtered_pcd.select_by_index(inliers, invert=True)

# DBSCAN 클러스터링
def cluster_pcd(pcd):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False))
    return labels


# 클러스터 매칭
def match_clusters(previous_clusters, current_clusters, previous_points, current_points):
    """현재 프레임과 이전 프레임 간 클러스터 매칭."""
    matched_clusters = {}

    if previous_points is None or len(previous_points) == 0:
        print("No previous points available for matching.")
        return matched_clusters

    if current_points is None or len(current_points) == 0:
        print("No current points available for matching.")
        return matched_clusters

    # Nearest Neighbors 모델 생성
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(previous_points)

    for cluster_id, points in current_clusters.items():
        current_center = np.mean(points, axis=0)  # 현재 클러스터 중심

        # Nearest Neighbors 결과 확인
        distances, indices = nbrs.kneighbors([current_center])
        if distances[0][0] > movement_threshold or len(indices[0]) == 0:
            print(f"No valid neighbor found for cluster {cluster_id}")
            continue  # 이웃이 없으면 건너뜀

        # 유효한 매칭만 추가
        try:
            closest_prev_id = list(previous_clusters.keys())[indices[0][0]]
            matched_clusters[cluster_id] = closest_prev_id
        except IndexError:
            print(f"IndexError for cluster {cluster_id}. Skipping...")
            continue

    return matched_clusters



# 클러스터링 결과를 정적 및 동적 클러스터로 분류
def classify_clusters(current_clusters, matched_clusters, current_labels, current_points):
    """동적 클러스터와 정적 클러스터를 분류."""
    dynamic_clusters = []
    bounding_boxes = []

    for cluster_id, points in current_clusters.items():
        if cluster_id in matched_clusters:
            prev_cluster_id = matched_clusters[cluster_id]
            movement = np.linalg.norm(np.mean(points, axis=0) - np.mean(previous_clusters[prev_cluster_id], axis=0))

            # 이동 조건 충족 시 동적 클러스터로 분류
            if movement >= movement_threshold:
                dynamic_clusters.append(cluster_id)

                # Bounding Box 생성
                cluster_pcd = o3d.geometry.PointCloud()
                cluster_pcd.points = o3d.utility.Vector3dVector(points)
                bbox = cluster_pcd.get_axis_aligned_bounding_box()
                bbox.color = (1, 0, 0)  # 빨간색
                bounding_boxes.append(bbox)

    return dynamic_clusters, bounding_boxes

# 시각화
def visualize_clusters_with_bboxes(pcd, labels, bounding_boxes):
    colors = np.zeros((len(labels), 3))  # 기본 검정색
    unique_labels = np.unique(labels)

    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue  # 노이즈는 검정색
        if cluster_id in global_cluster_map:
            cluster_color = np.random.rand(3)  # 랜덤 색상
            colors[labels == cluster_id] = cluster_color

    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Dynamic Clusters")
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.3)
    vis.destroy_window()

# 메인 처리 루프
def process_pcd_sequence(sequence_dir):
    global previous_clusters, previous_points
    file_paths = sorted([os.path.join(sequence_dir, f) for f in os.listdir(sequence_dir) if f.endswith('.pcd')])

    for i, file_path in enumerate(file_paths):
        print(f"Processing {file_path} ({i + 1}/{len(file_paths)})")

        # 전처리 및 클러스터링
        filtered_pcd = preprocess_pcd(file_path)
        labels = cluster_pcd(filtered_pcd)

        # 현재 프레임 클러스터 정보 추출
        current_clusters = {}
        current_points = np.asarray(filtered_pcd.points)
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue
            current_clusters[cluster_id] = current_points[labels == cluster_id]

        # 클러스터 매칭 및 동적 클러스터 분류
        if i > 0:  # 첫 프레임에서는 매칭을 건너뜀
            matched_clusters = match_clusters(previous_clusters, current_clusters, previous_points, current_points)
        else:
            matched_clusters = {}

        dynamic_clusters, bounding_boxes = classify_clusters(current_clusters, matched_clusters, labels, current_points)

        # 시각화
        visualize_clusters_with_bboxes(filtered_pcd, labels, bounding_boxes)

        # 현재 프레임 데이터를 이전 프레임으로 저장
        previous_clusters = current_clusters
        previous_points = current_points



# 실행
process_pcd_sequence(sequence_dir)
