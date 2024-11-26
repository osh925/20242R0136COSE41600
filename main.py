from model import visualize_pcd_sequence

folders = ["01", "02", "03", "04", "05", "06", "07"]

for folder in folders:
    folder_path = f"data/{folder}/pcd"
    visualize_pcd_sequence(folder_path)
