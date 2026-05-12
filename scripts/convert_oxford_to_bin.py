import os
import numpy as np
import open3d as o3d

root = "/home/code-server/D3Feat/data/oxford"

ply_dir = os.path.join(root, "pointcloud")
bin_dir = os.path.join(root, "bin")

os.makedirs(bin_dir, exist_ok=True)

files = sorted([f for f in os.listdir(ply_dir) if f.endswith(".ply")])

for f in files:

    path = os.path.join(ply_dir, f)

    pcd = o3d.io.read_point_cloud(path)

    points = np.asarray(pcd.points).astype(np.float32)

    reflectance = np.zeros((points.shape[0], 1), dtype=np.float32)

    points4 = np.hstack([points, reflectance])

    out_name = f.replace(".ply", ".bin")

    out_path = os.path.join(bin_dir, out_name)

    points4.astype(np.float32).tofile(out_path)

    print("Saved:", out_path)
