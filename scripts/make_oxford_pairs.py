import os
import numpy as np

root = "/home/code-server/D3Feat/data/oxford"

pcd_dir = os.path.join(root, "pointcloud")
pose_dir = os.path.join(root, "poses")
tf_dir = os.path.join(root, "transforms")
pair_dir = os.path.join(root, "pairs")

os.makedirs(tf_dir, exist_ok=True)
os.makedirs(pair_dir, exist_ok=True)

clouds = sorted([f for f in os.listdir(pcd_dir) if f.endswith(".ply")])

pair_file = os.path.join(pair_dir, "oxford_pairs.txt")

with open(pair_file, "w") as f:
    for i in range(len(clouds) - 1):
        src = clouds[i]
        tgt = clouds[i + 1]

        src_id = src.replace("cloud_", "").replace(".ply", "")
        tgt_id = tgt.replace("cloud_", "").replace(".ply", "")

        pose_i = np.loadtxt(os.path.join(pose_dir, "pose_{}.txt".format(src_id)))
        pose_j = np.loadtxt(os.path.join(pose_dir, "pose_{}.txt".format(tgt_id)))

        T_ij = np.linalg.inv(pose_i).dot(pose_j)

        tf_name = "T_{}_{}.txt".format(src_id, tgt_id)
        np.savetxt(os.path.join(tf_dir, tf_name), T_ij)

        f.write("{} {} {}\n".format(src, tgt, tf_name))

print("Done:", pair_file)
print("Total pairs:", len(clouds) - 1)

