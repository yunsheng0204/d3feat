import os
import glob
import numpy as np


def count_bin_points(file_path):
    """
    For KITTI / Oxford style .bin files.
    Usually each point is stored as x, y, z, intensity.
    """
    data = np.fromfile(file_path, dtype=np.float32)

    if data.size % 4 == 0:
        points = data.reshape(-1, 4)
    elif data.size % 3 == 0:
        points = data.reshape(-1, 3)
    else:
        print("[Warning] Unknown bin format:", file_path, "size:", data.size)
        return None

    return points.shape[0]


def count_ply_points(file_path):
    """
    Count number of vertices in ASCII or binary PLY file
    by reading the header.
    """
    with open(file_path, "rb") as f:
        for line in f:
            line = line.decode("utf-8", errors="ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return None


def summarize_dataset(dataset_name, file_list, file_type):
    counts = []

    print("\n==============================")
    print(dataset_name)
    print("==============================")
    print("Number of files:", len(file_list))

    for file_path in file_list:
        if file_type == "bin":
            n = count_bin_points(file_path)
        elif file_type == "ply":
            n = count_ply_points(file_path)
        else:
            n = None

        if n is not None:
            counts.append(n)

    if len(counts) == 0:
        print("No valid point cloud files found.")
        return

    counts = np.array(counts)

    print("Valid files:", len(counts))
    print("Min points:", counts.min())
    print("Max points:", counts.max())
    print("Average points:", counts.mean())
    print("Median points:", np.median(counts))

    print("\nFirst 10 files:")
    for file_path, n in zip(file_list[:10], counts[:10]):
        print(os.path.basename(file_path), ":", n)


if __name__ == "__main__":

    # 修改成你自己的實際資料路徑
    kitti_path = "data/kitti"
    oxford_path = "data/oxford"
    eth_path = "data/ETH"

    # KITTI: usually .bin
    kitti_files = sorted(glob.glob(os.path.join(kitti_path, "**", "*.bin"), recursive=True))

    # Oxford: usually .bin
    oxford_files = sorted(glob.glob(os.path.join(oxford_path, "**", "*.bin"), recursive=True))

    # ETH: usually .ply
    eth_files = sorted(glob.glob(os.path.join(eth_path, "**", "*.ply"), recursive=True))

    summarize_dataset("KITTI", kitti_files, "bin")
    summarize_dataset("Oxford RobotCar", oxford_files, "bin")
    summarize_dataset("ETH", eth_files, "ply")