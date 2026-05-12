import os
import numpy as np
import open3d as o3d


class OxfordDataset:

    def __init__(self, root, split='test'):

        self.root = root

        self.pointcloud_dir = os.path.join(root, 'pointcloud')
        self.transform_dir = os.path.join(root, 'transforms')

        pair_file = os.path.join(root, 'pairs', 'oxford_pairs.txt')

        self.pairs = []

        with open(pair_file, 'r') as f:
            for line in f:
                src, tgt, tf = line.strip().split()

                self.pairs.append({
                    'src': src,
                    'tgt': tgt,
                    'tf': tf
                })

        print("Oxford Dataset Loaded")
        print("Total pairs:", len(self.pairs))

    def __len__(self):
        return len(self.pairs)

    def load_pointcloud(self, filename):

        path = os.path.join(self.pointcloud_dir, filename)

        pcd = o3d.io.read_point_cloud(path)

        points = np.asarray(pcd.points).astype(np.float32)

        return points

    def load_transform(self, filename):

        path = os.path.join(self.transform_dir, filename)

        T = np.loadtxt(path).astype(np.float32)

        return T

    def __getitem__(self, idx):

        pair = self.pairs[idx]

        src_points = self.load_pointcloud(pair['src'])
        tgt_points = self.load_pointcloud(pair['tgt'])

        transform = self.load_transform(pair['tf'])

        return {
            'src_points': src_points,
            'tgt_points': tgt_points,
            'transform': transform,
            'src_name': pair['src'],
            'tgt_name': pair['tgt']
        }

