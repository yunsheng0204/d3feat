import numpy as np
from datasets.Oxford import OxfordDataset

root = "/home/code-server/D3Feat/data/oxford"

dataset = OxfordDataset(root=root)

sample = dataset[0]

src = sample["src_points"]
tgt = sample["tgt_points"]
T = sample["transform"]

print("src shape:", src.shape)
print("tgt shape:", tgt.shape)
print("T:")
print(T)

src_h = np.hstack([src, np.ones((src.shape[0], 1))])
src_transformed = (T @ src_h.T).T[:, :3]

print("Transformed src shape:", src_transformed.shape)

print("src min/max:", src.min(axis=0), src.max(axis=0))
print("tgt min/max:", tgt.min(axis=0), tgt.max(axis=0))
print("src_transformed min/max:", src_transformed.min(axis=0), src_transformed.max(axis=0))

print("OK: Oxford transform test finished.")

