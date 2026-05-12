from datasets.Oxford import OxfordDataset

dataset = OxfordDataset(
    root='/home/code-server/D3Feat/data/oxford'
)

print("Dataset size:", len(dataset))

sample = dataset[0]

print(sample.keys())

print("Source shape:", sample['src_points'].shape)
print("Target shape:", sample['tgt_points'].shape)

print("Transform:")
print(sample['transform'])

