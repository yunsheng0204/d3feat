import os

root = "/home/code-server/D3Feat/data/oxford/pairs"
pair_file = os.path.join(root, "oxford_pairs.txt")

with open(pair_file, "r") as f:
    lines = f.readlines()

n = len(lines)
train = lines[:int(n * 0.7)]
val = lines[int(n * 0.7):int(n * 0.85)]
test = lines[int(n * 0.85):]

for name, data in [
    ("train_oxford.txt", train),
    ("val_oxford.txt", val),
    ("test_oxford.txt", test),
]:
    with open(os.path.join(root, name), "w") as f:
        f.writelines(data)
    print(name, len(data))
