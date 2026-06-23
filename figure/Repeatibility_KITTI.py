import matplotlib.pyplot as plt

# Top-K settings
x = [4, 8, 16, 32, 64, 128, 256, 512]

# KITTI repeatability results (%)
d3feat = [0.86, 1.42, 1.84, 2.89, 4.02, 5.82, 8.60, 13.12]
roitr = [0.90, 1.51, 2.44, 3.62, 5.76, 8.70, 13.24, 19.95]
pare = [1.13, 1.73, 3.03, 4.05, 5.81, 8.26, 12.02, 17.23]
gakey = [6.75, 8.85, 10.44, 14.06, 18.62, 23.26, 29.23, 37.55]

colors = {
    "D3Feat": "#1f77b4",
    "RoITr": "#ff7f0e",
    "PARE": "#2ca02c",
    "GA-Key(Ours)": "#d62728",
}

plt.figure(figsize=(5, 5), dpi=300)

plt.plot(x, d3feat, marker='o', linewidth=2, label='D3Feat [1]', color=colors["D3Feat"])
plt.plot(x, roitr, marker='s', linewidth=2, label='RoITr [2]', color=colors["RoITr"])
plt.plot(x, pare, marker='^', linewidth=2, label='PARE [3]', color=colors["PARE"])
plt.plot(x, gakey, marker='D', linewidth=2.5, label='GA-Key(Ours)', color=colors["GA-Key(Ours)"])

plt.xscale('log', base=2)
plt.xticks(x, x)
plt.xlim(4, 512)
plt.margins(x=0)
plt.ylim(0, 40)

plt.xlabel('Top-K Keypoints on KITTI')
plt.ylabel('Repeatability (%)')

plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.savefig('Repeatibility_KITTI.png', bbox_inches='tight')
plt.savefig('Repeatibility_KITTI.pdf', bbox_inches='tight')
plt.show()
