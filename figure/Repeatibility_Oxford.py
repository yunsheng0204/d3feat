import matplotlib.pyplot as plt

# Top-K settings
x = [4, 8, 16]

# Oxford repeatability results (%)
d3feat = [39.58, 43.75, 47.92]
roitr = [32.29, 43.88, 51.56]
pare = [33.33, 39.84, 44.14]
gakey = [74.22, 83.98, 91.28]

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
plt.xlim(4, 16)
plt.margins(x=0)
plt.ylim(0, 100)

plt.xlabel('Top-K Keypoints on Oxford')
plt.ylabel('Repeatability (%)')

plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.savefig('Repeatibility_Oxford.png', bbox_inches='tight')
plt.savefig('Repeatibility_Oxford.pdf', bbox_inches='tight')
plt.show()