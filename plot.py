import matplotlib.pyplot as plt

x = [4, 8, 16, 32, 64, 128, 256, 512]

d3feat = [3.5, 5, 7, 10, 13.8, 18.5, 25, 33]
d3feat_base = [2.5, 3.5, 4, 5.5, 7, 9, 12.5, 16.8]
usip = [15, 17.8, 18.5, 18.9, 18.8, 19.5, 20.5, 20.3]
random = [0.2, 0.3, 2, 0.5, 1, 4, 7.5, 14]
sift = [0.1, 0.5, 0.7, 1.2, 2.5, 4.5, 6.8, 10.5]
harris = [0.5, 0.7, 2, 3.2, 6, 11.5, 14.8, 17]
iss = [0.2, 0.8, 2.2, 3.8, 7, 10.5, 11.2, 15]

attention_raw = [
    0.14594594594594595,
    0.19774774774774775,
    0.24448198198198198,
    0.30045045045045043,
    0.3583051801801802,
    0.4166807432432432,
    0.45783361486486485,
    0.49603392454954953
]
attention = [v * 100 for v in attention_raw]

colors = {
    "D3Feat": "#ff7f0e",        # 橘
    "D3Feat(base)": "#1f77b4",  # 藍
    "USIP": "#2ca02c",          # 綠
    "Random": "#7f7f7f",        # 灰
    "SIFT-3D": "#d62728",       # 紅
    "Harris-3D": "#17becf",     # 青
    "ISS": "#9467bd" ,          # 紫
    "D3Feat(attn)": "#bcbd22"   # 金
}


plt.figure(figsize=(5,5), dpi=300)

plt.plot(x, attention, label='D3Feat(attn)', color=colors["D3Feat(attn)"])
plt.plot(x, d3feat, label='D3Feat', color=colors["D3Feat"])
plt.plot(x, d3feat_base, label='D3Feat(base)', color=colors["D3Feat(base)"])
plt.plot(x, usip, label='USIP', color=colors["USIP"])
plt.plot(x, random, label='Random', color=colors["Random"])
plt.plot(x, sift, label='SIFT-3D', color=colors["SIFT-3D"])
plt.plot(x, harris, label='Harris-3D', color=colors["Harris-3D"])
plt.plot(x, iss, label='ISS', color=colors["ISS"])


plt.xscale('log', basex=2)
plt.xticks(x, x) 
plt.xlim(4, 512)
plt.xlim(4, 512)
plt.margins(x=0)
plt.ylim(0, None)
plt.xlabel('# Keypoints on KITTI')
plt.ylabel('Repeatability')

plt.legend()
plt.grid(True)

plt.savefig("result.png", bbox_inches='tight')
plt.show()

# plt.figure(figsize=(6,4), dpi=300)
# plt.savefig("kitti_plot.pdf", bbox_inches='tight')