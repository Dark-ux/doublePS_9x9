import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

data = np.array(
    [
        [0.128182228, 0.146588097, 0.165885503, 0.271393921, 0.039573064, 0.22878428],
        [0.157186966, 0.002224782, 0.009322545, 0.024369073, 0.704434979, 0.035107608],
        [0.16408516, 0.137467549, 0.161906114, 0.169528924, 0.03745238, 0.323625207],
        [0.335162078, 0.091791547, 0.109094537, 0.176724453, 0.212616208, 0.140088534],
        [0.105293152, 0.524459027, 0.110936063, 0.183060087, 0.004111545, 0.131235154],
        [0.110090417, 0.097468997, 0.442855238, 0.174923542, 0.001811823, 0.141159217],
    ]
)

white_to_blue = LinearSegmentedColormap.from_list("white_blue", ["#FFFFFF", "#1f77b4"])

plt.figure(figsize=(5.8, 5.8))
nrows, ncols = data.shape
x = np.arange(ncols + 1)
y = np.arange(nrows + 1)

pc = plt.pcolormesh(x, y, data, cmap=white_to_blue, edgecolors="black", linewidth=0.5, shading="flat")  # 小格子黑边框
ax = plt.gca()
ax.invert_yaxis()  # 与 imshow 视觉一致（行号从上到下）

# 红色 3×3 大框。pcolormesh 的边界坐标用整数网格：从 (0,0) 到 (3,3)
rect = Rectangle((0, 0), 3, 3, fill=False, edgecolor="red", linewidth=2.0)
ax.add_patch(rect)

cbar = plt.colorbar(pc, fraction=0.046, pad=0.04, label="Value")
plt.title("6x6 Heatmap")
# 可选：给格子中心加标签刻度
plt.xticks(np.arange(0.5, ncols + 0.5), range(1, ncols + 1))
plt.yticks(np.arange(0.5, nrows + 0.5), range(1, nrows + 1))
ax.set_aspect("equal")
plt.tight_layout()
plt.show()
