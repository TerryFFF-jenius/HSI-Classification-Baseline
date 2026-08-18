import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np

# 1. 读取底层张量
tif_path = "./results/own/PaviaU/Label_PRED.tif"
dataset = gdal.Open(tif_path)
if dataset is None:
    raise FileNotFoundError("找不到 TIF 文件，请检查路径。")

img_matrix = dataset.GetRasterBand(1).ReadAsArray()

# 2. 构建伪彩色画布
plt.figure(figsize=(6, 10))
# 使用 'jet' 调色板是高光谱分类论文里最标准的配色方案之一
plt.imshow(img_matrix, cmap='jet')
plt.colorbar(label='Class ID')
plt.title("PaviaU Classification Map")
plt.axis('off') # 去掉周围的坐标轴干扰

# 3. 保存为普通的 PNG 图像
output_path = "./results/own/PaviaU/Label_PRED_visual.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"伪彩色图已成功生成，请在左侧目录双击查看: {output_path}")

plt.show()