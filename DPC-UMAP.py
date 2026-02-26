from utils import cluster_acc, Calaculate_rho_KNN, choose_points, assign_points
import torch
import matplotlib.pyplot as plt
from DPC_clsuter import load_dataset, DensityPeakClustering
import numpy as np

from UMAP_draw import basic_umap_visualization
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

######################### 加载数据 ########################################################################
data, true_labels = load_dataset("J:\Code\Clustering_algorithm\dataset_real_world", "Phoneme.txt")
print(f"Shape of data: {data.shape}")
print(f"Type of data: {type(data)}")

###### N-sampling ##
n_UPMA = 3
ini_scope = 7
data_np = data.numpy()
X_UMAP, _ = basic_umap_visualization(data_np, true_labels, n_components=n_UPMA)

## 下采样：降低内存复杂度
distances, indices, rho, kdtree = Calaculate_rho_KNN(X_UMAP, k = 20)
print("KNN complected")
rep_data_idx, rho_sorted_KNN, rho_sorted_idx = choose_points(distances, indices, rho, ini_scope=ini_scope)
print(f"Number of Sampling Points:{rep_data_idx.shape}")
data_np = X_UMAP
rep_data = X_UMAP[rep_data_idx]
rep_data_tens = torch.Tensor(rep_data)

plt.figure(figsize=(10, 8))
plt.scatter(data_np[rep_data_idx, 0], data_np[rep_data_idx, 1],
            alpha=0.5,  # 设置透明度
            s=30,  # 设置点的大小
            c='r',  # 设置点的颜色
            edgecolors='none')  # 去掉边缘线

plt.tight_layout()
plt.show()

## 嵌入改进密度聚类方法替换K-means
dpc = DensityPeakClustering(dc_percent=2.0, use_gpu=False, verbose=False)
y_pred_downsample = dpc.fit_predict(rep_data_tens, manual_select=True)
y_pred = assign_points(data, y_pred_downsample, rep_data_idx, indices, distances, rho_sorted_idx)

unique_labels = np.unique(y_pred)
colormaps = ['tab20']

for cmap_name in colormaps:
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.get_cmap(cmap_name)

    for i, label in enumerate(unique_labels):
        mask = y_pred == label
        if label == -1:
            plt.scatter(X_UMAP[mask, 0], X_UMAP[mask, 1], c='k', label='noisy', alpha=0.7, edgecolors='w', s=30)
        else:
            # 从色谱中获取颜色
            color = cmap(i / max(1, len(unique_labels) - 1))
            plt.scatter(X_UMAP[mask, 0], X_UMAP[mask, 1], c=[color], label=f'Class {label}', alpha=0.7, edgecolors='w',
                        s=50)

## 绘制采样图，聚类中心
plt.scatter(rep_data[dpc.centers_, 0], rep_data[dpc.centers_, 1],
            c='red',  # 统一用红色
            marker='X',  # 用X标记
            s=200,  # 大一点
            label='Cluster Centers',
            edgecolors='k',  # 黑色边框
            linewidth=2,
            zorder=10)  # 确保在最上层
for i, center_idx in enumerate(dpc.centers_):
    x1, x2 = rep_data[center_idx, 0], rep_data[center_idx, 1]
    # 添加文本标注
    plt.text(x1 + 1, x2 + 1, f'{center_idx}',  # 显示原始样本编号
             fontsize=10, fontweight='bold',
             ha='center', va='center',  # 居中对齐
             color='white',  # 白色文字
             bbox=dict(boxstyle='round,pad=0.3',
                       facecolor='red', alpha=0.8))

plt.tight_layout()
plt.show()

print(y_pred)
print(type(y_pred))
print(y_pred.shape)  # 查看属性和方法

nmi_k = nmi_score(true_labels, y_pred)
acc_k = cluster_acc(true_labels, y_pred)
ari_k = ari_score(true_labels, y_pred)
print(':Acc {:.4f}'.format(acc_k),
      ', nmi {:.4f}'.format(nmi_k), ', ari {:.4f}'.format(ari_k))
