import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.datasets import load_digits, make_blobs
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
from typing import Optional, Union, Tuple
import warnings

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def basic_umap_visualization(
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        random_state: int = 4,
        figsize: Tuple[int, int] = (8, 6),
        title: str = "UMAP",
        save_path: Optional[str] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    UMAP可视化函数

    参数:
        X: 特征数据，形状为 (n_samples, n_features)。如果为None，则使用手写数字数据集
        y: 标签数据，形状为 (n_samples,)。如果为None，则不显示类别颜色
        n_components: 降维后的维度，通常为2或3
        n_neighbors: 邻居数量，控制局部与全局结构的平衡，通常5-50
        min_dist: 点之间的最小距离，控制聚类紧密度（0.0-1.0）
        metric: 距离度量方法
        random_state: 随机种子
        figsize: 图形大小
        title: 图表标题
        save_path: 保存路径，如果提供则保存图片

    返回:
        X_umap: 降维后的数据
        y: 标签数据（如果提供）
    """

    # 如果没有提供数据，使用手写数字数据集
    if X is None:
        print("使用默认数据集：手写数字数据集")
        digits = load_digits()
        X = digits.data
        y = digits.target

    if y is not None:
        print(f"Shape of label: {y.shape}")
        print(f"Cluster: {len(np.unique(y))}")
    else:
        print("未提供标签数据，将以单色显示")

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 自动调整n_neighbors（如果数据点太少）
    n_samples = X.shape[0]
    adjusted_n_neighbors = min(n_neighbors, n_samples - 1)
    if adjusted_n_neighbors < n_neighbors:
        print(f"注意：将邻居数从 {n_neighbors} 调整为 {adjusted_n_neighbors}（样本数限制）")
        n_neighbors = adjusted_n_neighbors

    # 创建UMAP模型
    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=-1
    )

    # 执行UMAP降维
    print("Start UMAP...")
    X_umap = umap_model.fit_transform(X_scaled)
    print("UMAP complete")

    # 可视化
    if n_components == 2:
        _plot_2d_umap(X_umap, y, figsize, title, save_path)
    elif n_components == 3:
        _plot_3d_umap(X_umap, y, figsize, title, save_path)
    else:
        print(f" n_components={n_components}, Cannot be visualized")

    return X_umap, y


def _plot_2d_umap(X_umap: np.ndarray,
                  y: Optional[np.ndarray],
                  figsize: Tuple[int, int],
                  title: str,
                  save_path: Optional[str]):
    """绘制2D UMAP图"""
    plt.figure(figsize=figsize)

    if y is not None:
        unique_labels = np.unique(y)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            plt.scatter(
                X_umap[y == label, 0],
                X_umap[y == label, 1],
                c=[colors[i]],
                label=str(label),
                alpha=0.7,
                s=30,
                edgecolors='w',
                linewidth=0.5
            )

        plt.legend(title='类别', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(
            X_umap[:, 0],
            X_umap[:, 1],
            c='blue',
            alpha=0.7,
            s=30,
            edgecolors='w',
            linewidth=0.5
        )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")

    plt.show()


def _plot_3d_umap(X_umap: np.ndarray,
                  y: Optional[np.ndarray],
                  figsize: Tuple[int, int],
                  title: str,
                  save_path: Optional[str]):
    """绘制3D UMAP图"""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    if y is not None:
        unique_labels = np.unique(y)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            ax.scatter(
                X_umap[y == label, 0],
                X_umap[y == label, 1],
                X_umap[y == label, 2],
                c=[colors[i]],
                label=str(label),
                alpha=0.7,
                s=20
            )

        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    else:
        ax.scatter(
            X_umap[:, 0],
            X_umap[:, 1],
            X_umap[:, 2],
            c='blue',
            alpha=0.7,
            s=20
        )

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_zlabel('Feature 3', fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")

    plt.tight_layout()
    plt.show()

