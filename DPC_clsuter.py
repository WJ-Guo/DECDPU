import torch
import numpy as np
# from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List

import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox
# import threading
import os


class DensityPeakClustering:
    """
    (Clustering by Fast Search and Find of Density Peaks)
    Matplotlib. GUI
    """

    def __init__(self,
                 dc_percent: float = 2.0,
                 use_gpu: bool = False,
                 verbose: bool = True):
        """
        初始化DPC模型

        参数:
            dc_percent: 截断距离的百分比 (默认2.0%)
            use_gpu: 是否使用GPU加速
            verbose: 是否显示进度信息
        """
        self.dc_percent = dc_percent
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.labels_ = None
        self.centers_ = None
        self.rho_ = None  # 保持为torch.Tensor
        self.delta_ = None  # 保持为torch.Tensor
        self.dc_ = None
        self.nearest_neighbor_ = None
        self.dist_ = None
        self.x_scaled = None
        self.selected_centers = []  # 存储选中的中心点
        self.fig = None
        self.ax = None
        self.rect_selector = None
        self.selected_points = set()  # 用于矩形选择
        self.rho_np = None  # 用于绘图的numpy版本
        self.delta_np = None  # 用于绘图的numpy版本
        self.gamma_np = None  # 用于绘图的numpy版本

    # 保持所有现有的计算方法不变
    def calculate_pairwise_distance(self, X: torch.Tensor) -> torch.Tensor:
        """计算成对距离矩阵（优化内存版本）"""
        n_samples = X.shape[0]
        dist = torch.cdist(X, X, p=2)

        return dist

    def calculate_cutoff_distance(self, dist: torch.Tensor) -> float:
        """计算截断距离dc"""
        n_samples = dist.shape[0]
        distances = dist.flatten()
        distances = distances[distances > 1e-10]

        k = int(self.dc_percent / 100.0 * len(distances))
        if k < 1:
            k = 1

        sorted_dist = torch.sort(distances)[0]
        dc = sorted_dist[k].item()

        if self.verbose:
            print(f" dc_percent: {dc:.4f}")

        return dc

    def calculate_local_density(self, dist: torch.Tensor, dc: float) -> torch.Tensor:
        """计算局部密度rho（高斯核版本）"""
        n_samples = dist.shape[0]

        with torch.no_grad():
            # 高斯核：exp(-(dist/dc)^2)
            gaussian = torch.exp(-(dist / dc) ** 2)

            # 将对角线置为1（自身贡献）
            gaussian.fill_diagonal_(1.0)

            # 对每行求和得到每个点的密度
            rho = torch.sum(gaussian, dim=1)

        return rho

    def calculate_min_distance_to_higher_density(self,
                                                 dist: torch.Tensor,
                                                 rho: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算到更高密度点的最小距离delta"""
        n_samples = dist.shape[0]
        delta = torch.zeros(n_samples, device=dist.device)
        nearest_neighbor = torch.zeros(n_samples, dtype=torch.long, device=dist.device)

        rho_sorted_idx = torch.argsort(rho, descending=True)
        rho_sorted = rho[rho_sorted_idx]

        max_dist = torch.max(dist)
        delta[rho_sorted_idx[0]] = max_dist
        nearest_neighbor[rho_sorted_idx[0]] = rho_sorted_idx[0]

        for i in range(1, n_samples):
            idx = rho_sorted_idx[i]
            higher_density_idx = rho_sorted_idx[:i]

            min_dist = torch.min(dist[idx, higher_density_idx])
            min_idx = higher_density_idx[torch.argmin(dist[idx, higher_density_idx])]

            delta[idx] = min_dist
            nearest_neighbor[idx] = min_idx

        return delta, nearest_neighbor

    def calculate_decision_values(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
        """
        计算决策图所需的值
        返回: (rho, delta, gamma_numpy, 最近邻索引)
        """
        if self.verbose:
            print("Calculating distance matrix...")
        with torch.no_grad():
            dist = self.calculate_pairwise_distance(X)
            self.dist_ = dist

        if self.verbose:
            print("Calculating dc ...")
        self.dc_ = self.calculate_cutoff_distance(dist)

        if self.verbose:
            print("Calculating Rho...")
        rho = self.calculate_local_density(dist, self.dc_)

        if self.verbose:
            print("Calculating Delta...")
        delta, nearest_neighbor = self.calculate_min_distance_to_higher_density(dist, rho)

        # 保存为torch张量
        self.rho_ = rho
        self.delta_ = delta
        self.nearest_neighbor_ = nearest_neighbor

        # 转换为numpy用于绘图
        rho_np = rho.cpu().numpy() if rho.is_cuda else rho.numpy()
        delta_np = delta.cpu().numpy() if delta.is_cuda else delta.numpy()

        # 计算gamma = rho * delta
        gamma_np = rho_np * delta_np

        return rho, delta, gamma_np, nearest_neighbor

    def on_rectangle_select(self, eclick, erelease):
        """矩形选择回调函数"""
        if self.ax is None:
            return

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        if x1 is None or y1 is None or x2 is None or y2 is None:
            return

        # 确定矩形区域
        left, right = min(x1, x2), max(x1, x2)
        bottom, top = min(y1, y2), max(y1, y2)

        # 找出在矩形区域内的点
        for i, (x, y) in enumerate(zip(self.rho_np, self.delta_np)):
            if left <= x <= right and bottom <= y <= top:
                self.selected_points.add(i)

        self.update_plot()

    def on_click(self, event):
        """点击选择/取消选择点"""
        if self.ax is None or event.inaxes != self.ax:
            return

        # 找到最近的点
        distances = np.sqrt((self.rho_np - event.xdata) ** 2 + (self.delta_np - event.ydata) ** 2)
        nearest_idx = np.argmin(distances)

        # 如果距离太远，不选择
        if distances[nearest_idx] > 0.05 * (np.max(self.rho_np) - np.min(self.rho_np)):
            return

        if nearest_idx in self.selected_points:
            self.selected_points.remove(nearest_idx)
        else:
            self.selected_points.add(nearest_idx)

        self.update_plot()

    def update_plot(self):
        """更新绘图"""
        if self.ax is None:
            return

        self.ax.clear()

        # 绘制所有点
        scatter = self.ax.scatter(self.rho_np, self.delta_np, c=self.gamma_np, cmap='viridis', s=30, alpha=0.7)
        # plt.colorbar(scatter, ax=self.ax, label='Gamma')

        # 高亮选中的点
        if self.selected_points:
            selected_indices = list(self.selected_points)
            self.ax.scatter(self.rho_np[selected_indices], self.delta_np[selected_indices],
                            c='red', s=80, marker='o', edgecolors='black', linewidths=2)

            # 显示选中点的信息
            for idx in selected_indices:
                self.ax.annotate(f'{idx}', (self.rho_np[idx], self.delta_np[idx]),
                                 xytext=(5, 5), textcoords='offset points', fontsize=12)

        self.ax.set_xlabel('rho', fontsize=15)
        self.ax.set_ylabel('delta', fontsize=15)
        self.ax.set_title('Decision Graph', fontsize=15)

        self.fig.canvas.draw()

    def create_gui_selector(self, rho_np: np.ndarray, delta_np: np.ndarray, gamma_np: np.ndarray) -> List[int]:
        """创建GUI界面进行中心点选择"""
        self.rho_np = rho_np
        self.delta_np = delta_np
        self.gamma_np = gamma_np
        self.selected_points = set()

        # 创建GUI窗口
        root = tk.Tk()
        root.title("Cluster center for DPC ")
        root.geometry("600x400")

        # 存储结果
        result = []

        def on_submit():
            nonlocal result
            if list(self.selected_points):
                result = list(self.selected_points)
                # 关闭matplotlib图形
                # if self.fig is not None:
                #     plt.close(self.fig)
                root.quit()
                root.destroy()
            else:
                messagebox.showwarning("Warning", "Select at least one point！")

        def on_auto():
            nonlocal result
            auto_centers = self.auto_select_centers(gamma_np)
            result = auto_centers
            # 关闭matplotlib图形
            if self.fig is not None:
                plt.close(self.fig)
            root.quit()
            root.destroy()

        def on_clear():
            self.selected_points.clear()
            if self.fig is not None:
                self.update_plot()

        def on_refresh_plot():
            # 在新线程中创建matplotlib图形
            self.create_matplotlib_window()

        # 创建界面
        frame = ttk.Frame(root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(frame, text="Cluster center for DPC", font=('Arial', 20, 'bold')).grid(row=0, column=0, columnspan=2,
                                                                                  pady=10)

        ttk.Label(frame, text="Instructions:").grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
        instructions = [
            "1. Click the 'Open Decision Diagram' button to display the decision diagram.",
            "2. Drag to select multiple points in the diagram.",
        ]

        for i, instruction in enumerate(instructions):
            ttk.Label(frame, text=instruction).grid(row=2 + i, column=0, columnspan=2, sticky=tk.W, padx=20)

        ttk.Button(frame, text="Open decision graph", command=on_refresh_plot).grid(row=6, column=0, pady=20, padx=10,
                                                                           sticky=tk.W + tk.E)
        ttk.Button(frame, text="Automatic Selection", command=on_auto).grid(row=6, column=1, pady=20, padx=10, sticky=tk.W + tk.E)
        ttk.Button(frame, text="Clear Selection", command=on_clear).grid(row=7, column=0, pady=10, padx=10, sticky=tk.W + tk.E)
        ttk.Button(frame, text="Submit Selection", command=on_submit).grid(row=7, column=1, pady=10, padx=10,
                                                                   sticky=tk.W + tk.E)

        # 配置权重
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        # 初始显示决策图
        on_refresh_plot()

        root.mainloop()
        return result

    def create_matplotlib_window(self):
        """在Tkinter中嵌入matplotlib"""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        # 创建新窗口
        plot_window = tk.Toplevel()
        plot_window.title("Decision Graph")
        plot_window.geometry("1000x800")

        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(5, 4))

        # 绘制散点图
        scatter = self.ax.scatter(self.rho_np, self.delta_np, c=self.gamma_np,
                                  cmap='viridis', s=30, alpha=0.7)
        # plt.colorbar(scatter, ax=self.ax, label='Gamma')

        self.ax.set_xlabel('rho', fontsize=20)
        self.ax.set_ylabel('delta', fontsize=20)
        self.ax.set_title('Decision Graph', fontsize=20)

        # 嵌入到Tkinter
        canvas = FigureCanvasTkAgg(self.fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加工具栏
        toolbar = NavigationToolbar2Tk(canvas, plot_window)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加交互功能
        self.rect_selector = RectangleSelector(
            self.ax, self.on_rectangle_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # 添加关闭按钮
        def on_close():
            plot_window.destroy()
            self.fig = None
            self.ax = None

        close_btn = ttk.Button(plot_window, text="close", command=on_close)
        close_btn.pack(pady=10)


    def plot_decision_graph(self, rho: torch.Tensor, delta: torch.Tensor) -> List[int]:
        """
        使用Matplotlib和GUI绘制决策图
        """
        self.selected_centers = []

        # 转换为numpy用于绘图
        rho_np = rho.cpu().numpy() if rho.is_cuda else rho.numpy()
        delta_np = delta.cpu().numpy() if delta.is_cuda else delta.numpy()
        gamma_np = rho_np * delta_np

        # 使用GUI选择器
        selected_centers = self.create_gui_selector(rho_np, delta_np, gamma_np)

        if not selected_centers:
            print("警告：未选择任何中心，使用自动选择")
            return self.auto_select_centers(gamma_np)

        print(f" {len(selected_centers)} centers has been selected: {selected_centers}")
        return selected_centers

    def fit_predict(self,
                    x: torch.Tensor,
                    manual_select: bool = True,
                    selected_centers: Optional[List[int]] = None) -> np.ndarray:
        """
        执行聚类并返回标签
        """
        if self.verbose:
            print(f"输入数据形状: {x.shape}")

        # 1. 标准化数据
        if self.verbose:
            print("1.标准化数据...")
        scaler = StandardScaler()
        x_np = x.cpu().numpy() if x.is_cuda else x.numpy()
        x_scaled_np = scaler.fit_transform(x_np)
        x_scaled = torch.from_numpy(x_scaled_np).float()
        self.x_scaled = x_scaled

        if self.use_gpu and torch.cuda.is_available():
            x_scaled = x_scaled.cuda()

        # 2. 计算决策值
        if self.verbose:
            print("2.计算决策值...")
        rho, delta, gamma_np, nearest_neighbor = self.calculate_decision_values(x_scaled)

        # 3. 选择聚类中心
        if self.verbose:
            print("3.选取聚类中心...")
        if manual_select and selected_centers is None:
            selected_centers = self.plot_decision_graph(rho, delta)

            if not selected_centers:
                print("警告：未选择任何聚类中心，将使用自动选择")
                selected_centers = self.auto_select_centers(gamma_np)
        elif selected_centers is None:
            # 自动选择
            selected_centers = self.auto_select_centers(gamma_np)

        self.centers_ = torch.tensor(selected_centers, dtype=torch.long)

        if self.verbose:
            print(f"\n已选择 {len(selected_centers)} 个聚类中心:")
            for i, center in enumerate(selected_centers):
                rho_val = rho[center].item() if isinstance(rho, torch.Tensor) else rho[center]
                delta_val = delta[center].item() if isinstance(delta, torch.Tensor) else delta[center]
                print(f"  中心 {i}: 索引={center}, rho={rho_val:.4f}, delta={delta_val:.4f}")

        # 4. 分配标签
        if self.verbose:
            print("4.分配标签...")
        labels = self.assign_labels(selected_centers, nearest_neighbor)

        # 5. 转换为numpy array
        self.labels_ = labels.cpu().numpy() if labels.is_cuda else labels.numpy()

        if self.verbose:
            print(f"\n聚类完成. 共发现 {len(selected_centers)} 个簇.")
            unique_labels = np.unique(self.labels_)
            print(f"标签分布:")
            for label in unique_labels:
                count = np.sum(self.labels_ == label)
                print(f"  簇 {label}: {count} 个样本 ({count / len(self.labels_) * 100:.1f}%)")

        return self.labels_

    def auto_select_centers(self, gamma: np.ndarray) -> List[int]:
        """自动选择聚类中心"""
        gamma_sorted_idx = np.argsort(gamma)[::-1]

        # 使用简单的阈值方法
        gamma_sorted = gamma[gamma_sorted_idx]
        gamma_ratio = gamma_sorted[:-1] / (gamma_sorted[1:] + 1e-10)

        # 找到最大的下降点
        max_drop_idx = np.argmax(gamma_ratio)
        n_clusters = max_drop_idx + 1

        n_clusters = max(2, min(n_clusters, 20))
        selected_centers = gamma_sorted_idx[:n_clusters].tolist()

        print(f"  {n_clusters} centers selected automatically.")
        return selected_centers

    def assign_labels(self,
                      centers: List[int],
                      nearest_neighbor: torch.Tensor) -> torch.Tensor:
        """分配标签"""
        n_samples = len(self.rho_)

        if isinstance(centers, list):
            centers = torch.tensor(centers, dtype=torch.long, device=nearest_neighbor.device)

        # 初始化标签
        labels = torch.full((n_samples,), -1, dtype=torch.long, device=nearest_neighbor.device)

        # 给中心点分配标签
        for i, center in enumerate(centers):
            labels[center] = i

        # 按密度降序分配其他点
        rho_sorted_idx = torch.argsort(self.rho_, descending=True)

        for idx in rho_sorted_idx:
            if labels[idx] == -1:  # 未分配的点
                labels[idx] = labels[nearest_neighbor[idx].item()]

        return labels

    def plot_clustering_result(self,
                               x_2d: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None):
        """
        使用Matplotlib可视化聚类结果
        """
        if self.labels_ is None:
            raise ValueError("Run fit_predict firstly")

        if x_2d is None:

            if self.x_scaled.shape[1] > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                x_2d = pca.fit_transform(self.x_scaled.cpu().numpy())
                explained_var = pca.explained_variance_ratio_.sum() * 100
                title_suffix = f" (PCA, explained variant: {explained_var:.1f}%)"
            else:
                x_2d = self.x_scaled.cpu().numpy()
                title_suffix = ""
        else:
            title_suffix = ""

        n_clusters = len(np.unique(self.labels_))

        # 创建matplotlib图形
        plt.figure(figsize=(10, 8))

        # 绘制聚类点
        scatter = plt.scatter(x_2d[:, 0], x_2d[:, 1], c=self.labels_, cmap='viridis',
                              s=30, alpha=0.7, edgecolors='white', linewidth=0.5)

        # 标记聚类中心
        if self.centers_ is not None:
            centers_np = self.centers_.cpu().numpy() if isinstance(self.centers_, torch.Tensor) else self.centers_
            if len(centers_np) > 0:
                plt.scatter(x_2d[centers_np, 0], x_2d[centers_np, 1],
                            c='red', s=200, marker='*', edgecolors='black', linewidth=2,
                            label='Clustering center')

        plt.colorbar(scatter, label='label')
        plt.xlabel('Feature 1' if x_2d.shape[1] > 1 else 'PC 1')
        plt.ylabel('Feature 2' if x_2d.shape[1] > 1 else 'PC 2')
        plt.title(f'DPC Result - {n_clusters} Clusters {title_suffix}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.tight_layout()
        plt.show()


def load_dataset(data_path, data_name):
    # 构建完整文件路径
    file_path = os.path.join(data_path, f"{data_name}")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        # 使用np.loadtxt加载数据
        data = np.loadtxt(file_path)

        # 检查数据维度
        if data.shape[1] < 2:
            raise ValueError("数据文件至少需要2列（特征+标签）")

        # 分离特征和标签
        features_np = data[:, :-1]  # 前n-1列为特征
        labels_np = data[:, -1]  # 最后一列为标签

        # 转换为tensor
        features = torch.from_numpy(features_np).float()
        labels = torch.from_numpy(labels_np).long()  # 标签通常用整数类型

        return features, labels

    except Exception as e:
        print(f"加载数据集时发生错误: {e}")
        raise  # 重新抛出异常


# 使用示例
def main():
    """示例：使用DPC算法并手动选择聚类中心"""

    # 生成混合高斯分布数据
    x, true_labels = load_dataset("J:\Code\Clustering_algorithm\dataset_Synthetic", "D31.txt")

    # 2. 创建DPC模型
    dpc = DensityPeakClustering(
        dc_percent=2.0,
        use_gpu=False,
        verbose=True
    )

    # 3. 执行聚类（手动选择中心）
    print("\n" + "=" * 60)
    print("开始密度峰值聚类...")
    print("=" * 60)

    labels = dpc.fit_predict(x, manual_select=True)

    # 4. 可视化聚类结果
    print("\n" + "=" * 60)
    print("聚类结果可视化:")
    print("=" * 60)

    dpc.plot_clustering_result()

    # 5. 评估聚类结果
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    ari = adjusted_rand_score(true_labels.numpy(), labels)
    nmi = normalized_mutual_info_score(true_labels.numpy(), labels)

    print(f"\n评估指标:")
    print(f"调整兰德指数 (ARI): {ari:.4f}")
    print(f"归一化互信息 (NMI): {nmi:.4f}")

    return labels, dpc


if __name__ == "__main__":
    labels, model = main()
