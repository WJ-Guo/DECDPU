# -*- coding: utf-8 -*-
#
# Copyright © dawnranger.
#
# 2018-05-08 10:15 <dawnranger123@gmail.com>
#
# Distributed under terms of the MIT license.
from __future__ import division, print_function
import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset

def load_mnist(path='./data/mnist.npz'):
    f = np.load(path)

    x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f[
        'x_test'], f['y_test']
    f.close()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test)).astype(np.int32)
    x = x.reshape((x.shape[0], -1)).astype(np.float32)
    x = np.divide(x, 255.)
    print('MNIST samples', x.shape)
    return x, y


class MnistDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_mnist()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))

#######################################################
# Calculate Rho by KNN
#######################################################

def Calaculate_rho_KNN(data, k = 40, leaf_size = 20):
    # 将张量转换为numpy数组
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy() if data.is_cuda else data.numpy()
    else:
        data_np = np.array(data)

    print(f"data_shape:{data_np.shape}")

    # 构建 KDtree
    kdt = KDTree(data_np,leaf_size=leaf_size)
    print("KD-tree finish built")

    # 查询 KNN
    distances, indices = kdt.query(data_np, k = k+1)

    # 排除自身
    distances = distances[:, 1:] # 邻居距离
    indices = indices[:, 1:] # 邻居索引

    rho = 1/distances.mean(1)

    return distances, indices, rho, kdt

#######################################################
# Downsample
#######################################################
def choose_points(distances, indices, rho, ini_scope):
    n_samples = indices.shape[0]
    knn_dim = indices.shape[1]

    rho_mean = rho.mean()
    print(f" Mean value of Rho：{rho_mean}")
    rho_sorted_idx = np.argsort(-rho) # 降序排
    precent_samples = np.ceil(0.05 * n_samples).astype(int)
    rho_precent = rho_sorted_idx[-precent_samples:]

    rho_sorted_knn = np.zeros([n_samples, knn_dim]) # 排列后的knn密度集合
    # 最高密度点先选为代表点
    max_idx = rho_sorted_idx[0]
    # print(max_idx)
    rep_data = [max_idx]
    rho_sorted_knn[0, ] = rho[indices[max_idx, :]]

    # 依据密度阈值，更新代表点
    for i in range(1, n_samples):
        idx = rho_sorted_idx[i] # 原始索引
        rho_sorted_knn[i,] = rho[indices[idx, :]]
        rho_mean_idx = rho_sorted_knn[i, ].mean()
        rho_median_idx = np.median(rho_sorted_knn[i, ])
        # 局部密度均值大于全局时，密度阈值不做处理
        shink_scale = (np.log(rho_mean_idx)/np.log(rho_mean))
        rho_t = rho_median_idx * shink_scale
        rho_t = np.min([rho_t, rho_median_idx])

        search_range = np.ceil(ini_scope * shink_scale).astype(int)
        search_range = np.min([ini_scope, search_range])
        # print(f"搜索范围：{search_range}")
        knn_idx = indices[idx, 0 : search_range]  # 邻居索引

        # 排除密度极小的样本，5%rho
        if np.intersect1d(rho_precent, idx):
            continue

        # 搜索邻域内已经存在采样点的排除
        elif len(np.intersect1d(rep_data, knn_idx)) > 0:
            # print(np.intersect1d(rep_data, knn_idx))
            continue

        elif rho[idx] > rho_t:
            rep_data.append(idx)
            # print(f"局部密度：{rho_mean_idx}")
            # print(f"阈值：{rho_t}")
            # print(rho[idx])

    rep_data_idx = np.array(rep_data)

    return rep_data_idx, rho_sorted_knn, rho_sorted_idx

#######################################################
# Assign remaining points
#######################################################
def assign_points(data, y_pred_downsample, rep_data_idx, indices, distances, rho_sorted_idx):
    # 硬标签，依最近邻划分
    n_samples = indices.shape[0]

    # 初始化标签
    labels = torch.full((n_samples,), -1, dtype=torch.long)

    # 给采样点分配标签
    for i, rep_data_idx in enumerate(rep_data_idx):
        labels[rep_data_idx] = y_pred_downsample[i]


    for idx in rho_sorted_idx:
        if labels[idx] == -1:  # 未分配的点
            label_id_knn = labels[indices[idx]]
            mask = label_id_knn >= 0
            filtered_label = label_id_knn[mask]
            # mask1 = label_id_knn == -1

            if len(filtered_label) > 0: # 邻域内有已标签点，将其分配到最近的有标签点中，同时为其领域其他无标签的点赋值同样标签
                labels[idx] = filtered_label[0]
                # labels[indices[idx, mask1]] = filtered_label[0]
            # else: # 邻域内无已标签点，将该点与领域内点都分配到距离最近的采样点相同的类中


    return labels


#######################################################
# Evaluate Critiron
#######################################################

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed.
    y_true: true labels, numpy.array with shape (n_samples,)
    y_pred: predicted labels, numpy.array with shape (n_samples,)

    return accuracy, in [0,1]
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # 使用 scipy 的 linear_sum_assignment
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        acc = w[row_ind, col_ind].sum() * 1.0 / y_pred.size
    except ImportError:
        # 如果没有 scipy，尝试使用旧的 sklearn
        try:
            from sklearn.utils.linear_assignment_ import linear_assignment
            ind = linear_assignment(w.max() - w)
            row_ind = ind[:, 0]
            col_ind = ind[:, 1]
            acc = w[row_ind, col_ind].sum() * 1.0 / y_pred.size
        except ImportError:
            # 如果都没有，使用简单的最大值匹配
            acc = np.max(w, axis=1).sum() * 1.0 / y_pred.size

    return acc