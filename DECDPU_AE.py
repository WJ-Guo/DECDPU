from __future__ import print_function, division
import argparse
import numpy as np

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
import matplotlib.pyplot as plt
import os

from UMAP_draw import basic_umap_visualization
from DPC_clsuter import DensityPeakClustering
from utils import cluster_acc, Calaculate_rho_KNN, choose_points, assign_points
from data_process import load_usps_from_arff, load_with_pytorch, load_from_mat,load_20news,TensorDatasetCustom



class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()

        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)

        self.z_layer = Linear(n_enc_3, n_z)

        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)

        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        # encoder
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))

        z = self.z_layer(enc_h3)

        # decoder
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


class IDEC(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z,
                 n_clusters,
                 alpha=1,
                 pretrain_path='data/ae_mnist.pkl'):  # 预训练模型
        super(IDEC, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))  # 可学习的聚类中心矩阵
        torch.nn.init.xavier_normal_(self.cluster_layer.data)  # 初始化中心，xavier为正态分布初始化

    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
        # load pretrain weights
        self.ae.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained ae from', path)

    def forward(self, x):
        x_bar, z = self.ae(x)
        # cluster，计算软分布概率
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)  # pow计算平方差，alpha为1的 t 分布
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, q


# 构建目标分布pij,锐化后平衡
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def pretrain_ae(model):
    '''
    pretrain autoencoder，预训练自编码器
    '''
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    epochs_pretrain = args.pretrain_epochs

    for epoch in range(epochs_pretrain):
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)  #
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / (batch_idx + 1)
        print("epoch {} loss={:.4f}".format(epoch, avg_loss))
        torch.save(model.state_dict(), args.pretrain_path)

        # loss save
        loss_file_path = "Loss/loss_ae.txt"
        with open(loss_file_path, 'a') as f:
            f.write(f"{epoch}\t{avg_loss:.6f}\n")

    print("model saved to {}.".format(args.pretrain_path))


def train_idec():
    model = IDEC(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=1000,
        n_dec_1=1000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        alpha=1.0,
        pretrain_path=args.pretrain_path).to(device)

    if os.path.exists(args.pretrain_path):
        model.pretrain(args.pretrain_path)
    else:
        model.pretrain()


    # cluster parameter initiate
    data = dataset.features
    y = dataset.labels
    data = torch.Tensor(data).to(device)
    x_bar, hidden = model.ae(data)

    print(hidden)
    print(type(hidden))
    print(hidden.shape)  # 查看属性和方法

    ######  DPC  #############################################################################
    n_UPMA = args.n_UPMA
    # X_UMAP, _ = basic_umap_visualization(data.cpu().detach().numpy(), y, n_components = n_UPMA)
    X_UMAP, _ = basic_umap_visualization(hidden.cpu().detach().numpy(), y, n_components = n_UPMA)

    ## 下采样：降低内存复杂度
    distances, indices, rho, kdtree = Calaculate_rho_KNN(X_UMAP, k = 50)
    print("knn构建完成")
    ini_scope = args.ini_scope
    rep_data_idx, rho_sorted_KNN, rho_sorted_idx = choose_points(distances, indices, rho, ini_scope = ini_scope)
    print(f"代表点个数:{rep_data_idx.shape}")
    data_np = X_UMAP
    rep_data = X_UMAP[rep_data_idx]
    rep_data_tens = torch.Tensor(rep_data).to(device)

    plt.figure(figsize=(10, 8))
    plt.scatter(data_np[rep_data_idx, 0], data_np[rep_data_idx, 1],
               alpha=0.5,  # 设置透明度
               s=30,       # 设置点的大小
               c='r',   # 设置点的颜色
               edgecolors='none')  # 去掉边缘线

    plt.tight_layout()
    plt.show()

    ## 嵌入改进密度聚类方法替换K-means
    dpc = DensityPeakClustering(dc_percent=2.0, use_gpu=False, verbose=False)
    y_pred_downsample = dpc.fit_predict(rep_data_tens, manual_select=True)
    model.cluster_layer.data = torch.tensor(hidden.data[rep_data_idx[dpc.centers_], :]).to(device)
    y_pred = assign_points(data, y_pred_downsample, rep_data_idx, indices, distances, rho_sorted_idx)

    unique_labels = np.unique(y_pred)
    colormaps = ['tab20']

    for cmap_name in colormaps:
        plt.figure(figsize=(10, 8))
        cmap = plt.cm.get_cmap(cmap_name)

        for i, label in enumerate(unique_labels):
            mask = y_pred == label
            if label == -1:
                plt.scatter(X_UMAP[mask, 0], X_UMAP[mask, 1], c='k', label='noisy', alpha=0.7, edgecolors='w',s=30)
            else:
                # 从色谱中获取颜色
                color = cmap(i / max(1, len(unique_labels) - 1))
                plt.scatter(X_UMAP[mask, 0], X_UMAP[mask, 1], c=[color], label=f'Class {label}',alpha=0.7,edgecolors='w',s=50)

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
        plt.text(x1+1, x2+1, f'{center_idx}',  # 显示原始样本编号
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

    nmi_k = nmi_score(y, y_pred)
    acc_k = cluster_acc(y, y_pred)
    ari_k = ari_score(y, y_pred)
    print(':Acc {:.4f}'.format(acc_k),
          ', nmi {:.4f}'.format(nmi_k), ', ari {:.4f}'.format(ari_k))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', default=256, type=int)

    parser.add_argument('--n_z', default=30, type=int)
    parser.add_argument('--n_UPMA', default=2, type=int)
    parser.add_argument('--ini_scope', default=30, type=int)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--pretrain_path', type=str, default=f'pretrain/ae_mnist30.pkl')

    parser.add_argument('--gamma', default=0.1, type=float, help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--pretrain_epochs', default=150, type=int)


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    print(args)
    if args.dataset in ['cifar10', 'mnist', 'fashion-mnist', 'STL10']:
        all_data, all_labels = load_with_pytorch(dataset_name = args.dataset)
        args.n_clusters = 10
        args.n_input = 784 # 784 for 28*28, 27648 for 96*96*3

    elif args.dataset == 'USPS':
        all_data, all_labels = load_usps_from_arff(file_path = 'data/usps.arff')
        args.n_clusters = 10
        args.n_input = 256

    elif args.dataset in ['reuters10k', 'HAR']:
        all_data, all_labels = load_from_mat(file_path=f'data/{args.dataset}')
        args.n_clusters = 4 if args.dataset == 'reuters10k' else 6
        args.n_input = 2000 if args.dataset == 'reuters10k' else 561

    elif args.dataset == '20newsgroups':
        all_data, all_labels = load_20news(max_features=2000)
        args.n_clusters = 20
        args.n_input = 2000

    dataset = TensorDatasetCustom(all_data, all_labels)

    train_idec()
