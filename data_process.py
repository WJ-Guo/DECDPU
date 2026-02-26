import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import numpy as np
from tqdm import tqdm
from scipy.io import arff, loadmat
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from scipy.io import arff, loadmat


def load_with_pytorch(dataset_name='cifar10', transform=None, flatten=True, normalize=True):
    """
    使用PyTorch官方接口加载数据

    参数:
        dataset_name: 数据集名称，可选 'cifar10', 'mnist', 'fashion-mnist', 'cifar100'
        transform: 数据转换管道，如果为None则使用默认转换
        flatten: 是否将特征平铺为一维
        normalize: 是否将数据标准化到0-1范围

    返回:
        all_features: 合并后的特征张量
        all_labels: 合并后的标签张量
    """

    # 定义可用的数据集
    dataset_map = {
        'cifar10': {
            'class': torchvision.datasets.CIFAR10,
            'channels': 3,
            'size': 32,
            'num_classes': 10
        },
        'STL10': {
            'class': torchvision.datasets.STL10,
            'channels': 3,
            'size': 96,
            'num_classes': 10
        },
        'mnist': {
            'class': torchvision.datasets.MNIST,
            'channels': 1,
            'size': 28,
            'num_classes': 10
        },
        'fashion-mnist': {
            'class': torchvision.datasets.FashionMNIST,
            'channels': 1,
            'size': 28,
            'num_classes': 10
        }
    }

    # 检查数据集是否支持
    if dataset_name not in dataset_map:
        raise ValueError(f"不支持的数据集: {dataset_name}。可选: {list(dataset_map.keys())}")

    dataset_info = dataset_map[dataset_name]
    dataset_class = dataset_info['class']

    print(f"加载数据集: {dataset_name.upper()}")
    print(f"通道数: {dataset_info['channels']}")
    print(f"图像尺寸: {dataset_info['size']}x{dataset_info['size']}")
    print(f"类别数: {dataset_info['num_classes']}")

    # 设置默认的transform
    if transform is None:
        transform_list = [transforms.ToTensor()]

        # 如果是MNIST数据集，确保转换为3通道以保持一致性
        if dataset_name in ['mnist', 'fashion-mnist']:
            transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

        transform = transforms.Compose(transform_list)

    if dataset_name.lower() == 'stl10':
        # STL-10使用split参数而不是train参数
        trainset = dataset_class(
            root='data/',
            split='train',  # 使用split参数
            download=True,
            transform=transform
        )
        testset = dataset_class(
            root='data/',
            split='test',  # 使用split参数
            download=True,
            transform=transform
        )
    else:
        # 其他数据集（如CIFAR10, CIFAR100等）使用train参数
        trainset = dataset_class(
            root='data/',
            train=True,  # 使用train参数
            download=True,
            transform=transform
        )
        testset = dataset_class(
            root='data/',
            train=False,  # 使用train参数
            download=True,
            transform=transform
        )
    # 合并数据
    all_features = []
    all_labels = []

    print("\n合并训练数据...")
    for i in tqdm(range(len(trainset)), desc="处理训练样本"):
        img, label = trainset[i]
        all_features.append(img)
        all_labels.append(label)

    print("合并测试数据...")
    for i in tqdm(range(len(testset)), desc="处理测试样本"):
        img, label = testset[i]
        all_features.append(img)
        all_labels.append(label)

    # 转换为张量
    all_features = torch.stack(all_features)
    all_labels = torch.tensor(all_labels)

    print(f"\n原始特征形状: {all_features.shape}")
    print(f"原始标签形状: {all_labels.shape}")

    # 平铺处理
    if flatten:
        original_shape = all_features.shape
        all_features = all_features.view(all_features.size(0), -1)  # 平铺为二维 [样本数, 特征维度]
        if dataset_name in ['mnist', 'fashion-mnist']:
            all_features = all_features[:,0:784]
        print(f"平铺后特征形状: {all_features.shape}")
        print(f"每个样本的特征维度: {all_features.size(1)}")

    # 标准化处理（确保在0-1范围内）
    if normalize:
        # 检查当前数据范围
        min_val = all_features.min().item()
        max_val = all_features.max().item()
        print(f"标准化前 - 最小值: {min_val:.6f}, 最大值: {max_val:.6f}")

        if min_val < 0 or max_val > 1:
            # 如果不在0-1范围内，进行标准化
            all_features = (all_features - min_val) / (max_val - min_val)
            print(f"执行了标准化到[0,1]范围")
        else:
            print(f"数据已在[0,1]范围内，跳过标准化")

        # 验证标准化结果
        min_val_norm = all_features.min().item()
        max_val_norm = all_features.max().item()
        print(f"标准化后 - 最小值: {min_val_norm:.6f}, 最大值: {max_val_norm:.6f}")

    # 输出数据统计信息
    print("\n" + "=" * 50)
    print("数据统计信息:")
    print(f"总样本数: {len(all_features)}")
    print(f"特征形状: {all_features.shape}")
    print(f"标签形状: {all_labels.shape}")
    print(f"特征数据类型: {all_features.dtype}")
    print(f"标签数据类型: {all_labels.dtype}")

    return all_features, all_labels

############# USPS ###########################################################

def load_usps_from_arff(file_path, flatten=True, normalize=True):
    """
    加载USPS.arff文件并转换为PyTorch张量

    参数:
        file_path: USPS.arff文件路径
        flatten: 是否平铺为一维特征
        normalize: 是否标准化到0-1范围

    返回:
        all_features: 特征张量
        all_labels: 标签张量
    """

    print(f"加载USPS数据集: {file_path}")

    # 读取ARFF文件
    try:
        # 方法1: 使用scipy
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
    except:
        # 方法2: 使用liac-arff (备用)
        import arff as liac_arff
        with open(file_path, 'r') as f:
            dataset = liac_arff.load(f)
        df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])

    print(f"原始数据形状: {df.shape}")

    # 分离特征和标签
    # USPS数据集的第一列是标签
    labels = df.iloc[:, 0].values

    # 处理标签（ARFF文件中的标签可能是字节字符串）
    if isinstance(labels[0], bytes):
        labels = [label.decode('utf-8') for label in labels]

    # 转换为数值标签
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    labels = np.array([label_to_idx[label] for label in labels])

    # 特征数据（排除最后一列标签）
    features = df.iloc[:, 1:].values.astype(np.float32)

    print(f"特征形状: {features.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"类别数: {len(unique_labels)}")
    print(f"类别映射: {label_to_idx}")

    # 转换为PyTorch张量
    all_features = torch.from_numpy(features)
    all_labels = torch.from_numpy(labels)

    # USPS数据通常是16x16的图像，如果需要可以重塑为图像格式
    if not flatten:
        # 重塑为图像格式 [样本数, 通道数, 高度, 宽度]
        # USPS是16x16的灰度图像
        all_features = all_features.view(-1, 1, 16, 16)
        print(f"重塑为图像格式: {all_features.shape}")

    # 标准化处理
    if normalize:
        min_val = all_features.min().item()
        max_val = all_features.max().item()
        print(f"标准化前 - 最小值: {min_val:.6f}, 最大值: {max_val:.6f}")

        if min_val < 0 or max_val > 1:
            all_features = (all_features - min_val) / (max_val - min_val)
            print("执行了标准化到[0,1]范围")

        min_val_norm = all_features.min().item()
        max_val_norm = all_features.max().item()
        print(f"标准化后 - 最小值: {min_val_norm:.6f}, 最大值: {max_val_norm:.6f}")

    # 输出统计信息
    print("\n" + "=" * 50)
    print("USPS数据统计信息:")
    print(f"总样本数: {len(all_features)}")
    print(f"特征形状: {all_features.shape}")
    print(f"标签形状: {all_labels.shape}")
    print(f"特征数据类型: {all_features.dtype}")
    print(f"标签数据类型: {all_labels.dtype}")
    print(f"类别分布: {torch.bincount(all_labels).tolist()}")

    return all_features, all_labels

########################  reuters10k / HAR ###################################

def load_from_mat(file_path):
    print(f"loading: {file_path}")
    if file_path == 'data/HAR':

        data = loadmat(file_path)
        X_train = torch.from_numpy(data['X']).float()
        y_train = torch.from_numpy(data['Y']).long()
        y_train = y_train.transpose(dim0=1,dim1=0)
        y_train = y_train.squeeze()
    else:
        data = loadmat(file_path)

        X_dense = data['X']
        X_log = np.log1p(X_dense)
        # Scales to [0, 1] for BCE Loss
        scaler = MaxAbsScaler()
        X_processed = scaler.fit_transform(X_log)

        X_train = torch.from_numpy(X_processed).float()
        y_train = torch.from_numpy(data['Y']).long()
        y_train = y_train.transpose(dim0=1, dim1=0)
        y_train = y_train.squeeze()

    return X_train, y_train

def load_20news(max_features=2000):

    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    docs, true_labels = data.data, data.target

    # TF-IDF Processing
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', sublinear_tf=True)
    X = vectorizer.fit_transform(docs)

    # Log smoothing and Scaling
    X_dense = X.toarray()
    X_log = np.log1p(X_dense)

    # Scales to [0, 1] for BCE Loss
    scaler = MaxAbsScaler()
    X_processed = scaler.fit_transform(X_log)

    all_data = torch.from_numpy(X_processed).float()
    all_labels = torch.from_numpy(true_labels).long()

    return all_data, all_labels

class TensorDatasetCustom(Dataset):
    """自定义Dataset类，用于包装特征和标签张量"""

    def __init__(self, features, labels):
        """
        参数:
            features: 特征张量，形状为 [样本数, 特征维度]
            labels: 标签张量，形状为 [样本数]
        """
        self.features = features
        self.labels = labels

        # 验证数据形状
        assert len(self.features) == len(self.labels), \
            f"Features and labels not match : {len(self.features)} != {len(self.labels)}"

    def __len__(self):
        """返回数据集大小"""
        return len(self.features)

    def __getitem__(self, idx):
        """根据索引获取单个样本"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.features[idx], self.labels[idx], torch.from_numpy(np.array(idx))




