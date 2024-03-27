import os
import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# 定义日志数据集类
class LogDataset(Dataset):
    def __init__(self, features, domain_labels, labels):
        super(LogDataset, self).__init__()
        self.features = features  # 特征
        self.domain_labels = domain_labels  # 域标签
        self.labels = labels  # 标签

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  # 将索引转换为列表类型

        return self.features[idx], self.domain_labels[idx], self.labels[idx]


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_dist(ts, center):
    ts = ts.cpu().detach().numpy()
    center = center.cpu().numpy()
    temp = []
    for i in ts:
        temp.append(np.linalg.norm(i-center))
    return temp


def get_center(emb, label=None):
    if label is None:
        return torch.mean(emb, 0)
    else:
        return 'Not defined'


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_iter(X, y_d, y, batch_size=1024, shuffle=True):
    dataset = LogDataset(X, y_d, y)
    if shuffle:
        iter = DataLoader(dataset, batch_size, shuffle=True, worker_init_fn=seed_worker)
    else:
        iter = DataLoader(dataset, batch_size)
    return iter


def get_train_eval_iter(train_normal_s, train_normal_t, window_size=20, emb_size=300):
    """
    获取训练集iter和验证集iter
    :param train_normal_s: 源域训练集正常数据
    :param train_normal_t: 目标域训练集正常数据
    :param window_size: 序列划分窗长
    :param emb_size: 编码维度
    :return: 返回训练集迭代器train_iter和验证集迭代器eval_iter
    """
    X = list(train_normal_s.Embedding.values)  # 将源域训练集正常数据放到列表中
    X.extend(list(train_normal_t.Embedding.values))  # 将目标域训练集正常数据合并到源域训练集正常数据列表中。（此时每个元素为20*300的列表）
    X_new = []
    for i in tqdm(X):  # 对于200,000个序列循环转换为np.array数据
        temp = []
        for j in i:
            temp.extend(j)  # 对于每个序列中的20个logkey循环extend (300->600->900->...->6000) 添加到临时列表中
        X_new.append(np.array(temp).reshape(window_size, emb_size))  # 将每个序列划分并转换为张量。（此时每个元素转换为20*300的张量）
    y_d = list(train_normal_s.target.values)  # 源域标签
    y_d.extend(list(train_normal_t.target.values))  # 目标域域标签合并到源域域标签
    y = list(train_normal_s.Label.values)  # 源域标签
    y.extend(list(train_normal_t.Label.values))  # 目标域标签合并到源域标签
    X_train, X_eval, y_d_train, y_d_eval, y_train, y_eval = train_test_split(X_new, y_d, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, requires_grad=False)
    X_eval = torch.tensor(X_eval, requires_grad=False)
    y_d_train = torch.tensor(y_d_train).reshape(-1, 1).long()
    y_d_eval = torch.tensor(y_d_eval).reshape(-1, 1).long()
    y_train = torch.tensor(y_train).reshape(-1, 1).long()
    y_eval = torch.tensor(y_eval).reshape(-1, 1).long()
    train_iter = get_iter(X_train, y_d_train, y_train)
    eval_iter = get_iter(X_eval, y_d_eval, y_eval)
    return train_iter, eval_iter


def dist2label(lst_dist, R):
    y = []
    for i in lst_dist:
        if i <= R:
            y.append(0)
        else:
            y.append(1)
    return y


def plot_train_valid_loss(loss_dir):
    train_loss = pd.read_csv(loss_dir + "/train_log.csv")
    valid_loss = pd.read_csv(loss_dir + "/valid_log.csv")
    sns.lineplot(x="epoch", y="loss", data=train_loss, label="train loss")
    sns.lineplot(x="epoch", y="loss", data=valid_loss, label="valid loss")
    plt.title("epoch vs train loss vs valid loss")
    plt.legend()
    plt.savefig(loss_dir + "/train_valid_loss.png")
    plt.show()
    print("plot done")
