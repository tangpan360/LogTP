import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from transformers import BertTokenizer
# from concurrent.futures import ThreadPoolExecutor, as_completed


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


# 定义日志数据集类
class LogDataset(Dataset):
    def __init__(self, features, attention_mask, domain_labels, labels):
        super(LogDataset, self).__init__()
        self.features = features  # 特征
        self.attention_mask = attention_mask  # 掩码
        self.domain_labels = domain_labels  # 域标签
        self.labels = labels  # 标签

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  # 将索引转换为列表类型

        return self.features[idx], self.attention_mask[idx], self.domain_labels[idx], self.labels[idx]


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


def get_iter(X, X_mask, y_d, y, batch_size=1024, shuffle=True):
    dataset = LogDataset(X, X_mask, y_d, y)
    if shuffle:
        iter = DataLoader(dataset, batch_size, shuffle=True, worker_init_fn=seed_worker)
    else:
        iter = DataLoader(dataset, batch_size)
    return iter


def get_train_eval_iter(train_normal_s, train_normal_t, window_size=20, emb_dim=300, batch_size=1024):
    """
    获取训练集iter和验证集iter
    :param train_normal_s: 源域训练集正常数据
    :param train_normal_t: 目标域训练集正常数据
    :param window_size: 序列划分窗长
    :param emb_dim: 编码维度
    :param batch_size: 每批训练数据量
    :return: 返回训练集迭代器train_iter和验证集迭代器eval_iter
    """
    # X = list(train_normal_s.Embedding.values)  # 将源域训练集正常数据放到列表中
    # X.extend(list(train_normal_t.Embedding.values))  # 将目标域训练集正常数据合并到源域训练集正常数据列表中。（此时每个元素为20*300的列表）
    # X_new = []
    # for i in tqdm(X):  # 对于200,000个序列循环转换为np.array数据
    #     temp = []
    #     for j in i:
    #         temp.extend(j)  # 对于每个序列中的20个logkey循环extend (300->600->900->...->6000) 添加到临时列表中
    #     X_new.append(np.array(temp).reshape(window_size, emb_dim))  # 将每个序列划分并转换为张量。（此时每个元素转换为20*300的张量）
    # print(train_normal_s.Content.values)
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    X2 = list(train_normal_s.Content.values)
    # TODO 不合并目标域和源域数据
    # X2.extend(train_normal_t.Content.values)
    X2_new = []
    for i in tqdm(X2):
        # 使用"[SEP]"连接每个字符串，得到每个列表的串联结果
        temp_string = " [SEP] ".join(i)
        X2_new.append(temp_string)
    # TODO 对拼接后的日志进行分词和编码转换成input_ids和attention_mask
    # 对拼接后的日志字符串进行分词处理
    start_time = time.time()
    inputs = tokenizer(X2_new, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # def tokenize_text(text):
    #     return tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # 使用ThreadPoolExecutor来并行处理
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     # 提交所有文本到线程池
    #     futures = [executor.submit(tokenize_text, text) for text in X2_new]
    #
    #     # 收集处理结果
    #     results = []
    #     for future in as_completed(futures):
    #         results.append(future.result())

    end_time = time.time()
    print("Tokenizer time: ", end_time - start_time)
    # X_new = results
    # TODO 并且提取出来 input_ids 和 attention_mask
    input_ids = torch.tensor(inputs["input_ids"])
    attention_mask = torch.tensor(inputs["attention_mask"])
    y_d = list(train_normal_s.target.values)  # 源域标签
    # y_d.extend(list(train_normal_t.target.values))  # 目标域域标签合并到源域域标签
    y = list(train_normal_s.Label.values)  # 源域标签
    # y.extend(list(train_normal_t.Label.values))  # 目标域标签合并到源域标签
    # TODO 此时应该只有100,000个源域数据
    # TODO input_ids 和 attention_mask 放到一起进行训练验证集合划分
    # X_train, X_eval, y_d_train, y_d_eval, y_train, y_eval = train_test_split(X_new, y_d, y, test_size=0.2, random_state=42)
    X_train, X_eval, X_mask_train, X_mask_eval, y_d_train, y_d_eval, y_train, y_eval = train_test_split(input_ids, attention_mask, y_d, y, test_size=0.2, random_state=42)
    # X_train = torch.tensor(X_train, requires_grad=False)
    # X_eval = torch.tensor(X_eval, requires_grad=False)
    y_d_train = torch.tensor(y_d_train).reshape(-1, 1).long()
    y_d_eval = torch.tensor(y_d_eval).reshape(-1, 1).long()
    y_train = torch.tensor(y_train).reshape(-1, 1).long()
    y_eval = torch.tensor(y_eval).reshape(-1, 1).long()
    train_iter = get_iter(X_train, X_mask_train, y_d_train, y_train, batch_size)
    eval_iter = get_iter(X_eval, X_mask_eval, y_d_eval, y_eval, batch_size)
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

    # 从第五个epoch开始绘制
    train_loss = train_loss[train_loss['epoch'] >= 5]
    valid_loss = valid_loss[valid_loss['epoch'] >= 5]

    sns.lineplot(x="epoch", y="loss", data=train_loss, label="train loss")
    sns.lineplot(x="epoch", y="loss", data=valid_loss, label="valid loss")

    # 找到验证损失中的最低点
    min_valid_loss = valid_loss.loc[valid_loss['loss'].idxmin()]
    plt.scatter(min_valid_loss['epoch'], min_valid_loss['loss'], color='red')  # 用红色原点标记最低点

    plt.title("epoch vs train loss vs valid loss")
    plt.legend()
    plt.savefig(loss_dir + "/train_valid_loss.png")
    plt.show()
    print("plot done")
