import sys

import numpy as np
from gensim.models import Word2Vec
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

sys.path.append("..")


def word2vec_train(lst, emb_dim=150, seed=42):
    """
    对于列表lst中的所有字符串进行切割，训练，最终返回一个训练好的word2vec模型-w2v。
    :param lst: (list of string)用来训练word2vec模型的列表，列表中的每个元素都是字符串句子
    :param emb_dim: word2vec编码的向量维度
    :param seed: 随机种子
    :return: 训练好的word2vec模型
    """
    tokenizer = RegexpTokenizer(r'\w+')
    sentences = []
    # 本来一个列表，列表中的每个元素为字符串句子，切割后元素为子列表，子列表中的元素由原字符串句子分割后的单词组成。
    for i in lst:
        sentences.append([x.lower() for x in tokenizer.tokenize(str(i))])
    # 对切割后的2维列表进行训练，得到训练好的模型w2v
    w2v = Word2Vec(sentences, size=emb_dim, min_count=1, seed=seed, workers=1)
    return w2v


def get_sentence_emb(sentence, w2v):
    """
    把句子中的
    :param sentence:
    :param w2v:
    :return:
    """
    tokenizer = RegexpTokenizer(r'\w+')
    lst = []
    # 把列表中仅有的一个字符串元素切割成多个单词元素。‘<*>’会被去掉
    tokens = [x.lower() for x in tokenizer.tokenize(str(sentence))]
    if tokens == []:
        tokens.append('EmptyParametersTokens')
    for i in range(len(tokens)):
        words = list(w2v.wv.vocab.keys())
        if tokens[i] in words:
            lst.append(w2v[tokens[i]])
        else:
            w2v.build_vocab([[tokens[i]]], update=True)
            w2v.train([tokens[i]], epochs=1, total_examples=len([tokens[i]]))
            lst.append(w2v[tokens[i]])
    drop = 1
    # 先检查lst中的向量是不是二维的，也就是一句话中是不是有多个单词。
    if len(np.array(lst).shape) >= 2:
        # 如果有多个单词，则对多个词的向量求平均值，比如6个词，那就是6个300维向量，求平均值后变成1个300维向量。
        sen_emb = np.mean(np.array(lst), axis=0)
        # 再判断lst中的向量是否大于5个，如果大于5个，则drop为0，否则drop为1。注意此处，很多小于5个的EventTemplate被删掉了。
        if len(np.array(lst)) >= 5:
            drop = 0
    else:
        sen_emb = np.array(lst)
    return list(sen_emb), drop


def word2emb(df_source, df_target, train_size_s, train_size_t, step_size, emb_dim, seed):
    """
    将df_source和df_target进行向量编码处理，最后返回的是添加了2列数据（1.编码后的EventTemplate和2.drop标签）的新dataframe
    :param df_source: 源域dataframe
    :param df_target: 目标域dataframe
    :param train_size_s: 源域的训练集size
    :param train_size_t: 目标域的训练集size
    :param step_size: 每隔step_size步，进行依次序列切分切分的步长
    :param emb_dim: 编码向量的维度
    :param seed: 随机种子
    :return: df_source 多了 向量编码列 以及 drop标签 的新的dataframe
    """
    # 利用源域和目标域的数据训练一个word2vec模型，此处只利用了源域和目标域的一部分数据进行训练，所以后边可能会遇到w2v模型中不认识的单词，具体解决方案看后边。
    w2v = word2vec_train(np.concatenate((df_source.EventTemplate.values[:step_size * train_size_s],
                                         df_target.EventTemplate.values[:step_size * train_size_t])),
                         emb_dim=emb_dim, seed=seed)
    print('Processing words in the source dataset')
    # 字典dic内容：key为EventTemplate，value为元组(句子向量-300维列表，是否丢弃-0)
    dic = {}
    # 获取源域所有EventTemplate的集合并转化成列表
    lst_temp = list(set(df_source.EventTemplate.values))
    # 为每个EventTemplate获取其向量以及根据长度得到的是否丢弃的标签
    for i in tqdm(range(len(lst_temp))):
        (temp_val, drop) = get_sentence_emb([lst_temp[i]], w2v)
        dic[lst_temp[i]] = (temp_val, drop)
    lst_emb = []
    lst_drop = []
    for i in tqdm(range(len(df_source))):
        # 获取所有的df_source对应的向量，赋值到lst_emb里
        lst_emb.append(dic[df_source.EventTemplate.loc[i]][0])
        # 获取所有的df_source对应的drop标签，赋值到lst_drop里
        lst_drop.append(dic[df_source.EventTemplate.loc[i]][1])
    # 将df_source中所有元素对应的向量添加到df_source的新一列
    df_source['Embedding'] = lst_emb
    # 将df_source中所有元素对应的drop标签添加到df_source的新一列
    df_source['drop'] = lst_drop
    print('Processing words in the target dataset')
    dic = {}
    # 获取目标域所有EventTemplate的集合并转化成列表
    lst_temp = list(set(df_target.EventTemplate.values))
    # 为每个EventTemplate获取其向量以及根据长度得到的是否丢弃的标签
    for i in tqdm(range(len(lst_temp))):
        (temp_val, drop) = get_sentence_emb([lst_temp[i]], w2v)
        dic[lst_temp[i]] = (temp_val, drop)
    lst_emb = []
    lst_drop = []
    for i in tqdm(range(len(df_target))):
        lst_emb.append(dic[df_target.EventTemplate.loc[i]][0])
        lst_drop.append(dic[df_target.EventTemplate.loc[i]][1])
    df_target['Embedding'] = lst_emb
    df_target['drop'] = lst_drop

    df_source = df_source.loc[df_source['drop'] == 0]
    df_target = df_target.loc[df_target['drop'] == 0]

    print(f'Source length after drop none word logs: {len(df_source)}')
    print(f'Target length after drop none word logs: {len(df_target)}')

    return df_source, df_target, w2v


def sliding_window(df, window_size=20, step_size=4, target=0, val_date='2005.11.15'):
    """
    对dataframe进行日志序列的划分，并且添加是否为目标域的标签以及是否为验证集的标签
    :param df: 需要进行窗口划分的df数据
    :param window_size: 窗口长度，window_size长度的日志将被划分为一个序列
    :param step_size: 隔step_size步长进行依次窗口划分
    :param target: 是否是目标域，0不是，1是
    :param val_date: 作为验证集的日期
    :return: 返回每隔4个切一个序列，并且序列中包含是否为目标域的标签（1表示为目标域），以及是否为验证集的标签（1表示为验证集）
    """
    # 将标签转换为0和1，对于x!='-'，标签变为1，x='-'标签变为0。
    df["Label"] = df["Label"].apply(lambda x: int(x != '-'))
    # df 只保留其中的4列
    df = df[["Label", "Content", "Embedding", "Date"]]
    # 添加一列['target']=0，此时代表这些数据为源域的数据
    df["target"] = target
    # 添加一列['val'] = 0，此时代表这些数据都不是验证集数据。
    df["val"] = 0
    # log_size代表数据数量，代表有多少个日志
    log_size = df.shape[0]
    label_data = df.iloc[:, 0]  # 代表['Label']这一列的异常标签
    logkey_data = df.iloc[:, 1]  # 代表['Content']这一列的logkey
    emb_data = df.iloc[:, 2]  # 代表['Embedding'] 这一列的向量
    date_data = df.iloc[:, 3]  # 代表['Date']这一列的日期
    new_data = []
    index = 0
    # 对所有的日志进行序列划分
    while index <= log_size - window_size:
        # 如果日期等于验证集的日期，在进行序列划分的时候将验证标签赋值为1
        if date_data.iloc[index] == val_date:
            new_data.append([
                max(label_data[index: index + window_size]),  # 序列标签，当序列中所有日志均为0时，序列标签为0，否则序列标签为1
                logkey_data[index: index + window_size].values,  # 将20组logkey放到一个列表里
                emb_data[index: index + window_size].values,  # 将20组emb_data编码向量放到一个列表里
                date_data.iloc[index],  # 序列中第一个日志的日期作为序列的日期放到列表里
                target,  # 将target为0放到列表里
                1  # 将验证集标签1（为验证集）放到列表里
            ])
            index += step_size
        # 如果日期不等于验证集的日期，在进行序列划分的时候将验证标签赋值为0
        else:
            new_data.append([
                max(label_data[index:index + window_size]),
                logkey_data[index:index + window_size].values,
                emb_data[index:index + window_size].values,
                date_data.iloc[index],
                target,
                0
            ])
            index += step_size
    return pd.DataFrame(new_data, columns=df.columns)


def get_datasets(df_source, df_target, options, val_date="2005.11.15"):
    # Get source data preprocessed
    window_size = options["window_size"]
    print("window_size: ", window_size)
    step_size = options["step_size"]
    print("step_size: ", step_size)
    source = options["source_dataset_name"]
    target = options["target_dataset_name"]
    train_size_s = options["train_size_s"]
    train_size_t = options["train_size_t"]
    emb_dim = options["emb_dim"]
    times = int(train_size_s / train_size_t) - 1
    seed = options["random_seed"]

    # 编码成向量添加到 df_source 和 df_target
    df_source, df_target, w2v = word2emb(df_source, df_target, train_size_s, train_size_t, step_size, emb_dim, seed)

    print(f'Starting preprocessing for the source: {source} dataset')
    # 将数据集进行切割，输出6列的dataframe，包括Label，Content（序列logkey），Embedding（序列embedding），Date，target（目标域标签），val（验证集标签）。
    window_df = sliding_window(df_source, window_size, step_size, 0, val_date)
    r_s_val_df = window_df[window_df['val'] == 1]  # 源域验证集序列
    window_df = window_df[window_df['val'] == 0]  # 源域训练集序列

    # Training normal data
    df_normal = window_df[window_df["Label"] == 0]  # 源域训练集中的正常序列

    # shuffle normal data
    df_normal = df_normal.sample(frac=1, random_state=42).reset_index(drop=True)  # 对源域训练集中正常数据顺序打乱
    train_len = train_size_s  # 源域训练集数据数量为10,000

    train_normal_s = df_normal[:train_len]  # 取源域中正常序列的前100,000个作为训练数据
    print("Source training size {}".format(len(train_normal_s)))

    # Test normal data
    test_normal_s = df_normal[train_len:]  # 取源域中正常序列剩下的部分作为测试集
    print("Source test normal size {}".format(len(test_normal_s)))

    # Test abnormal data
    test_abnormal_s = window_df[window_df["Label"] == 1]  # 取标签为1的作为源域异常序列测试集
    print('Source test abnormal size {}'.format(len(test_abnormal_s)))

    print('------------------------------------------')
    print(f'Start preprocessing for the target: {target} dataset')
    # Get target data preprocessed
    window_df = sliding_window(df_target, window_size, step_size, 1, val_date)
    r_t_val_df = window_df[window_df['val'] == 1]
    window_df = window_df[window_df['val'] == 0]

    # Training normal data
    df_normal = window_df[window_df["Label"] == 0]
    # shuffle normal data
    df_normal = df_normal.sample(frac=1, random_state=42).reset_index(drop=True)
    train_len = train_size_t

    train_normal_t = df_normal[:train_len]
    print("Target training size {}".format(len(train_normal_t)))
    temp = train_normal_t[:]
    # 对于目标域前1000个日志序列，经过99次扩充到了100,000个数据，与源域数量保持一致。
    for _ in range(times):
        train_normal_t = pd.concat([train_normal_t, temp])

    # Testing normal data
    test_normal_t = df_normal[train_len:]
    print("Target test normal size {}".format(len(test_normal_t)))

    # Testing abnormal data
    test_abnormal_t = window_df[window_df["Label"] == 1]
    print("Target test abnormal size {}".format(len(test_abnormal_t)))

    return train_normal_s, test_normal_s, test_abnormal_s, r_s_val_df, \
           train_normal_t, test_normal_t, test_abnormal_t, r_t_val_df, w2v


if __name__ == '__main__':
    # df_source = pd.read_csv('../Dataset/Thunderbird.log_structured.csv')
    # df_target = pd.read_csv('../Dataset/BGL.log_structured.csv')
    from main_LogTP import arg_parser

    # import main_LogTP.

    parser = arg_parser()
    args = parser.parse_args()

    options = vars(args)
    print(options["window_size"])
    # get_datasets(df_source, df_target, options)
