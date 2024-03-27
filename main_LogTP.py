import os
from argparse import ArgumentParser
from utils.utils import set_seed, get_train_eval_iter, plot_train_valid_loss
from utils import preprocessing, SlidingWindow
import pandas as pd
import time
from model.LogTAD import LogTAD


def arg_parser():
    """
    Add parser parameters
    :return:
    """
    parser = ArgumentParser()
    parser.add_argument("--source_dataset_name", help="please choose source dataset name from BGL or Thunderbird", default="Thunderbird")
    # parser.add_argument("--source_dataset_name", help="please choose source dataset name from BGL or Thunderbird", default="BGL")
    parser.add_argument("--target_dataset_name", help="please choose target dataset name from BGL or Thunderbird", default="BGL")
    # parser.add_argument("--target_dataset_name", help="please choose target dataset name from BGL or Thunderbird", default="Thunderbird")
    parser.add_argument("--device", help="hardware device", default="cpu")
    parser.add_argument("--random_seed", help="random seed", default=42)
    parser.add_argument("--download_datasets", help="download datasets or not", default=0)
    parser.add_argument("--output_dir", metavar="DIR", help="output directory", default="/Dataset")
    parser.add_argument("--model_dir", metavar="DIR", help="output directory", default="/Dataset")

    # training parameters
    parser.add_argument("--max_epoch", help="epochs", default=100)
    parser.add_argument("--batch_size", help="batch size", default=1024)
    parser.add_argument("--lr", help="learning size", default=0.001)
    parser.add_argument("--weight_decay", help="weight decay", default=1e-6)
    parser.add_argument("--eps", help="minimum center value", default=0.1)
    parser.add_argument("--n_epochs_stop", help="n epochs stop if not improve in valid loss", default=10)
    parser.add_argument("--loss_dir", metavar="DIR", help="loss directory", default="/loss_dir")
    parser.add_argument("--saved_model", metavar="DIR", help="saved model dir", default="/saved_model")

    # word2vec parameters
    parser.add_argument("--emb_dim", help="word2vec vector size", default=300)

    # data preprocessing parameters
    parser.add_argument("--window_size", help="size of sliding window", default=20)
    parser.add_argument("--step_size", help="step size of sliding window", default=4)
    parser.add_argument("--train_size_s", help="source training size", default=100000)
    parser.add_argument("--train_size_t", help="target training size", default=1000)

    # LSTM parameters
    parser.add_argument("--hid_dim", help="hidden dimensions", default=128)
    parser.add_argument("--out_dim", help="output dimensions", default=2)
    parser.add_argument("--n_layers", help="layers of LSTM", default=2)
    parser.add_argument("--dropout", help="dropout", default=0.3)
    parser.add_argument("--bias", help="bias for LSTM", default=True)

    # gradient reversal parameters
    parser.add_argument("--alpha", help="alpha value for the gradient reversal layer", default=0.1)

    # test parameters
    parser.add_argument("--test_ratio", help="testing ratio", default=0.1)

    return parser


def main():
    parser = arg_parser()
    args = parser.parse_args()

    options = vars(args)

    # 设定随机种子 (utils/utils)
    set_seed(options["random_seed"])  # 设定随机种子
    print(f"Set seed: {options['random_seed']}")

    current_path = os.getcwd()
    loss_dir = current_path + options["loss_dir"]
    if not os.path.exists(loss_dir):
        print('Making directory for loss storage')
        os.makedirs(loss_dir)
    saved_model = current_path + options["saved_model"]
    if not os.path.exists(saved_model):
        print('Making directory for model storage')
        os.makedirs(saved_model)

    # 下载并利用Drain解析数据集 (preprocessing/parsing)
    path = "./Dataset"
    if options["download_datasets"] == 1:
        print("==============")
        print("下载并预处理数据集")
        print("==============")
        preprocessing.parsing(options['source_dataset_name'], output_dir=options['output_dir'])
        preprocessing.parsing(options['target_dataset_name'], output_dir=options['output_dir'])
        print("==============")
        print("完成数据下载和预处理")
        print("==============")
    elif options["download_datasets"] == 0:
        print("==============")
        print("不下载数据集")
        print("==============")

    # 利用pandas读取csv中的源域数据和目标域数据
    start_time = time.time()
    df_source = pd.read_csv(path + f'/{options["source_dataset_name"]}.log_structured.csv')
    print(f'Reading source dataset: {options["source_dataset_name"]} dataset')
    end_time = time.time()
    spend_time = end_time - start_time
    print(f'{options["source_dataset_name"]} spend {spend_time}s')

    start_time = time.time()
    df_target = pd.read_csv(path + f'/{options["target_dataset_name"]}.log_structured.csv')
    print(f'Reading target dataset: {options["target_dataset_name"]} dataset')
    end_time = time.time()
    spend_time = end_time - start_time
    print(f'{options["target_dataset_name"]} spend {spend_time}s')

    # 对源域数据和目标域数据进行序列划分处理获得数据集
    train_normal_s, test_normal_s, test_abnormal_s, r_s_val_df, train_normal_t, test_normal_t, test_abnormal_t, r_t_val_df, w2v = SlidingWindow.get_datasets(df_source, df_target, options, val_date="2005.11.15")

    # 划分train_iter和test_iter
    train_iter, test_iter = get_train_eval_iter(train_normal_s, train_normal_t, window_size=options["window_size"],
                                                emb_dim=options["emb_dim"], batch_size=options["batch_size"])

    # 加载模型
    demo_logtad = LogTAD(options)

    # 训练模型
    demo_logtad.train_LogTAD(train_iter, test_iter, w2v)

    # 加载模型权重
    demo_logtad.load_model()

    # 异常半径阈值
    R_src, _ = demo_logtad.get_r_from_val(r_s_val_df)
    R_trg, _ = demo_logtad.get_r_from_val(r_t_val_df)

    print(f'Starting to test source dataset: {options["source_dataset_name"]}')
    demo_logtad.testing(test_normal_s, test_abnormal_s, R_src, target=0)
    print(f'Starting to test target dataset: {options["target_dataset_name"]}')
    demo_logtad.testing(test_normal_t, test_abnormal_t, R_trg, target=1)

    plot_train_valid_loss(loss_dir)


if __name__ == '__main__':
    main()
