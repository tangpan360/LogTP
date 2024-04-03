import os

import numpy as np
import pandas as pd

from . import DomainAdversarial
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import time
from utils.utils import epoch_time, get_dist, get_center, get_iter, dist2label
from gensim.models import Word2Vec
from sklearn import metrics


class LogTAD(nn.Module):
    def __init__(self, options):
        super(LogTAD, self).__init__()
        self.emb_dim = options["emb_dim"]
        self.hid_dim = options["hid_dim"]
        self.output_dim = options["out_dim"]
        self.n_layers = options["n_layers"]
        self.dropout = options["dropout"]
        self.bias = options["bias"]
        self.device = options["device"]
        self.weight_decay = options["weight_decay"]
        self.window_size = options["window_size"]
        self.step_size = options["step_size"]
        self.encoder = DomainAdversarial.DA_LSTM(self.emb_dim, self.hid_dim, self.output_dim, self.n_layers,
                                                 self.dropout, self.bias).to(self.device)
        self.optimizer = optim.Adam(self.encoder.parameters(), weight_decay=self.weight_decay)
        self.alpha = options["alpha"]
        self.max_epoch = options["max_epoch"]
        self.eps = options["eps"]
        self.source_dataset_name = options["source_dataset_name"]
        self.target_dataset_name = options["target_dataset_name"]
        self.test_ratio = options["test_ratio"]
        self.loss_mse = nn.MSELoss()
        self.loss_cel = nn.CrossEntropyLoss()
        self.w2v = None
        self.center = None
        self.early_stopping = False
        self.epochs_no_improve = 0
        self.n_epochs_stop = options["n_epochs_stop"]
        self.log = {
            "train": {
                key: [] for key in ["epoch", "loss"]
            },
            "valid": {
                key: [] for key in ["epoch", "loss"]
            }
        }
        self.current_dir = os.getcwd()
        self.loss_dir = self.current_dir + options["loss_dir"]
        print("test dir1: ", self.loss_dir)
        self.batch_size = options["batch_size"]

    def _train(self, iterator, center):

        print("test dir2: ", self.loss_dir)
        # TODO 前向和反向传播处理不走域判别的网路路径
        self.encoder.train()

        epoch_loss = 0

        for (i, batch) in enumerate(iterator):
            src = batch[0].to(self.device)
            # domain_label = batch[1].to(self.device)
            labels = batch[2]
            self.optimizer.zero_grad()
            # output, y_d = self.encoder(src, self.alpha)
            output = self.encoder(src, self.alpha)

            # domain_label = domain_label.view(-1)
            center = center.to(self.device)

            mse = 0
            for (ind, val) in enumerate(output):
                if labels[ind] == 1:
                    mse += (10 - self.loss_mse(val, center))
                else:
                    mse += self.loss_mse(val, center)
            # cel = self.loss_cel(y_d, domain_label.to(dtype=torch.long))
            # loss = mse * 10e4 + cel
            loss = mse
            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()

            center.cpu()
            src.cpu()
            # domain_label.cpu()
            output.cpu()
            # y_d.cpu()
        self.log['train']['loss'].append(epoch_loss / len(iterator))

        return epoch_loss / len(iterator)

    def _evaluate(self, iterator, center, epoch):
        self.log['valid']['epoch'].append(epoch)

        self.encoder.eval()

        epoch_loss = 0

        lst_dist = []

        lst_mse = []
        # lst_cel = []

        with torch.no_grad():
            for (i, batch) in enumerate(iterator):
                src = batch[0].to(self.device)
                domain_label = batch[1].to(self.device)
                labels = batch[2]
                # output, y_d = self.encoder(src, self.alpha)
                output = self.encoder(src, self.alpha)
                if i == 0:
                    lst_emb = output
                else:
                    lst_emb = torch.cat((lst_emb, output), dim=0)

                # domain_label = domain_label.view(-1)

                center = center.to(self.device)

                mse = 0
                for (ind, val) in enumerate(output):
                    if labels[ind] == 1:
                        mse += (10 - self.loss_mse(val, center))
                    else:
                        mse += self.loss_mse(val, center)

                # cel = self.loss_cel(y_d, domain_label.to(dtype=torch.long))

                lst_mse.append(mse.detach().cpu().numpy())
                # lst_cel.append(cel.detach().cpu().numpy())

                # loss = mse * 10e4 + cel
                loss = mse

                epoch_loss += loss.item()

                lst_dist.extend(get_dist(output, center))

                src.cpu()
                # domain_label.cpu()
                lst_emb.cpu()
                output.cpu()
                # y_d.cpu()
        if epoch < 10:
            center = get_center(lst_emb)
            print('get center:', center)
            center[(abs(center) < self.eps) & (center < 0)] = -self.eps
            center[(abs(center) < self.eps) & (center > 0)] = self.eps
            print('new center', center)

        print('\nmse: ', np.mean(np.array(lst_mse)))
        # print('cel: ', np.mean(np.array(lst_cel)))
        self.log['valid']['loss'].append(epoch_loss / len(iterator))
        return epoch_loss / len(iterator), center, lst_dist

    def save_log(self):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(self.loss_dir + '/' + key + "_log.csv", index=False)
            print("Log saved")
        except:
            print("Failed to save logs")

    def train_LogTAD(self, train_iter, eval_iter, w2v):
        best_eval_loss = float('inf')

        for epoch in tqdm(range(self.max_epoch)):
            if self.early_stopping:
                break

            if epoch == 0:
                center = torch.Tensor([0.0 for _ in range(self.hid_dim)])
            if epoch > 9:
                center = fixed_center
            start_time = time.time()
            self.log['train']['epoch'].append(epoch)
            train_loss = self._train(train_iter, center)

            eval_loss, center, _ = self._evaluate(eval_iter, center, epoch)

            if epoch == 9:
                fixed_center = center

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if eval_loss < best_eval_loss and epoch >= 9:
                best_eval_loss = eval_loss
                # 此处注意存储模型权重的相对位置，是“当前目录”下的saved_model，这个“当前目录”是指调用该文件的主文件
                torch.save(self.encoder.state_dict(),
                           f'./saved_model/{self.source_dataset_name}-{self.target_dataset_name}.pt')

                self.center = fixed_center.cpu()
                pd.DataFrame(fixed_center.cpu().numpy()).to_csv(
                    f'./saved_model/{self.source_dataset_name}-{self.target_dataset_name}_center.csv')
                self.epochs_no_improve = 0

            if eval_loss >= best_eval_loss and epoch >= 9:
                self.epochs_no_improve += 1

            if self.epochs_no_improve == self.n_epochs_stop:
                self.early_stopping = True
                print("Early stopping")

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss: .10f}')
            print(f'\tVal. Loss: {eval_loss: .10f}')
        self.w2v = w2v
        w2v.save(f'./saved_model/{self.source_dataset_name}-{self.target_dataset_name}_w2v.bin')
        self.save_log()

    def load_model(self):
        self.w2v = Word2Vec.load(f'./saved_model/{self.source_dataset_name}-{self.target_dataset_name}_w2v.bin')
        self.encoder.load_state_dict(
            torch.load(f'./saved_model/{self.source_dataset_name}-{self.target_dataset_name}.pt'))
        self.encoder.to(self.device)
        self.center = torch.Tensor(
            pd.read_csv(f'./saved_model/{self.source_dataset_name}-{self.target_dataset_name}_center.csv',
                        index_col=0).iloc[:, 0]
        )

    def _test(self, iterator):
        self.encoder.eval()

        y = []
        lst_dist = []

        with torch.no_grad():
            for (i, batch) in enumerate(iterator):
                src = batch[0].to(self.device)
                label = batch[2]
                output, _ = self.encoder(src, self.alpha)
                for j in label:
                    y.append(int(j))
                lst_dist.extend(get_dist(output, self.center))
                src.cpu()
        return y, lst_dist

    def get_best_r(self, iterator, steps=100):
        y, lst_dist = self._test(iterator)
        df = pd.DataFrame()
        df['label'] = y
        df['dist'] = lst_dist
        print(df.groupby(['label']).describe())
        mean_normal = np.mean(df['dist'].loc[df['label'] == 0])
        mean_abnormal = np.mean(df['dist'].loc[df['label'] == 1])
        step_len = (mean_abnormal - mean_normal) / steps
        best_r = 0
        best_auc = -1
        R = mean_normal
        for i in range(steps):
            y_pre = dist2label(lst_dist, R)
            auc = metrics.roc_auc_score(y, y_pre)
            if auc > best_auc:
                best_r = R
                best_auc = auc
            R += step_len
        return best_r, best_auc

    def get_r_from_val(self, val_df):
        X = list(val_df.Embedding)
        X_new = []
        for i in X:
            temp = []
            for j in i:
                temp.extend(j)
            X_new.append(np.array(temp).reshape(self.window_size, self.emb_dim))
        y_d = list(val_df.target.values)
        y = list(val_df.Label.values)
        X = torch.tensor(X_new, requires_grad=False)
        y_d = torch.tensor(y_d).reshape(-1, 1).long()
        y = torch.tensor(y).reshape(-1, 1).long()
        iterator = get_iter(X, y_d, y, batch_size=self.batch_size)
        R, auc = self.get_best_r(iterator)
        return R, auc

    def testing(self, test_normal_df, test_abnormal_df, r, target=0):
        X = list(test_normal_df.Embedding.values[::int(1 / self.test_ratio)])
        X.extend(list(test_abnormal_df.Embedding.values[::int(1 / self.test_ratio)]))
        X_new = []
        for i in tqdm(X):
            temp = []
            for j in i:
                temp.extend(j)
            X_new.append(np.array(temp).reshape(self.window_size, self.emb_dim))
        y_d = list(test_normal_df.target.values[::int(1 / self.test_ratio)])
        y_d.extend(list(test_abnormal_df.target.values[::int(1 / self.test_ratio)]))
        y = list(test_normal_df.Label.values[::int(1 / self.test_ratio)])
        y.extend(list(test_abnormal_df.Label.values[::int(1 / self.test_ratio)]))
        X_test = torch.tensor(X_new, requires_grad=False)
        y_d_test = torch.tensor(y_d).reshape(-1, 1).long()
        y_test = torch.tensor(y).reshape(-1, 1).long()
        test_iter = get_iter(X_test, y_d_test, y_test, batch_size=self.batch_size)
        y, lst_dist = self._test(test_iter)
        y_pred = dist2label(lst_dist, r)
        if target:
            print(f'Testing result for {self.target_dataset_name}:\n')
        else:
            print(f'Testing result for {self.source_dataset_name}:\n')

        print('Accuracy:', metrics.accuracy_score(y, y_pred))
        print(metrics.classification_report(y, y_pred, digits=5))
