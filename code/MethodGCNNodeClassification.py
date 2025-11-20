import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import numpy as np
from sklearn.metrics import f1_score

from code.EvaluateAcc import EvaluateAcc
from code.EvaluateOtherAcc import EvaluateOtherAcc


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=F.relu, dropout=0.1):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = torch.spmm(adj, x)  # Sparse matrix multiplication
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x


class GCNNodeClassification(nn.Module):
    lr = 0.01
    weight_decay = 5e-4
    max_epoch = 500
    learning_record_dict = {}

    def __init__(self, config):
        super(GCNNodeClassification, self).__init__()
        self.config = config
        self.gc1 = GCNLayer(config.x_size, config.hidden_size)
        self.gc2 = GCNLayer(config.hidden_size, config.y_size, activation=None, dropout=0.1)

        self.eval_acc = EvaluateAcc('', '')
        self.eval_other = EvaluateOtherAcc('', '')

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        return x

    def train_model(self, max_epoch):
        t_begin = time.time()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        max_score = 0.0

        for epoch in range(max_epoch):
            t_epoch_begin = time.time()
            self.train()
            optimizer.zero_grad()

            loss_train = 0.0
            acc_train = 0.0
            
            
            for i in self.data['idx_train']:
                i = i.item()
                x = self.data['X'][i-1]
                y = self.data['y'][i-1]
                adj = self.data['A'][i-1]
                
                output = self.forward(x, adj)
                loss = F.cross_entropy(output, y)
                loss.backward()
                loss_train += loss.item()

                self.eval_acc.data = {'true_y': y, 'pred_y': output.max(1)[1]}
                acc_train += self.eval_acc.evaluate()

            optimizer.step()
            acc_train /= len(self.data['idx_train'])

            # Test phase
            self.eval()
            loss_test = 0.0
            acc_test = 0.0
            total_true = torch.Tensor()
            total_pred = torch.Tensor()
            f1_list = []
            prec_list = []
            recall_list = []
            for i in self.data['idx_test']:
                i = i.item()
                x = self.data['X'][i-1]
                y = self.data['y'][i-1]
                adj = self.data['A'][i-1]

                output = self.forward(x, adj)
                loss = F.cross_entropy(output, y)
                loss_test += loss.item()

                self.eval_acc.data = {'true_y': y, 'pred_y': output.max(1)[1]}
                self.eval_other.data = {'true_y': y, 'pred_y': output.max(1)[1]}
                acc_test += self.eval_acc.evaluate()

                total_true = torch.cat((total_true, y), dim=0)
                total_pred = torch.cat((total_pred, output.max(1)[1]), dim=0)
                auc, ap = self.eval_other.evaluate()
                f1_list.append(self.eval_acc.evaluate())
                prec_list.append(auc)
                recall_list.append(ap)
            acc_test = acc_test / (50.0 - self.data['idx_test'][0].item())

            self.eval_acc.data = {'true_y': total_true, 'pred_y': total_pred}
            self.eval_other.data = {'true_y': total_true, 'pred_y': total_pred}
            total_f1 = self.eval_acc.evaluate()
            total_auc, total_ap = self.eval_other.evaluate()

            self.learning_record_dict[epoch] = {'loss_train': loss_train, 'acc_train': acc_train,
                                                'loss_test': loss_test, 'acc_test': acc_test,
                                                'time': time.time() - t_epoch_begin, 'f1_epoch': f1_list,
                                                'prec_epoch': prec_list, 'recall_epoch': recall_list,
                                                'total_f1': total_f1, 'total_prec': total_auc, 
                                                'total_recall': total_ap, 'micro_f1': f1_score(total_true, total_pred, average='micro'),
                                                'weighted_f1': f1_score(total_true, total_pred, average='weighted')
                                                }

            # -------------------------
            if epoch % 10 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train),
                      'acc_train: {:.4f}'.format(acc_train),
                      'loss_test: {:.4f}'.format(loss_test),
                      'acc_test: {:.4f}'.format(acc_test),
                      'time: {:.4f}s'.format(time.time() - t_epoch_begin))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_begin) + ', best testing performance {: 4f}'.format(np.max([self.learning_record_dict[epoch]['acc_test'] for epoch in self.learning_record_dict])) + ', minimun loss {: 4f}'.format(np.min([self.learning_record_dict[epoch]['loss_test'] for epoch in self.learning_record_dict])))
        return time.time() - t_begin, np.max([self.learning_record_dict[epoch]['acc_test'] for epoch in self.learning_record_dict])

    def run(self):
        self.train_model(self.max_epoch)
        return self.learning_record_dict