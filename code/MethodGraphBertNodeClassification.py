import torch
import torch.nn.functional as F
import torch.optim as optim

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from code.MethodGraphBert import MethodGraphBert
from code.Summarize import SummarizeLayer

import time
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from code.EvaluateAcc import EvaluateAcc
from code.EvaluateOtherAcc import EvaluateOtherAcc

# from sklearn.decomposition import PCA
# pca = PCA(n_components=1)
BertLayerNorm = torch.nn.LayerNorm

class MethodGraphBertNodeClassification(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    spy_tag = True

    load_pretrained_path = ''
    save_pretrained_path = ''

    def __init__(self, config):
        super(MethodGraphBertNodeClassification, self).__init__(config)
        self.config = config
        self.bert = MethodGraphBert(config)
        self.gru_cell = torch.nn.GRUCell(input_size = config.hidden_size, hidden_size=config.hidden_size)
        self.res_h = torch.nn.Linear(config.x_size, config.hidden_size)
        self.res_y = torch.nn.Linear(config.x_size, config.y_size)
        self.cls_y = torch.nn.Linear(config.hidden_size, config.y_size)
        self.summarize = SummarizeLayer()
        self.init_weights()
        
    def forward(self, raw_features, idx=None, timestep = 0, hidden_state = None):
        bert_w = 0.5
        gru_w = 0.5
        summ = 100
        residual_h, residual_y = self.residual_term(timestep)
        if idx is not None:
            if residual_h is None:
                outputs = self.bert(raw_features[idx], residual_h=None)
            else:
                outputs = self.bert(raw_features[idx], residual_h=residual_h[idx])
                residual_y = residual_y[idx]
        else:
            if residual_h is None:
                outputs = self.bert(raw_features, residual_h=None)
            else:
                outputs = self.bert(raw_features, residual_h=residual_h)
        #1 BERT Transformer context vectors
        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output += outputs[0][:,i,:]
        sequence_output /= float(self.config.k+1)
        #2 GRU time information
        #FOR mean pooling
        meanSummary = sequence_output.detach().mean(dim=0,keepdim=True) 
        # FOR Attention based summarize pooling
        # summary = self.summarize(sequence_output.detach(),summ) 
        # meanSummary = summary.mean(dim=0, keepdim=True)
        ###FOR PCA based pooling
        # meanSummary = torch.tensor(pca.fit_transform(sequence_output.detach().numpy().T)).T 
        hidden_state = self.gru_cell(meanSummary, hidden_state)
        
        labels = self.cls_y((bert_w*sequence_output) + (gru_w*hidden_state)) ### 50-50 weights for BERT context and GRU time information
        if residual_y is not None:
            labels += residual_y

        return F.log_softmax(labels, dim=1)

    def residual_term(self, timestep):
        if self.config.residual_type == 'none':
            return None, None
        elif self.config.residual_type == 'raw':
            return self.res_h(self.data['X'][timestep]), self.res_y(self.data['X'][timestep])
        elif self.config.residual_type == 'graph_raw':
            return torch.spmm(self.data['A'][timestep], self.res_h(self.data['X'][timestep])), torch.spmm(self.data['A'][timestep], self.res_y(self.data['X'][timestep]))

    def train_model(self, max_epoch):
        t_begin = time.time()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        accuracy = EvaluateAcc('', '')
        otherAcc = EvaluateOtherAcc('', '')
        max_score = 0.0
        for epoch in range(max_epoch):
            t_epoch_begin = time.time()

            # -------------------------

            self.train()
            optimizer.zero_grad()
            loss_train = 0.0  
            acc_train = 0.0
            hidden_state = torch.zeros(1, self.config.hidden_size)
            for i in self.data['idx_train']:
              i = i.item()
              output = self.forward(self.data['raw_embeddings'][i-1],idx=None, timestep = i-1, hidden_state = hidden_state)
              loss_train += F.cross_entropy(output, self.data['y'][i-1])
              accuracy.data = {'true_y': self.data['y'][i-1], 'pred_y': output.max(1)[1]}
              acc_train += accuracy.evaluate()

            loss_train.backward()
            optimizer.step()
            acc_train = acc_train / self.data['idx_train'][-1].item()

            self.eval()
            # output = self.forward(self.data['raw_embeddings'], self.data['idx_val'])

            # loss_val = F.cross_entropy(output, self.data['y'][self.data['idx_val']])
            # accuracy.data = {'true_y': self.data['y'][self.data['idx_val']],
            #                  'pred_y': output.max(1)[1]}
            # acc_val = accuracy.evaluate()

            #-------------------------
            #---- keep records for drawing convergence plots ----
            loss_test = 0.0
            acc_test = 0.0
            prec = 0.0
            recall = 0.0
            f1_list = []
            prec_list = []
            recall_list = []
            total_true = torch.Tensor()
            total_pred = torch.Tensor()
            total_f1 = 0.0
            total_prec = 0.0
            total_recall = 0.0
            for i in self.data['idx_test']:
              i = i.item()
              output = self.forward(self.data['raw_embeddings'][i-1], idx = None, timestep = i-1)
              loss_test += F.cross_entropy(output, self.data['y'][i-1])
              accuracy.data = {'true_y': self.data['y'][i-1],
                              'pred_y': output.max(1)[1]}
              otherAcc.data = {'true_y': self.data['y'][i-1],
                              'pred_y': output.max(1)[1]}
              total_true = torch.cat((total_true, self.data['y'][i-1]), dim=0)
              total_pred = torch.cat((total_pred, output.max(1)[1]), dim=0)
              auc, ap = otherAcc.evaluate()
              f1_list.append(accuracy.evaluate())
              prec_list.append(auc)
              recall_list.append(ap)
              acc_test += accuracy.evaluate()
            acc_test = acc_test / (50.0 - self.data['idx_test'][0].item())
            accuracy.data = {'true_y': total_true,
                              'pred_y': total_pred}
            otherAcc.data = {'true_y': total_true,
                              'pred_y': total_pred}
            total_f1 = accuracy.evaluate()
            total_auc, total_ap = otherAcc.evaluate()
            
            self.learning_record_dict[epoch] = {'loss_train': loss_train.item(), 'acc_train': acc_train,
                                                'loss_test': loss_test.item(), 'acc_test': acc_test,
                                                'time': time.time() - t_epoch_begin, 'f1_epoch': f1_list,
                                                'prec_epoch': prec_list, 'recall_epoch': recall_list,
                                                'total_f1': total_f1, 'total_prec': total_auc, 
                                                'total_recall': total_ap, 'micro_f1': f1_score(total_true, total_pred, average='micro'),
                                                'weighted_f1': f1_score(total_true, total_pred, average='weighted')
                                                }

            # -------------------------
            if epoch % 10 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train),
                      'loss_test: {:.4f}'.format(loss_test.item()),
                      'acc_test: {:.4f}'.format(acc_test),
                      'time: {:.4f}s'.format(time.time() - t_epoch_begin))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_begin) + ', best testing performance {: 4f}'.format(np.max([self.learning_record_dict[epoch]['acc_test'] for epoch in self.learning_record_dict])) + ', minimun loss {: 4f}'.format(np.min([self.learning_record_dict[epoch]['loss_test'] for epoch in self.learning_record_dict])))
        return time.time() - t_begin, np.max([self.learning_record_dict[epoch]['acc_test'] for epoch in self.learning_record_dict])

    def run(self):

        self.train_model(self.max_epoch)

        return self.learning_record_dict