from code.DatasetLoader import DatasetLoader
from code.MethodBertComp import GraphBertConfig
from code.MethodGraphBertNodeClassification import MethodGraphBertNodeClassification
from code.ResultSaving import ResultSaving
from code.Settings import Settings
import numpy as np
import torch
import pickle

#---- 'cora' , 'citeseer', 'pubmed' ----
pretrain = 0
dataset_name = 'DGraphFin'

np.random.seed(1)
torch.manual_seed(1)

#---- cora-small is for debuging only ----
if dataset_name == 'DGraphFin':
    nclass = 2
    nfeature = 18
    ngraph = 3546
if dataset_name == 'elliptic':
    nclass = 2
    nfeature = 166
    ngraph = 2147
#---- Fine-Tuning Task 1: Graph Bert Node Classification (Cora, Citeseer, and Pubmed) ----
if 1:
    #---- hyper-parameters ----
    if dataset_name == 'elliptic':
        k = 11
        lr = 0.001
        max_epoch = 200
    elif dataset_name == 'DGraphFin':
        k = 3
        lr = 0.001
        max_epoch = 200

    x_size = nfeature
    hidden_size = intermediate_size = 18
    num_attention_heads = 2
    num_hidden_layers = 1
    y_size = nclass
    graph_size = ngraph
    residual_type = 'graph_raw'
    # --------------------------

    print('************ Start ************')
    print('GraphBert, dataset: ' + dataset_name + ', residual: ' + residual_type + ', k: ' + str(k) + ', hidden dimension: ' + str(hidden_size) +', hidden layer: ' + str(num_hidden_layers) + ', attention head: ' + str(num_attention_heads))
    # ---- objection initialization setction ---------------
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
    data_obj.dataset_name = dataset_name
    data_obj.k = k
    data_obj.load_all_tag = True

    bert_config = GraphBertConfig(residual_type = residual_type, k=k, x_size=nfeature, y_size=y_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers)
    if(pretrain==0): 
      method_obj = MethodGraphBertNodeClassification(bert_config)
    elif(pretrain==1):
      model_init = MethodGraphBertNodeClassification.from_pretrained('./result/PreTrained_GraphBert/' + dataset_name + '/node_classification_pretrained_model/', ignore_mismatched_sizes=True)
      pretrained_bert_weights = model_init.bert.state_dict()
      method_obj = MethodGraphBertNodeClassification(bert_config)
      method_obj.bert.load_state_dict(pretrained_bert_weights)
    #---- set to false to run faster ----
    method_obj.spy_tag = True
    method_obj.max_epoch = max_epoch
    method_obj.lr = lr

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = './result/GraphBert/'
    # result_obj.result_destination_file_name = dataset_name + '_' + str(num_hidden_layers)
    result_obj.result_destination_file_name = 'with_gru' + '_' + str(k) + '_' + str(num_attention_heads) + '_' + str(hidden_size)# + '_summ' + str(100)
    # result_obj.result_destination_file_name = 'only_bert' + '_' + str(k) + '_' + str(num_attention_heads) + '_' + str(num_hidden_layers) 

    setting_obj = Settings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate()
    # ------------------------------------------------------
    #######
    f = open(result_obj.result_destination_folder_path + result_obj.result_destination_file_name, 'rb')
    result = pickle.load(f)
    f.close()

    print(result[199]['f1_epoch'])
    best_epoch = max(result, key=lambda epoch: result[epoch]['total_f1'])
    print("F1 score at best epoch ({}): {}".format(best_epoch, result[best_epoch]['f1_epoch']))
    print("AUC score at best epoch ({}): {}".format(best_epoch, result[best_epoch]['prec_epoch']))
    print("AP score at best epoch ({}): {}".format(best_epoch, result[best_epoch]['recall_epoch']))
    #######
    method_obj.save_pretrained('./result/PreTrained_GraphBert/' + dataset_name + '/node_classification_complete_model/')
    print('************ Finish ************')
#------------------------------------