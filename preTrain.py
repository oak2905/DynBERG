import numpy as np
import torch

from code.DatasetLoader import DatasetLoader
from code.MethodBertComp import GraphBertConfig
from code.MethodGraphBertNodeConstruct import MethodGraphBertNodeConstruct
from code.ResultSaving import ResultSaving
from code.Settings import Settings

#---- 'cora' , 'citeseer', 'pubmed' ----

dataset_name = 'DGraphFin'

np.random.seed(1)
torch.manual_seed(1)

#---- cora-small is for debuging only ----

if dataset_name == 'elliptic':
    nclass = 2
    nfeature = 166
    ngraph = 2147
elif dataset_name == 'DGraphFin':
    nclass = 2
    nfeature = 18
    ngraph = 3546

#---- Pre-Training Task #1: Graph Bert Node Attribute Reconstruction (Cora, Citeseer, and Pubmed) ----
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
    hidden_size = intermediate_size = 32
    num_attention_heads = 4
    num_hidden_layers = 7
    y_size = nclass
    graph_size = ngraph
    residual_type = 'graph_raw'
    # --------------------------

    print('************ Start ************')
    print('GraphBert, dataset: ' + dataset_name + ', Pre-training, Node Attribute Reconstruction.')
    # ---- objection initialization setction ---------------
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
    data_obj.dataset_name = dataset_name
    data_obj.k = k
    data_obj.load_all_tag = True
  
    bert_config = GraphBertConfig(residual_type = residual_type, k=k, x_size=nfeature, y_size=y_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers)
    method_obj = MethodGraphBertNodeConstruct(bert_config)
    method_obj.max_epoch = max_epoch
    method_obj.lr = lr

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = './result/GraphBert/'
    result_obj.result_destination_file_name = dataset_name + '_' + str(k) + '_node_reconstruction'

    setting_obj = Settings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate()
    # ------------------------------------------------------

    method_obj.save_pretrained('./result/PreTrained_GraphBert/' + dataset_name + '/node_classification_pretrained_model/')
    print('************ Finish ************')
#------------------------------------