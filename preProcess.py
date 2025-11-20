import numpy as np
import torch

from code.DatasetLoader import DatasetLoader
from code.MethodGraphBatching import MethodGraphBatching
from code.ResultSaving import ResultSaving
from code.Settings import Settings

#---- 'cora' , 'citeseer', 'pubmed' ----

dataset_name = 'DGraphFin'

np.random.seed(1)
torch.manual_seed(1)

#---- cora-small is for debuging only ----
if dataset_name == 'cora-small':
    nclass = 7
    nfeature = 1433
    ngraph = 10
elif dataset_name == 'cora':
    nclass = 7
    nfeature = 1433
    ngraph = 2708
elif dataset_name == 'citeseer':
    nclass = 6
    nfeature = 3703
    ngraph = 3312
elif dataset_name == 'pubmed':
    nclass = 3
    nfeature = 500
    ngraph = 19717
elif dataset_name == 'DGraphFin':
    nclass = 2
    nfeature = 18
    ngraph = 3546
elif dataset_name == 'elliptic':
    nclass = 2
    nfeature = 166
    ngraph = 2147


#---- Step 1: intimacy calculation and subgraph batching ----
if 1:
    for k in [3,4,5]:
        print('************ Start ************')
        print('Subgraph Batching, dataset: ' + dataset_name + ', k: ' + str(k))
        # ---- objection initialization setction ---------------
        data_obj = DatasetLoader()
        data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
        data_obj.dataset_name = dataset_name
        data_obj.compute_s = True

        method_obj = MethodGraphBatching()
        method_obj.k = k

        result_obj = ResultSaving()
        result_obj.result_destination_folder_path = './result/Batch/'
        result_obj.result_destination_file_name = dataset_name + '_' + str(k)

        setting_obj = Settings()

        evaluate_obj = None
        # ------------------------------------------------------

        # ---- running section ---------------------------------
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        setting_obj.load_run_save_evaluate()
        # ------------------------------------------------------

        print('************ Finish ************')
#------------------------------------
