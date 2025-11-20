'''
Concrete IO class for a specific dataset
'''

from code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv
import pickle


class DatasetLoader(dataset):
    c = 0.15
    k = 5
    data = None
    batch_size = None

    dataset_source_folder_path = None
    dataset_name = None
  
    load_all_tag = False
    compute_s = False

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(DatasetLoader, self).__init__(dName, dDescription)

    def load_hop_wl_batch(self):
      
        # print('Load Subgraph Batches')
        f = open('./result/Batch/' + self.dataset_name + '_' + str(self.k), 'rb')
        batch_dict = pickle.load(f)
        f.close()

        # return hop_dict, wl_dict, batch_dict
        return batch_dict

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def adj_normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = sorted(set(labels), key=lambda x: int(float(x)))
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def load(self):
        """Load citation network dataset (cora only for now)"""
        print('Loading {} dataset...'.format(self.dataset_name))

        featuresList = []
        labelsList = []
        adjList = []
        eigen_adjList = []
        index_id_mapList = []
        edgesList = []
        raw_embeddingsList = []
        idxList = []
        idx_train = torch.LongTensor(range(1,35))
        idx_test = torch.LongTensor(range(35,50))
        for i in range(1,50):
          idx_features_labels = np.genfromtxt("{}/node_{}".format(self.dataset_source_folder_path, i), dtype=np.dtype(str))

          mask_node = np.isin(idx_features_labels[:, 0], idx_features_labels[idx_features_labels[:,-1]=='unknown'][:,0])

          features = sp.csr_matrix(idx_features_labels[~mask_node][:, 1:-1], dtype=np.float32)
          one_hot_labels = self.encode_onehot(idx_features_labels[~mask_node][:, -1])
          # build graph
          idx = np.array(idx_features_labels[~mask_node][:, 0], dtype=np.float64).astype(np.int32)
          idx_map = {j: i for i, j in enumerate(idx)}
          index_id_map = {i: j for i, j in enumerate(idx)}
          edges_unordered = np.genfromtxt("{}/link_{}".format(self.dataset_source_folder_path, i), dtype=np.float64).astype(np.int32)
    
          mask_link = np.isin(edges_unordered, idx_features_labels[idx_features_labels[:,-1]=='unknown'][:,0]).any(axis=1)

          edges_mapped = [(idx_map[u], idx_map[v]) for u, v in edges_unordered[~mask_link] if u in idx_map and v in idx_map]

          # Convert to NumPy array
          edges = np.array(edges_mapped, dtype=np.int32)
          adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                        shape=(one_hot_labels.shape[0], one_hot_labels.shape[0]),
                                        dtype=np.float32)
          
          eigen_adj = None
          if self.compute_s:
              eigen_adj = self.c * inv((sp.eye(adj.shape[0]) - (1 - self.c) * self.normalize(adj)).toarray())

          norm_adj = self.normalize(adj + sp.eye(adj.shape[0]))

          # if self.dataset_name == 'cora':
          #     idx_train = range(140)
          #     idx_test = range(200, 1200)
          #     idx_val = range(1200, 1500)
          # elif self.dataset_name == 'citeseer':
          #     idx_train = range(120)
          #     idx_test = range(200, 1200)
          #     idx_val = range(1200, 1500)
          #     #features = self.normalize(features)
          # elif self.dataset_name == 'pubmed':
          #     idx_train = range(60)
          #     idx_test = range(6300, 7300)
          #     idx_val = range(6000, 6300)
          # elif self.dataset_name == 'cora-small':
          #     idx_train = range(5)
          #     idx_val = range(5, 10)
          #     idx_test = range(5, 10)
          # elif self.dataset_name == 'elliptic': ############
          #     idx_train = range(int(0.5*adj.shape[0]))
          #     idx_val = range(int(0.5*adj.shape[0]), int(0.75*adj.shape[0]))
          #     idx_test = range(int(0.75*adj.shape[0]), int(1.0*adj.shape[0]))

          features = torch.FloatTensor(np.array(features.todense()))
          labels = torch.LongTensor(np.where(one_hot_labels)[1])
          adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

          if self.load_all_tag:
              batch_dict = self.load_hop_wl_batch()
              raw_feature_list = []
              for node in idx:
                  node_index = idx_map[node]
                  neighbors_list = batch_dict[i-1][node]
                  raw_feature = [features[node_index].tolist()]
                  for neighbor, intimacy_score in neighbors_list:
                      neighbor_index = idx_map[neighbor]
                      raw_feature.append(features[neighbor_index].tolist())

                  raw_feature_list.append(raw_feature)
              raw_embeddings = torch.FloatTensor(raw_feature_list)
          else:
              raw_embeddings = None

          featuresList.append(features)
          labelsList.append(labels)
          adjList.append(adj)
          eigen_adjList.append(eigen_adj)
          index_id_mapList.append(index_id_map)
          edgesList.append(edges_unordered[~mask_link])
          raw_embeddingsList.append(raw_embeddings)
          idxList.append(idx)
        return {'X': featuresList, 'A': adjList, 'S': eigen_adjList, 'index_id_map': index_id_mapList, 'edges': edgesList, 'raw_embeddings': raw_embeddingsList, 'y': labelsList, 'idx': idxList, 'idx_train': idx_train, 'idx_test': idx_test}
