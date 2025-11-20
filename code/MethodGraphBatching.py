'''
Concrete MethodModule class for a specific learning MethodModule
'''

from code.base_class.method import method


class MethodGraphBatching(method):
    data = None
    k = 5

    def run(self):
        top_k_intimacy_time = []
        for i in range(1,50):
          S = self.data['S'][i-1]
          index_id_dict = self.data['index_id_map'][i-1]
          user_top_k_neighbor_intimacy_dict = {}
          for node_index in index_id_dict:
              node_id = index_id_dict[node_index]
              s = S[node_index]
              s[node_index] = -1000.0
              top_k_neighbor_index = s.argsort()[-self.k:][::-1]
              user_top_k_neighbor_intimacy_dict[node_id] = []
              for neighbor_index in top_k_neighbor_index:
                  neighbor_id = index_id_dict[neighbor_index]
                  user_top_k_neighbor_intimacy_dict[node_id].append((neighbor_id, s[neighbor_index]))
          top_k_intimacy_time.append(user_top_k_neighbor_intimacy_dict)
        return top_k_intimacy_time




