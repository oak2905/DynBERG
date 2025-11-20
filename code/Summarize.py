import torch
import torch.nn as nn
import torch.nn.functional as F

class SummarizeLayer(nn.Module):
    def __init__(self):
        super(SummarizeLayer, self).__init__()
        self.p = None

    def forward(self, x, k):
        """
        x: Input tensor of shape (num_subgraphs, feature_dim)
        k: Number of top-k subgraphs to consider
        Returns: out: Summarized tensor after top-k selection
        """
        _, feature_dim = x.size()
        if self.p is None:
            self.p = nn.Parameter(torch.Tensor(feature_dim))
            nn.init.uniform_(self.p)

        y = torch.matmul(x, self.p) / torch.norm(self.p)
        
        top_y, top_indices = torch.topk(y, k)

        out = torch.gather(x, 0, top_indices.unsqueeze(-1).expand(-1, x.size(1)))
        out = out * torch.tanh(top_y).unsqueeze(-1)
        return out