import torch
from baselines.modules.layers import CWNN, GNN
from torch_geometric.utils import scatter

class CellNetwork(torch.nn.Module):
    def __init__(self, conv_layers, dropout=0.0):
        super(CellNetwork, self).__init__()
        self.gnn = GNN(conv_layers, dropout=dropout)
        self.cwnn = CWNN(conv_layers)
        
    def forward(self, data):
        x = self.gnn(data["x"], data["edges"]).relu()
        xe = self.cwnn(data["xe"], data["Ldo"], data["Lup"]).relu()
        xed = scatter(xe, data["row"], dim=0, dim_size=x.shape[0], reduce="sum") + scatter(
            xe, data["col"], dim=0, dim_size=x.shape[0], reduce="sum"
        )
        x = torch.cat([x, xed], dim=-1)
        return x