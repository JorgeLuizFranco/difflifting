import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from utils import set_seed

seeds = [42, 9, 3]
dataset = TUDataset(root='.', name='MUTAG').shuffle()


class GCN(torch.nn.Module):
        """GCN"""

        def __init__(self, in_channels, dim_h, out_channels):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(out_channels, dim_h)
            self.conv2 = GCNConv(dim_h, dim_h)
            self.conv3 = GCNConv(dim_h, dim_h)
            self.lin = Linear(dim_h, out_channels)

        def forward(self, x, edge_index, batch):
            # Node embeddings
            h = self.conv1(x, edge_index)
            h = h.relu()
            h = self.conv2(h, edge_index)
            h = h.relu()
            h = self.conv3(h, edge_index)

            # Graph-level readout
            hG = global_mean_pool(h, batch)

            # Classifier
            h = F.dropout(hG, p=0.5, training=self.training)
            h = self.lin(h)

            return hG, F.log_softmax(h, dim=1)


class GIN(torch.nn.Module):
    """GIN"""

    def __init__(self,in_channels, dim_h, out_channels):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(in_channels, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h * 3, dim_h * 3)
        self.lin2 = Linear(dim_h * 3, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Node embeddings
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h

