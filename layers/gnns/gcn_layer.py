import torch.nn as nn
from torch_geometric.nn import GCNConv

from layers.gnns.gnn_interface import GNNFactoryInterface
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(
        self, in_features, out_features, activation, batch_norm, residual=True
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()

        self.residual = residual
        self.conv = GCNConv(in_features, out_features, add_self_loops=False)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.batchnorm(h)
        h = self.activation(h)
        if self.residual:
            h = h + x
        return h

class GcnCreator(GNNFactoryInterface):
    def __init__(self, hidden_dim, batch_norm):
        self.hidden_dim = hidden_dim
        self.batch_norm = batch_norm

    def return_gnn_instance(self, is_last=False):
        return GCNLayer(
            self.hidden_dim,
            self.hidden_dim,
            nn.Identity() if is_last else F.relu,
            batch_norm=self.batch_norm,
        )