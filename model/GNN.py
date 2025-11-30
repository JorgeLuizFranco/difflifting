from typing import Optional

import torch
import torch.nn as nn
from torch.nn import (
    BatchNorm1d,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch_geometric.nn import GINConv
from torch_geometric.nn import GPSConv
from torch_geometric.nn.attention import PerformerAttention


class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,n_layers_gnn , task="classification"):
        super(GIN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.initialization = GINConv(
            torch.nn.Sequential(
                Linear(in_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BatchNorm1d(hidden_channels),
            ),
            eps=0.,
            train_eps=False)
        self.mp_layers = torch.nn.ModuleList()
        for i in range(n_layers_gnn - 1):
            self.mp_layers.append(
                GINConv(
                    torch.nn.Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BatchNorm1d(hidden_channels),
                    ),
                    eps=0.,
                    train_eps=False)
            )



    def forward(self, data ):
        x, edge_index, batch =  data.x, data.edge_index, data.batch
        x = self.initialization(x, edge_index)
        for conv in self.mp_layers:
            x = conv(x, edge_index)
        return x



class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1



class GPS(torch.nn.Module):
    def __init__(self,node_embedding_dim, hidden_channels: int,  pe_channels,edge_embedding_dim=4,  pe_dim: int=8, num_layers: int=5,
                 is_zinc=False,attn_type: str="multihead"):
        super().__init__()

        self.node_emb = Linear(node_embedding_dim, hidden_channels - pe_dim)
        self.pe_lin = Linear(pe_channels, pe_dim)
        self.pe_norm = BatchNorm1d(pe_channels)
        self.edge_emb = Linear(edge_embedding_dim, hidden_channels)
        self.is_zinc = is_zinc
        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = torch.nn.Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
            )
            conv = GPSConv(hidden_channels, GINConv(nn), heads=4,
                           attn_type="multihead")
            self.convs.append(conv)

        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, data):
        x, pe, edge_index, edge_attr, batch = data.x.float() , data.pe, data.edge_index, data.edge_attr, data.batch
        x_pe = self.pe_norm(pe)
        if self.is_zinc:
            x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
        else:
            x = torch.cat((self.node_emb(x).squeeze(-1), self.pe_lin(x_pe)), 1)

        for conv in self.convs:
            x = conv(x, edge_index, batch)
        return x