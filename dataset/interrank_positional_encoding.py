from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform, AddRandomWalkPE
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
    to_scipy_sparse_matrix,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)


def add_node_attr(data: Data, value: Any, attr_name: Optional[str] = None) -> Data:
    """Adiciona o valor ao atributo da estrutura de dados."""
    if attr_name is None:
        if data.x is not None:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value
    return data


class AddRandomWalkPEInterrank(BaseTransform):
    def __init__(self, walk_length: int, attr_name: Optional[str] = 'random_walk_pe') -> None:
        self.walk_length = walk_length
        self.attr_name = attr_name

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        row, col = data.edge_index
        N = data.num_nodes
        assert N is not None

        if data.edge_weight is None:
            value = torch.ones(data.num_edges, device=row.device)
        else:
            value = data.edge_weight
        value = scatter(value, row, dim_size=N, reduce='sum').clamp(min=1)[row]
        value = 1.0 / value

        if N <= 2_000:  # Dense code path for faster computation:
            adj = torch.zeros((N, N), device=row.device)
            adj[row, col] = value
            loop_index = torch.arange(N, device=row.device)
        else:
            adj = to_torch_csr_tensor(data.edge_index, value, size=data.size())

        def get_pe(out: Tensor) -> Tensor:
            loop_index = torch.arange(N, device=out.device)
            return out[loop_index, loop_index]

        out = adj
        pe_list = [get_pe(out)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            pe_list.append(get_pe(out))

        pe = torch.stack(pe_list, dim=-1)
        data = add_node_attr(data, pe, attr_name=self.attr_name)
        return data


def apply_positional_encoding(data: Data) -> Data:
    # 1. Converta as representações de adjacência para o formato `edge_index`
    adjacency_1_edge_index = to_edge_index(data.adjacency_1)
    adjacency_0_edge_index = to_edge_index(data.adjacency_0)

    # 2. Aplique a transformação para cada uma das representações de adjacência
    pe_transform_1 = AddRandomWalkPE(walk_length=5)  # Ajuste o valor de `walk_length` conforme necessário
    data_pe_1 = pe_transform_1(data.clone())  # Clone para preservar os dados originais
    data_pe_1.edge_index = adjacency_1_edge_index  # Aplique o edge_index da adjacency_1

    pe_transform_0 = AddRandomWalkPE(walk_length=5)
    data_pe_0 = pe_transform_0(data.clone())  # Clone novamente para preservar os dados
    data_pe_0.edge_index = adjacency_0_edge_index  # Aplique o edge_index da adjacency_0

    return data_pe_1, data_pe_0