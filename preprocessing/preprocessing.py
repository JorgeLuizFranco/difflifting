from typing import Any

import torch
from torch_scatter import scatter
from torch_geometric.transforms import RemoveDuplicatedEdges
from torch_geometric.data import Data
from torch_geometric.utils.sparse import index2ptr
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
    sort_edge_index
)
import numpy as np

def remove_duplicate_edges(batch):
    with torch.no_grad():
        batch = batch.clone().detach()

        device = batch.x.device
        edge_slices = batch._slice_dict["edge_index"].clone().detach()
        edge_slices = edge_slices.to(device)

        edge_diff_slices = edge_slices[1:] - edge_slices[:-1]
        n_batch = len(edge_diff_slices)
        batch_e = torch.repeat_interleave(
            torch.arange(n_batch, device=device), edge_diff_slices
        )

        correct_idx = batch.edge_index[0] <= batch.edge_index[1]
        # batch_e_idx = batch_e[correct_idx]

        n_edges = scatter(correct_idx.long(), batch_e, reduce="sum")

        #           batch.edge_index = batch.edge_index[:,correct_idx]

        new_slices = torch.cumsum(
            torch.cat((torch.zeros(1, device=device, dtype=torch.long), n_edges)), 0
        )

        vertex_slice = batch._slice_dict["x"].clone()
        #           batch._slice_dict['edge_index'] = new_slices
        new_edge_index = batch.edge_index[:, correct_idx]

        return new_edge_index, vertex_slice, new_slices, batch.batch

def remove_duplicate_edges_for_nodes_dataset(data):
    with torch.no_grad():
        data = data.clone().detach()

        device = data.x.device

        RemoveDuplicatedEdges()
        edge_slices = data.edge_index
        # edge_slices = data._slice_dict["edge_index"].clone().detach()
        # edge_slices = edge_slices.to(device)

        edge_diff_slices = edge_slices[1:] - edge_slices[:-1]
        n_batch = len(edge_diff_slices)
        # batch_e = torch.repeat_interleave(
        #     torch.arange(n_batch, device=device), edge_diff_slices
        # )

        correct_idx = data.edge_index[0] <= data.edge_index[1]
        # batch_e_idx = batch_e[correct_idx]

        # n_edges = scatter(correct_idx.long(), batch_e, reduce="sum")

        #           batch.edge_index = batch.edge_index[:,correct_idx]

        # new_slices = torch.cumsum(
        #     torch.cat((torch.zeros(1, device=device, dtype=torch.long), n_edges)), 0
        # )

        vertex_slice = data.x.clone()
        #           batch._slice_dict['edge_index'] = new_slices
        new_edge_index = data.edge_index[:, correct_idx]

        return new_edge_index.to(device), vertex_slice.to(device)

class AddLaplacianEigenvectorPE:
    """
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.AddLaplacianEigenvectorPE.html
    """
    # Number of nodes from which to use sparse eigenvector computation:
    SPARSE_THRESHOLD: int = 100

    def __init__(
        self,
        k: int,
        is_undirected: bool = False,
        **kwargs: Any,
    ) -> None:
        self.k = k
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def __call__(self, data: Data):
        assert data.edge_index is not None
        num_nodes = data.num_nodes
        assert num_nodes is not None

        edge_index, edge_weight = get_laplacian(
            data.edge_index,
            data.edge_weight,
            normalization='sym',
            num_nodes=num_nodes,
        )

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        if num_nodes < self.SPARSE_THRESHOLD:
            from numpy.linalg import eig, eigh
            eig_fn = eig if not self.is_undirected else eigh

            eig_vals, eig_vecs = eig_fn(L.todense())  # type: ignore
        else:
            from scipy.sparse.linalg import eigs, eigsh
            eig_fn = eigs if not self.is_undirected else eigsh

            eig_vals, eig_vecs = eig_fn(  # type: ignore
                L,
                k=self.k + 1,
                which='SR' if not self.is_undirected else 'SA',
                return_eigenvectors=True,
                **self.kwargs,
            )

        sort_idx = eig_vals.argsort()
        eig_vecs = np.real(eig_vecs[:, sort_idx])

        data.EigVecs = torch.from_numpy(eig_vecs[:, 1:self.k + 1])
        data.EigVals = torch.from_numpy(np.real(eig_vals[sort_idx][1:self.k + 1]))

        # pad
        if data.EigVecs.shape[1] < self.k:
            data.EigVecs = torch.cat([data.EigVecs,
                                      data.EigVecs.new_zeros(num_nodes, self.k - data.EigVecs.shape[1])], dim=1)
            data.EigVals = torch.cat([data.EigVals,
                                      data.EigVals.new_zeros(self.k - data.EigVecs.shape[1])], dim=0)

        return data

