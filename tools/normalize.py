
from torch_geometric.data import Data
import torch
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, remove_self_loops


def normalize_hodge_laplacians(dataset, args):
    data_list = list(dataset[0])
    data = {}

    for value, key in zip(data_list[0], data_list[1]):
        if key.startswith(f"hodge_laplacian_"):
            if key.endswith(f"0"):
                data.update({key: normalize_matrix(value)})
            else:
                data.update({key: normalize_matrix(value, int(key[-1]))})
        else:
            data[key] = value

    return [Data(**data)]


def normalize_matrix(matrix, dim=0, apply_normalization=True):
    """
    Normalize the input matrix.
    """
    if not apply_normalization:
        return matrix

    matrix_dense = matrix.to_dense()

    n = matrix_dense.shape[0]
    identity = torch.eye(n, device=matrix_dense.device)

    if dim > 0:
        matrix_dense += 2 * identity
    else:
        matrix_dense += identity

    abs_matrix = torch.abs(matrix_dense)
    row_sum = abs_matrix.sum(dim=1)

    row_sum = torch.where(row_sum != 0, row_sum, torch.tensor(1.0, device=row_sum.device))
    inv_sqrt_row_sum = 1.0 / torch.sqrt(row_sum)

    diag_matrix = torch.diag(inv_sqrt_row_sum)

    normalized_matrix = diag_matrix @ matrix_dense @ diag_matrix

    return normalized_matrix