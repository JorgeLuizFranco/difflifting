import torch
from torch_scatter import scatter
from torch_geometric.transforms import RemoveDuplicatedEdges

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