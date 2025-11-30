import networkx as nx
import torch
import torch.nn as nn
import torch_geometric
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import is_undirected
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data

from torch.nn.utils.rnn import pad_sequence


import torch.nn.functional as F


from layers.encoders.all_cell_features_encoders import AllCellFeatureEncoder
from model.GNN import GIN, GPS
# from layers.diff_lifting import DiffLifting
from model.TNN import TNN
from model.tnn_with_lifiting import ProjectionSum
from preprocessing.preprocessing import remove_duplicate_edges
from tools.redout import DirectReadout
from tools.redout import PropagateSignalDown
HYPERGRAPH_MODULES = ["UniGCNII", "UniGCN", "AST", "HyperGAT", "UniGIN", "UniSAGE"]

class AttentionLifting(nn.Module):
    """Lift node features to hyperedge features using attention mechanism."""

    def __init__(self, feature_dim=None, device="cpu"):
        super().__init__()
        self.W1 = torch.randn(feature_dim, feature_dim, device=device) if feature_dim else torch.randn(64, 64,
                                                                                                       device=device)
        self.W2 = torch.randn(feature_dim, feature_dim, device=device) if feature_dim else torch.randn(64, 64,
                                                                                                       device=device)
        self.W3 = torch.randn(feature_dim, feature_dim, device=device) if feature_dim else torch.randn(64, 64,
                                                                                                       device=device)
        self.k_v = torch.tensor(2.0)  # Automatically requires_grad=True
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(feature_dim if feature_dim else 64, feature_dim * 2 if feature_dim else 128),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_dim * 2 if feature_dim else 128, feature_dim if feature_dim else 64)
        ).to(device)

    def print_grad_W1(grad):
        print("Gradient for W1:", grad)

    def print_grad_W2(grad):
        print("Gradient for W2:", grad)

    def print_grad_W3(grad):
        print("Gradient for W3:", grad)

    def print_grad_kv(grad):
        print("Gradient for k_v:", grad)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def lift_features(self, data):
        """Project node features to higher-order structures using attention."""
        keys = sorted(
            [key.split("_")[1] for key in data if ("incidence" in key and "-" not in key)]
        )

        self.k_v = data["k_v"]  # Automatically requires_grad=True

        for elem in keys:
            if f"x_{elem}" not in data:
                idx_to_project = 0 if elem == "hyperedges" else int(elem) - 1
                incidence = data["incidence_" + elem]

                # Get nodes involved in each structure
                node_features = data[f"x_{idx_to_project}"]

                # For each structure (edge/hyperedge), get its incident nodes
                structures = []
                for i in range(incidence.shape[1]):
                    nodes = torch.where(incidence[:, i] != 0)[0]
                    if len(nodes) > 0:
                        structures.append((i, nodes))

                # Compute lifted features for each structure
                lifted_features = []
                for struct_idx, nodes in structures:
                    features = node_features[nodes]

                    # Compute attention scores using the gradient-preserving k_v
                    scaling_factor = torch.sqrt(self.k_v)
                    query = torch.matmul(features, self.W1.t())
                    key = torch.matmul(features, self.W2.t())
                    scores = torch.matmul(query, key.t()) / scaling_factor

                    # Apply attention
                    attention = torch.softmax(scores, dim=-1)
                    values = torch.matmul(features, self.W3.t())
                    messages = torch.matmul(attention, values)

                    # Apply order-invariant aggregation
                    structure_feature = self.phi(messages.mean(dim=0, keepdim=True))
                    lifted_features.append(structure_feature)

                # Combine all lifted features
                if lifted_features:
                    data["x_" + elem] = torch.cat(lifted_features, dim=0)
                else:
                    data["x_" + elem] = torch.zeros(
                        (incidence.shape[1], node_features.shape[1]),
                        device=node_features.device
                    )

        return data

    def forward(self, data):
        """Apply the lifting to the input data."""
        data = self.lift_features(data)
        data["x_1"]= data["incidence_1"].T @ data["x_0"]
        return data



def compute_node_cell_matrix(
    cycles: list[list[int]],
    embeddings: torch.Tensor,
    cell_mlp: torch.nn.Module,
    sharpening_factor: float = 10.0
):
    """
    cycles:    list of C cycles (each a list of node indices)
    embeddings:[N, D] node embeddings
    cell_mlp:  Module mapping [C, D] → [C, 2] logits
    returns:
      pooled:             [C, D] mean‑pooled cycle embeddings
      incidence_sampled:  sparse [N, C] with STE inclusion values
    """
    device = embeddings.device
    N, _ = embeddings.shape
    C = len(cycles)

    # 1) Pad cycles to [C, L]
    cycle_tensors = [torch.tensor(c, dtype=torch.long, device=device) for c in cycles]
    cycle_idx     = pad_sequence(cycle_tensors, batch_first=True, padding_value=-1)  # [C, L]
    L             = cycle_idx.size(1)

    # 2) Mean‑pool node embeddings per cycle → [C, D]
    clamped = cycle_idx.clamp(min=0)                          # replace -1 with 0
    gathered = embeddings[clamped]                            # [C, L, D]
    mask = (cycle_idx != -1).unsqueeze(-1).float()            # [C, L, 1]
    summed = (gathered * mask).sum(dim=1)                     # [C, D]
    counts = mask.sum(dim=1).clamp(min=1)                     # [C, 1]
    pooled = summed / counts                                  # [C, D]

    # 3) Compute 2‑class logits & sharpened probabilities
    #    just like your edge code, but on cycles
    cell_logits = cell_mlp(pooled)                            # [C, 2]
    sharp_logits = cell_logits * sharpening_factor            # [C, 2]
    cell_probs = F.softmax(sharp_logits, dim=-1)[:, 1]         # sharpened p(class=1), [C]

    # 4) Straight‑through estimator
    cell_hard = (cell_probs > 0.5).float()                    # [C], 0 or 1
    cell_ste  = cell_hard + (cell_probs - cell_probs.detach())# [C], STE

    # ⚠️ Check if no cycle was selected
    if cell_ste.sum() == 0:
        empty_indices = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_values = torch.empty((0,), device=device)
        node_cell = torch.sparse_coo_tensor(
            empty_indices,
            empty_values,
            size=(N, 0),
            device=device
        ).coalesce()
        return pooled, node_cell


    # 5) Scatter these into a sparse [N, C] incidence matrix
    #    a) flatten real (node,cycle) pairs
    valid = (cycle_idx != -1)                                 # [C, L]
    flat_nodes  = cycle_idx[valid]                            # [K]
    flat_cycles = torch.arange(C, device=device)              \
                     .unsqueeze(1).expand(-1, L)[valid]       # [K]
    flat_vals   = cell_ste.unsqueeze(1).expand(-1, L)[valid]  # [K]

    #    b) build COO sparse tensor
    indices = torch.stack([flat_nodes, flat_cycles], dim=0)   # [2, K]
    values  = flat_vals                                       # [K]
    node_cell_sampled = torch.sparse_coo_tensor(
        indices, values,
        size=(N, C),
        device=device,
        requires_grad=True
    ).coalesce()

    ### >>>> REMOVE ALL‐ZERO COLUMNS (cycles) WITHOUT BREAKING GRADIENTS <<<< ###

    # 1. Sum per cycle‐column to detect empty cycles
    col_sums = torch.sparse.sum(node_cell_sampled, dim=0).to_dense()  # [C]

    # 2. Mask of cycles to keep
    keep_mask = col_sums > 0  # [C] bool
    if keep_mask.numel() == 0 or not keep_mask.any():
        # Check if keep_mask is empty
        # Handle the case where no cycles are kept
        empty_indices = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_values = torch.empty((0,), device=device)
        node_cell_sampled = torch.sparse_coo_tensor(
            empty_indices,
            empty_values,
            size=(N, 0),
            device=device
        ).coalesce()
        return pooled, node_cell_sampled
    # 3. Duplicate mask to match the flattened indices length
    #    Here each index in `indices` refers directly to a cycle,
    #    so we just index by the second row:
    mask_cycles = keep_mask[indices[1]]  # [K]

    # 4. Filter indices & values
    filtered_indices = indices[:, mask_cycles]
    filtered_values  = values[mask_cycles]

    kept_cycles = torch.nonzero(keep_mask, as_tuple=False).view(-1)
    # now kept_cycles is always 1‑D, even if it has 0 or 1 element

    if kept_cycles.numel() == 0:
        # nothing kept → return empty [N,0]
        empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_val = torch.empty((0,), device=device)
        node_cell_empty = torch.sparse_coo_tensor(
            empty_idx, empty_val,
            size=(N, 0),
            device=device,
            requires_grad=True
        ).coalesce()
        return pooled, node_cell_empty

    # remap columns 0..C‑1 → 0..C'‑1
    new_ids = torch.arange(kept_cycles.size(0), device=device)
    old2new = torch.full((C,), -1, dtype=torch.long, device=device)
    old2new[kept_cycles] = new_ids

    # filter indices & values
    mask_k = keep_mask[indices[1]]          # [K]
    filt_idx = indices[:, mask_k]
    filt_idx[1] = old2new[filt_idx[1]]
    filt_val = values[mask_k]

    node_cell_pruned = torch.sparse_coo_tensor(
        filt_idx, filt_val,
        size=(N, kept_cycles.size(0)),
        device=device,
        requires_grad=True
    ).coalesce()

    return pooled, node_cell_pruned

class TNN_KNN_MLP_G(nn.Module):

    def __init__(self, in_channels, args, hidden_dim, num_classes, k=2, diff_lifting=False, global_pool="sum",
             device="cpu", tnn_type="SCN2", num_layers_tnn=4, num_layers_gnn=3, embedding_dim=64, k_max=10, deterministic=False):
        super().__init__()
        self.k = k
        self.k_min = 2
        self.k_max = k_max

        self.triangle_count = 0  # Add this to track triangles
        self.diff_lifting = diff_lifting
        self.tnn_type = tnn_type
        self.num_classes = num_classes
        self.dataset = args.dataset
        self.feature_encoder = AllCellFeatureEncoder(in_channels=[in_channels,in_channels,in_channels], out_channels=hidden_dim,
                                                  proj_dropout=0.5)
        if args.dataset == "ogbg-molhiv":

            self.atom_encoder = AtomEncoder(emb_dim=in_channels)

        if diff_lifting:

            if args.gnn == "GIN":
                self.gnn = GIN(in_channels, embedding_dim, embedding_dim, num_layers_gnn).to(device)
            elif args.gnn == "GPS":
                if args.dataset == "ZINC":
                    self.gnn = GPS(in_channels, embedding_dim, args.positional_walking_len, num_layers_gnn, is_zinc=True).to(device)
                else:
                    self.gnn = GPS(in_channels, embedding_dim, args.positional_walking_len, num_layers_gnn).to(device)

            self.pool = global_mean_pool
            self.k = k

            if tnn_type in ["UniGCNII", "UniGCN","AST", "HyperGAT", "UniGIN", "UniSAGE"]:
                self.mlp = nn.Sequential(
                    nn.Linear(embedding_dim, 2 * hidden_dim),
                    nn.ReLU(),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(hidden_dim, 1),
                )
            if tnn_type not in ["UniGCNII", "UniGCN","AST", "HyperGAT", "UniGIN", "UniSAGE"]:
                # self.mlp_cell = nn.Sequential(
                #     nn.Linear(k, 2 * hidden_dim),  # Use k as input dimension
                #     nn.ReLU(),
                #     nn.Linear(2 * hidden_dim, hidden_dim),
                #     nn.ReLU(),
                #     nn.Dropout(0.5),
                #     nn.Linear(hidden_dim, k),  # Output k features
                # )
                self.edge_mlp = nn.Sequential(
                    nn.Linear(embedding_dim, 64),  # Input: concatenated edge embeddings
                    nn.ReLU(),
                    nn.Linear(64, 2)  # Output logits for two classes (0 and 1)
                )
                # self.mlp_cell2 = nn.Sequential(
                #     nn.Linear(128, 2 * hidden_dim),  # Change input dimension to 128
                #     nn.ReLU(),
                #     nn.Linear(2 * hidden_dim, hidden_dim),
                #     nn.ReLU(),
                #     nn.Dropout(0.5),
                #     nn.Linear(hidden_dim, 1),  # Output: probability per cycle
                # )
            self.k_mlp = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, self.k_max - self.k_min + 1)  # Output size covers all possible k values
            )

            self.triangle_count = 0
            self.projection_sum = ProjectionSum()

            #self.attention_lift = AttentionLifting(feature_dim=in_channels, device=device)
        if tnn_type in HYPERGRAPH_MODULES:
             hidden_dim = hidden_dim
        else:
            if diff_lifting:
                hidden_dim = embedding_dim
        self.tnn = TNN(
            model_type=tnn_type,  # choose TNN model
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            in_channels_1=hidden_dim,
            in_channels_2=hidden_dim,
            n_layers=num_layers_tnn,
            device=device,
            sub_gccn=args.sub_gccn_model
        )


        if args.no_readout:
            self.readout = DirectReadout(**{
                "readout_name": "DirectReadout",
                "task_level": "graph",
                "hidden_dim": hidden_dim,
                "out_channels": num_classes,
            })
        else:
            self.readout = PropagateSignalDown(**{
                "readout_name": "PropagateSignalDownLinear",
                "num_cell_dimensions": 3,
                "hidden_dim": hidden_dim,
                "out_channels": num_classes,
                "task_level": "graph",
                "pooling_type": global_pool,
            })

    def __create_laplacians(self, data, incidence_matrix_1, lifted_data, data_for_lifting):
        new_edge_index, new_edge_attr = torch_geometric.utils.get_laplacian(data.edge_index)
        data.x_1 = lifted_data["x_1"]
        data.x_2 = lifted_data["x_2"]
        laplacian_0 = torch.sparse_coo_tensor(
            indices=new_edge_index,
            values=new_edge_attr,
            size=(data.x.shape[0], data.x.shape[0])
        )

        data.laplacian_up_0 = laplacian_0
        data.laplacian_up_1 = torch.spmm(data_for_lifting["incidence_2"],
                                         data_for_lifting["incidence_2"].T).to_sparse_coo()
        data.laplacian_down_1 = torch.spmm(data_for_lifting["incidence_1"].T,
                                           data_for_lifting["incidence_1"]).to_sparse_coo()

        data.laplacian_down_2 = torch.spmm(data_for_lifting["incidence_2"].T,
                                           data_for_lifting["incidence_2"]).to_sparse_coo()
        data.node_edge_matrix = incidence_matrix_1

        data.incidence_1 = data_for_lifting.get("incidence_1")
        data.incidence_2 = data_for_lifting.get("incidence_2")

        data.hodge_laplacian_0 = data.laplacian_up_0  # + data.laplacian_down_0
        data.hodge_laplacian_1 = data.laplacian_up_1 + data.laplacian_down_1
        data.hodge_laplacian_2 = data.laplacian_down_2
        return data

    def forward(self, batch):
        data = batch
        if self.diff_lifting:
            x, edge_index = data.x, data.edge_index
            if self.dataset=="ogbg-molhiv":
                x = self.atom_encoder(x).float()  # x is input atom feature

            #print("Initial data.x shape:", data.x.shape)  # Initial shape
            edge_index_undirected, vertex_slice, new_slices, data.batch = remove_duplicate_edges(data)

            #print("number of edges: ", edge_index_undirected.size(1))
            embeddings = self.gnn(data)

            # Keep gradient through mean operation
            embedding_mean = embeddings

            #print("embeddings requires_grad:", embeddings.requires_grad)

            # --- k_v sampling ---
            if getattr(self, "deterministic", False):
                k_logits = self.k_mlp(embedding_mean)
                p = F.softmax(k_logits, dim=-1)
                k_range = torch.arange(self.k_min, self.k_max + 1, device=k_logits.device, dtype=k_logits.dtype)
                k_v = (p * k_range.unsqueeze(0)).sum(dim=-1)
                k_v = k_v.round()
                self.k_v = k_v
            else:
                k_logits = self.k_mlp(embedding_mean)
                k_sample = F.gumbel_softmax(k_logits, tau=1.0, hard=True)
                k_range = torch.arange(self.k_min, self.k_max + 1, device=k_logits.device, dtype=k_logits.dtype)
                k_v = torch.sum(k_sample * k_range, dim=-1)
                self.k_v = k_v

            mask_knn = torch.nn.functional.one_hot(data.batch_0, num_classes=vertex_slice.shape[0] - 1)

            mask_knn = mask_knn.float() @ mask_knn.T.float()

            distances = torch.cdist(embeddings, embeddings)

            distances = mask_knn * distances + (1 - mask_knn) * 1e7

            if (self.tnn_type == "UniGCNII" or self.tnn_type == "UniGCN" or
                self.tnn_type == "HyperGAT" or self.tnn_type == "UniGIN" or self.tnn_type == "UniSAGE"):

                knn_indices = torch.topk(-distances, torch.max(self.k_v).long().item(), dim=-1)[1]
                aranged_indices = torch.arange(torch.max(self.k_v).long().item(), device=x.device).expand(self.k_v.shape[0], -1)
                kv_mask = aranged_indices < k_v.unsqueeze(1)
                first_neighbor = knn_indices[:, 0].unsqueeze(1)
                knn_selected = torch.where(kv_mask, knn_indices, first_neighbor)
                knn_indices = knn_selected
                pooled_embeddings = embeddings[knn_indices.long()].mean(axis=1, keepdim=True).squeeze()

                # --- inclusion sampling ---
                include_probs = torch.sigmoid(self.mlp(pooled_embeddings))  # Shape: [num_nodes, 1]
                if getattr(self, "deterministic", False):
                    straight_through_samples = (include_probs > 0.5).float()
                else:
                    inclusion_samples = (torch.rand_like(include_probs) < include_probs).float()
                    straight_through_samples = inclusion_samples + (include_probs - include_probs.detach())

                num_edges = edge_index_undirected.size(1)

                incidence_matrix_1 = torch.zeros((data.x.size(0), num_edges), device=data.x.device)

                for idx, edge in enumerate(edge_index_undirected.T):
                    incidence_matrix_1[edge[0], idx] = 1
                    incidence_matrix_1[edge[1], idx] = 1

        
                num_nodes = data.x.size(0)

                mask = torch.zeros((num_nodes, num_nodes), device=data.x.device)

                node_triangle_matrix = mask.scatter_(1, knn_indices, straight_through_samples.repeat(1, torch.max(self.k_v).long().item()))

                # This is not used, but we keep it for future use
                incidence_matrix_2 = incidence_matrix_1.T @ node_triangle_matrix
                incidence_matrix_2 = torch.div(incidence_matrix_2, 2, rounding_mode='trunc')
                # ---- # ----- # ----- #

                #print("incidence matrix before concatenation:", incidence_matrix_1.shape)
                incidence_matrix_1 = torch.cat((incidence_matrix_1, node_triangle_matrix), dim=1)

                #print("incidence matrix after concatenation:", incidence_matrix_1.shape)
                #print(incidence_matrix_1.grad_fn)
                
                data.x_0 = x.float()

                data.x_0 = torch.div(data.x_0, torch.max(self.k_v))

                #print("data.x_0 after division:", data.x_0.shape)

                col_sums = incidence_matrix_1.sum(dim=0)              # [total_cols]

                # 2) Build keep‐mask
                keep = col_sums > 0                                   # [total_cols], bool

                # # Print the number of columns before and after pruning
                # print(f"Number of columns before pruning: {incidence_matrix_1.size(1)}")
                # print(f"Number of columns after pruning: {keep.sum().item()}")

                # 3) Index out zero columns (gather on dim=1 preserves grads)
                incidence_pruned = incidence_matrix_1[:, keep]        # [num_nodes, num_kept]

                # 4) Convert to sparse‐COO if that’s what your pipeline expects
                data.incidence_1 = incidence_pruned.to_sparse_coo()
            

                

                #print(data)
                data = self.feature_encoder(data)
                #print("data after feature encoder", data)
                #print("shapes before tnn: ", data["x_0"].shape)
                tnn_output = self.tnn(data)
                #print("shapes tnn: ", tnn_output["x_0"].shape, tnn_output["x_1"].shape)
                out = self.readout(tnn_output, batch)
                #print(out)
                return out["logits"]    

            ## Now the Cellular Complex Diff Lifiting
            else:

                knn_indices = torch.topk(-distances, torch.max(self.k_v).long().item(), dim=-1)[1]
                aranged_indices = torch.arange(torch.max(self.k_v).long().item(), device=x.device).expand(self.k_v.shape[0], -1)
                kv_mask = aranged_indices < k_v.unsqueeze(1)
                first_neighbor = knn_indices[:, 0].unsqueeze(1)
                knn_selected = torch.where(kv_mask, knn_indices, first_neighbor)
                knn_indices = knn_selected


                num_nodes, k_edges = knn_indices.shape

                node_indices = torch.arange(num_nodes, device=knn_indices.device)

                # Unsqueeze to make it a column vector of shape [num_nodes, 1]
                node_indices = node_indices.unsqueeze(1)

                # Expand node_indices to repeat each node index for each candidate neighbor.
                # After expansion, node_indices has shape [num_nodes, k_edges]
                node_indices = node_indices.expand(num_nodes, k_edges)

                # Reshape both node_indices and knn_indices to a flat vector so that each element corresponds to an edge.
                node_indices_flat = node_indices.reshape(-1)
                knn_indices_flat = knn_indices.reshape(-1)

                # Stack the flattened node_indices and knn_indices to form an edge_index tensor.
                # The resulting edge_index tensor will have shape [2, num_nodes * k_edges],
                # where the first row is the source node and the second row is the target node.
                edge_indices_knn = torch.stack([node_indices_flat, knn_indices_flat], dim=0)

                # print(edge_indices_knn)

                source_embeddings = embeddings[edge_indices_knn[0]]  # Shape: [num_selected_edges, embedding_dim]
                target_embeddings = embeddings[edge_indices_knn[1]]  # Shape: [num_selected_edges, embedding_dim]
                edge_embeddings = (source_embeddings + target_embeddings) /2
                # Step 2: Compute logits and sharpen probabilities
                edge_logits = self.edge_mlp(edge_embeddings)  # Shape: [num_edges, 2]

                # Define the sharpening factor directly in the code
                sharpening_factor = 10.0  # Example value for sharpening

                # Apply sharpening to logits and compute probabilities
                edge_probs = torch.softmax(edge_logits * sharpening_factor, dim=-1)[:, 1]  # Sharpened probability of class 1

                # Step 3: Apply straight-through estimator
                edge_classes = (edge_probs > 0.5).float()  # Binary values (0 or 1) during forward pass
                edge_classes = edge_classes + (edge_probs - edge_probs.detach())  # Preserve gradients


                                # Step 4: Construct incidence matrix as a sparse tensor with gradients
                                # Step 4: Construct incidence matrix as a sparse tensor with gradients
                num_nodes           = embeddings.size(0)
                num_edges_sampled   = edge_indices_knn.size(1)
                num_edges           = edge_index_undirected.size(1)

                # Prepare indices for sparse incidence matrix
                rows_u = edge_indices_knn[0]                         # [num_edges_sampled]
                rows_v = edge_indices_knn[1]                         # [num_edges_sampled]
                cols   = torch.arange(num_edges_sampled, device=edge_classes.device)

                # Stack (node, edge) pairs twice: source and target
                indices_u   = torch.stack([rows_u, cols], dim=0)     # [2, num_edges_sampled]
                indices_v   = torch.stack([rows_v, cols], dim=0)
                all_indices = torch.cat([indices_u, indices_v], dim=1)  # [2, 2*num_edges_sampled]

                # Duplicate values for both ends of each edge
                all_values = torch.cat([edge_classes, edge_classes], dim=0)  # [2*num_edges_sampled]

                # Build the full sampled incidence matrix (sparse COO)
                incidence_matrix_sampled = torch.sparse_coo_tensor(
                    indices=all_indices,
                    values=all_values,
                    size=(num_nodes, num_edges_sampled),
                    device=edge_classes.device
                ).coalesce()

                ### >>>> REMOVE ZERO‐ONLY COLUMNS (keep gradient) <<<< ###

                # 1. Sum per column to detect non‑zero columns
                col_sums   = torch.sparse.sum(incidence_matrix_sampled, dim=0).to_dense()  # [num_edges_sampled]

                # 2. Boolean mask of columns to keep
                keep_mask  = col_sums > 0                                                  # [num_edges_sampled]

                # 3. Duplicate mask so it matches the doubled‑up indices
                mask_pairs = torch.cat([keep_mask, keep_mask], dim=0)                      # [2 * num_edges_sampled]

                # 4. Filter indices & values
                filtered_indices = all_indices[:, mask_pairs]
                filtered_values  = all_values[mask_pairs]

                edge_sampling = True
                # 5. Remap old edge‑column IDs → new compact range [0..num_kept-1]
                kept_cols = torch.nonzero(keep_mask, as_tuple=False).view(-1)               # always 1‑D
                if kept_cols.numel() == 0:
                    # nothing kept → empty [num_nodes, 0]
                    empty_idx = torch.empty((2, 0), dtype=torch.long, device=edge_classes.device)
                    empty_val = torch.empty((0,),     device=edge_classes.device)
                    incidence_matrix_sampled = torch.sparse_coo_tensor(
                        empty_idx, empty_val,
                        size=(num_nodes, 0),
                        device=edge_classes.device,
                        requires_grad=True
                    ).coalesce()

                    edge_sampling= False
                else:
                    new_col_range = torch.arange(kept_cols.size(0), device=edge_classes.device)
                    old2new       = torch.full((num_edges_sampled,), -1, dtype=torch.long,
                                            device=edge_classes.device)
                    old2new[kept_cols] = new_col_range

                    # apply the remapping
                    filtered_indices[1] = old2new[filtered_indices[1]]

                    # 6. Rebuild the filtered sparse incidence matrix
                    incidence_matrix_sampled = torch.sparse_coo_tensor(
                        indices=filtered_indices,
                        values=filtered_values,
                        size=(num_nodes, kept_cols.size(0)),
                        device=edge_classes.device,
                        requires_grad=True
                    ).coalesce()

                # Now combine with original edges
                original_incidence = torch.zeros((num_nodes, num_edges),
                                                device=embeddings.device)
                for idx, edge in enumerate(edge_index_undirected.T):
                    original_incidence[edge[0], idx] = 1
                    original_incidence[edge[1], idx] = 1


                if self.tnn_type == "CXN":
                    original_incidence_sparse = original_incidence
                else:
                    original_incidence_sparse = original_incidence.to_sparse_coo()
                if edge_sampling:
                    if self.tnn_type == "CXN":
                        incidence_matrix_sampled = incidence_matrix_sampled.to_dense()
                        incidence_matrix_1 = torch.cat(
                            [original_incidence_sparse, incidence_matrix_sampled],
                            dim=1
                        )
                    else:
                        incidence_matrix_1 = torch.cat(
                            [original_incidence_sparse, incidence_matrix_sampled],
                            dim=1
                        ).to_sparse_coo()
                else:
                    incidence_matrix_1 = original_incidence_sparse
                incidence_matrix_1.requires_grad_(True)

                # # Step 1: compute edge-edge adjacency via shared face
                # Step 1: Build adjacency matrix
                A = incidence_matrix_1.T @ incidence_matrix_1  # [num_edges, num_edges]
                A = A.to_sparse() if self.tnn_type == "CXN" else A

                # Step 2: Remove diagonal (self-loops) using element-wise multiplication
                num_tot_edges = A.size(0)
                identity_indices = torch.arange(num_tot_edges, device=A.device).repeat(2, 1)
                identity_values = torch.ones(num_tot_edges, device=A.device)
                identity_mask = torch.sparse_coo_tensor(identity_indices, identity_values, size=A.size()).coalesce()

                # Perform element-wise multiplication to remove diagonal entries
                A = A * (1 - identity_mask.to_dense())

                # Step 3: Replace all 2s with 1, keeping 0s and 1s untouched (differentiably!)
                A = torch.sparse_coo_tensor(
                    A.indices(),
                    torch.clamp(A.values(), max=1),
                    A.size(),
                    device=A.device
                ).coalesce()
                # Step 4: Store
                data.adjacency_1 = A

                A_0=  incidence_matrix_1 @ incidence_matrix_1.T
                A_0 = A_0.to_sparse() if self.tnn_type == "CXN" else A_0

                num_tot_edges = A_0.size(0)
                identity_indices = torch.arange(num_nodes, device=A_0.device).repeat(2, 1)
                identity_values = torch.ones(num_nodes, device=A_0.device)
                identity_mask = torch.sparse_coo_tensor(identity_indices, identity_values, size=A_0.size()).coalesce()

                A_0 = A_0 * (1 - identity_mask.to_dense())

                # Step 3: Replace all 2s with 1, keeping 0s and 1s untouched (differentiably!)
                A_0 = torch.sparse_coo_tensor(
                    A_0.indices(),
                    torch.clamp(A_0.values(), max=1),
                    A_0.size(),
                    device=A.device
                ).coalesce()
                # Step 4: Store
                data.adjacency_0 = A_0

                A_0_dense = A_0.to_dense().detach().cpu().numpy()
                G = nx.from_numpy_array(A_0_dense)

                # Save the graph figure
                # import matplotlib.pyplot as plt
                # plt.figure(figsize=(10, 10))
                # nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
                # plt.savefig("graph_visualization.png")
                # plt.close()

                # Find cycles using networkx (non-differentiable step)
                cycles = nx.cycle_basis(G)
                if self.dataset != "ogbg-molhiv":
                    cycles = [cycle for cycle in cycles if len(cycle) >= 3]  # Remove small cycles
                else:
                    cycles = [
                        cycle for cycle in cycles if len(cycle) <= 6
                    ]
                #print(cycles)
                if len(cycles)> 0:
                    polled_cycles, node_cell_matrix = compute_node_cell_matrix(cycles, embeddings, self.edge_mlp)
                    if node_cell_matrix._nnz() > 0:
                        incidence_matrix_2= incidence_matrix_1.T @ node_cell_matrix
                    else:
                        num_edges_fake = incidence_matrix_1.size(1)  # Number of edges
                        dummy_indices = torch.empty((2, 0), dtype=torch.long, device=incidence_matrix_1.device)  # No non-zero entries
                        dummy_values = torch.empty((0,), device=incidence_matrix_1.device)  # No values
                        incidence_matrix_2 = torch.sparse_coo_tensor(
                            dummy_indices,
                            dummy_values,
                            size=(num_edges_fake, 2),  # Two columns for two cells
                            device=incidence_matrix_1.device
                        ).coalesce()
                else:
                    num_edges_fake = incidence_matrix_1.size(1)  # Number of edges
                    dummy_indices = torch.empty((2, 0), dtype=torch.long, device=incidence_matrix_1.device)  # No non-zero entries
                    dummy_values = torch.empty((0,), device=incidence_matrix_1.device)  # No values
                    incidence_matrix_2 = torch.sparse_coo_tensor(
                        dummy_indices,
                        dummy_values,
                        size=(num_edges_fake, 2),  # Two columns for two cells
                        device=incidence_matrix_1.device
                    ).coalesce()



                # incidence_matrix_2= torch.div(incidence_matrix_2,2,rounding_mode='trunc')



                data_for_lifting = {}

                data_for_lifting = {
                    "x_0": embeddings,  # Node features
                    "incidence_1": incidence_matrix_1,  # Node-to-edge incidence matrix
                    "incidence_2": incidence_matrix_2,  # edge_to_triangle
                    "adjacency_1": A,
                    "adjacency_0": A_0
                }

                lifted_data = self.projection_sum(data_for_lifting)

                lifted_data["adjacency_1"] = A
                #print(lifted_data)
                lifted_data["x_0"] = torch.div(lifted_data["x_0"], torch.max(self.k_v))
                lifted_data["cell_statistics"] = cycles

                #print(lifted_data)
                lifted_data_obj = Data(**lifted_data)
                tnn_output = self.tnn(lifted_data_obj)
                # print("shapes tnn: ", tnn_output["x_0"].shape, tnn_output["x_1"].shape)
                batch["incidence_1"]= incidence_matrix_1

                batch["incidence_2"]= incidence_matrix_2

                out = self.readout(tnn_output, batch)
                # print(out)
                return out["logits"]


        #print("data ": data)
        data = self.feature_encoder(data)
        # print("data after feature encoder", data)
        # print("shapes before tnn: ", data["x_0"].shape)
        tnn_output = self.tnn(data)
        # print("shapes tnn: ", tnn_output["x_0"].shape, tnn_output["x_1"].shape)
        out = self.readout(tnn_output, batch)
        # print(out)
        return out["logits"]


def generate_graph_from_data(
        data: torch_geometric.data.Data
) -> nx.Graph:
    r"""Generate a NetworkX graph from the input data object.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The input data.

    Returns
    -------
    nx.Graph
        The generated NetworkX graph.
    """
    # Check if data object have edge_attr, return list of tuples as [(node_id, {'features':data}, 'dim':1)] or ??
    nodes = [
        (n, dict(features=data.x[n], dim=0))
        for n in range(data.x.shape[0])
    ]

    if hasattr(data, "edge_attr"):
        # In case edge features are given, assign features to every edge
        edge_index, edge_attr = (
            data.edge_index,
            (
                data.edge_attr
                if is_undirected(data.edge_index, data.edge_attr)
                else to_undirected(data.edge_index, data.edge_attr)
            ),
        )
        edges = [
            (i.item(), j.item(), dict(features=edge_attr[edge_idx], dim=1))
            for edge_idx, (i, j) in enumerate(
                zip(edge_index[0], edge_index[1], strict=False)
            )
        ]
    else:
        # If edge_attr is not present, return list list of edges
        edges = [
            (i.item(), j.item(), {})
            for i, j in zip(
                data.edge_index[0], data.edge_index[1], strict=False
            )
        ]
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph



