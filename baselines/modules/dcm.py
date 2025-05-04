import torch
from baselines.modules.cell_utils import compute_boundary, compute_Lup, compute_Lup_entmax, line_graph
from baselines.modules.dgm import DGM, LayerNorm
from baselines.modules.layers import GNN, MLP
from torch_geometric.utils import scatter

class DCM(torch.nn.Module):
    def __init__(self, use_gcn=True, dgm_layers=2, dropout=0.5, gamma=50, std=0, k=4):
        super(DCM, self).__init__()
        dgm_layers = [64 for _ in range(2 + 1)]
        if use_gcn:
            self.graph_f = DGM(
                GNN(dgm_layers, dropout=dropout),
                gamma=gamma,
                std=std,
            )
        else:
            self.graph_f = DGM(
                MLP(dgm_layers, dropout=dropout),
                gamma=gamma,
                std=std,
            )
        self.poly_ln = LayerNorm(1)
        self.k = k
        self.std = std
    def forward(self, data):#x, edge_index=None, batch=None, edge_weight=None):
        x_aux = data["x_0"].detach()
        x_aux, edges, ne_probs = self.graph_f(x_aux, data["edge_index"], None)
        boundaries, row, col, xe, xe_aux, i, id_maps = compute_boundary(
            data["x_0"].detach(), x_aux.detach(), edges, max_k=self.k
        )
        Ldo, xe, xe_aux = line_graph(data["x_0"], row, col, i, xe, xe_aux)
        

        
        np_probs = None

        Lup, poly_probs = compute_Lup_entmax(
            xe_aux, boundaries, id_maps, self.poly_ln, self.std
        )
        np_probs = scatter(
            poly_probs, row, dim=0, dim_size=data["x_0"].shape[0], reduce="sum"
        ) + scatter(poly_probs, col, dim=0, dim_size=data["x_0"].shape[0], reduce="sum")
      
        return {"x": data["x_0"], 
                "xe": xe, 
                "edges": edges, 
                "row": row, 
                "col": col, 
                "ne_probs": ne_probs, 
                "Ldo": Ldo, 
                "Lup": Lup, 
                "np_probs": np_probs}