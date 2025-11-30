from sympy import false
from topomodelx.nn.cell.ccxn import CCXN
from topomodelx.nn.hypergraph.allset_transformer import AllSetTransformer
from topomodelx.nn.hypergraph.unisage import UniSAGE
from topomodelx.nn.hypergraph.unigin import UniGIN
from topomodelx.nn.hypergraph.unigcn import UniGCN

from layers.hypergnns.hypergat import HyperGAT
from topomodelx.nn.simplicial.scn2 import SCN2
from torch import nn
from torch_geometric.nn import global_mean_pool, GAT, GCNConv

from model.models.topotune import TopoTune
from tools.normalize import normalize_matrix
import torch

from model.GNN import GIN


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.conv1 = GCNConv(in_channels, hidden_channels,
                             normalize=False)
        self.conv2 = GCNConv(hidden_channels, out_channels,
                             normalize=False)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class TNN(nn.Module):
    def __init__(self, model_type, in_channels, hidden_channels, in_channels_1=7, in_channels_2=7, normalize_laplacians=True,n_layers=4,device="cpu",sub_gccn="GAT", **kwargs):
        super().__init__()
        if model_type == "CWN":
            self.base_model = CWN(in_channels, in_channels_1, in_channels_2, hidden_channels, n_layers=n_layers, **kwargs).to(device)
        elif model_type == "SCN2":
            self.base_model = SCN2(in_channels, in_channels, in_channels, n_layers=n_layers, **kwargs).to(device)
        elif model_type == "CXN":
            self.base_model =  CCXN(in_channels, in_channels, in_channels, n_layers=n_layers).to(device)
        elif model_type == "UniGCNII":
            self.base_model =  UniGCNII(in_channels, in_channels).to(device)
        elif model_type == "UniSAGE":
            self.base_model =  UniSAGE(in_channels, in_channels).to(device)

        elif model_type == "UniGCN":
            self.base_model = UniGCN(in_channels, in_channels).to(device)
        elif model_type == "HyperGAT":
            self.base_model = HyperGAT(in_channels, in_channels, n_layers=n_layers).to(device)
        elif model_type == "UniGIN":
            self.base_model = UniGIN(in_channels, in_channels, n_layers=n_layers).to(device)
        elif model_type == "AST":
            self.base_model =  AllSetTransformer(in_channels, in_channels,  n_layers=n_layers, n_heads=4).to(device)
        elif model_type == "TOPOTUNE":

            neighborhoods = ["adjacency_0", "incidence_0","adjacency_1", "incidence_1"]
            dim_hidden = hidden_channels
            if sub_gccn == "GAT":
                sub_gccn_model = GAT(in_channels=in_channels, hidden_channels=dim_hidden, num_layers=1,
                                 out_channels=dim_hidden,
                             heads=2, v2=False)
            elif sub_gccn == "GIN":
                sub_gccn_model = GIN(in_channels, dim_hidden, dim_hidden, 2).to(device)
            else:
                sub_gccn_model = GCN(in_channels=in_channels, hidden_channels=dim_hidden, out_channels=dim_hidden,)
            backbone_config = {
                "GNN": sub_gccn_model,
                "neighborhoods": neighborhoods,
                "layers": 2,
                "use_edge_attr": False,
                "activation": "relu",
                "gnn_type": sub_gccn,
            }
            self.base_model = TopoTune(**backbone_config).to(device)
        self.incidence_models = ["UniGCN", "HyperGAT", "UniGIN", "UniSAGE"]
        self.pooling_fun = global_mean_pool
        self.normalize_laplacians = normalize_laplacians
        self.model_type = model_type

    def forward(self, data):
        model_out = {}
        x=[]
        if self.model_type == "SCN2":
            print("SCN2")
            x = self.base_model(data.x_0, data.x_1, data.x_2,
                            normalize_matrix(data.hodge_laplacian_0, 0),
                            normalize_matrix(data.hodge_laplacian_1, 1),
                            normalize_matrix(data.hodge_laplacian_2, 2))
        elif self.model_type == "CWN":
            x = self.base_model(data.x_0, data.x_1, data.x_2,
                            data.adjacency_1,
                            data.incidence_2,
                            data.incidence_1.T)
        elif self.model_type == "CXN":
            x = self.base_model(data.x_0, data.x_1,
                            data.adjacency_0,
                            data.incidence_2.T)

        elif self.model_type == "AST":
            x = self.base_model(data.x_0, data.incidence_1)

            model_out["x_0"] = x[0]
            model_out["x_1"] = x[1]

            return model_out
        elif self.model_type in self.incidence_models:

            x = self.base_model(data.x_0, data.incidence_1)

            model_out["x_0"] = x[0]
            model_out["x_1"] = x[1]

            return model_out

        elif self.model_type == "UniGCNII":

            
            x = self.base_model(data.x_0, data.incidence_1)
            
            model_out["x_0"] = x[0]
            model_out["x_1"] = x[1]

            return model_out
        elif self.model_type == "TOPOTUNE":
            x = self.base_model(data)

        model_out["x_0"] = x[0]
        model_out["x_1"] = x[1]
        model_out["x_2"] = x[2]
        return model_out
    



import torch
import math


class CWN(torch.nn.Module):
    """Implementation of a specific version of CW network [1]_.

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on nodes (0-cells).
    in_channels_1 : int
        Dimension of input features on edges (1-cells).
    in_channels_2 : int
        Dimension of input features on faces (2-cells).
    hid_channels : int
        Dimension of hidden features.
    n_layers : int
        Number of CWN layers.
    **kwargs : optional
        Additional arguments CWNLayer.

    References
    ----------
    .. [1] Bodnar, et al.
        Weisfeiler and Lehman go cellular: CW networks.
        NeurIPS 2021.
        https://arxiv.org/abs/2106.12575
    """

    def __init__(
        self,
        in_channels_0,
        in_channels_1,
        in_channels_2,
        hid_channels,
        n_layers,
        **kwargs,
    ):
        super().__init__()
        self.proj_0 = torch.nn.Linear(in_channels_0, hid_channels)
        self.proj_1 = torch.nn.Linear(in_channels_1, hid_channels)
        self.proj_2 = torch.nn.Linear(in_channels_2, hid_channels)

        self.layers = torch.nn.ModuleList(
            CWNLayer(
                in_channels_0=hid_channels,
                in_channels_1=hid_channels,
                in_channels_2=hid_channels,
                out_channels=hid_channels,
                **kwargs,
            )
            for _ in range(n_layers)
        )

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        adjacency_0,
        incidence_2,
        incidence_1_t,
    ):
        """Forward computation through projection, convolutions, linear layers and average pooling.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, in_channels_0)
            Input features on the nodes (0-cells).
        x_1 : torch.Tensor, shape = (n_edges, in_channels_1)
            Input features on the edges (1-cells).
        x_2 : torch.Tensor, shape = (n_faces, in_channels_2)
            Input features on the faces (2-cells).
        adjacency_0 : torch.Tensor, shape = (n_edges, n_edges)
            Upper-adjacency matrix of rank 1.
        incidence_2 : torch.Tensor, shape = (n_edges, n_faces)
            Boundary matrix of rank 2.
        incidence_1_t : torch.Tensor, shape = (n_edges, n_nodes)
            Coboundary matrix of rank 1.

        Returns
        -------
        x_0 : torch.Tensor, shape = (n_nodes, in_channels_0)
            Final hidden states of the nodes (0-cells).
        x_1 : torch.Tensor, shape = (n_edges, in_channels_1)
            Final hidden states the edges (1-cells).
        x_2 : torch.Tensor, shape = (n_edges, in_channels_2)
            Final hidden states of the faces (2-cells).
        """

        x_0 = F.elu(self.proj_0(x_0))
        x_1 = F.elu(self.proj_1(x_1))
        x_2 = F.elu(self.proj_2(x_2))

        for layer in self.layers:
            x_1 = layer(
                x_0,
                x_1,
                x_2,
                adjacency_0,
                incidence_2,
                incidence_1_t,
            )

        return x_0, x_1, x_2


import torch.nn.functional as F



class CWNLayer(nn.Module):
    r"""Layer of a CW Network (CWN).

    Implementation of the CWN layer proposed in [1]_.

    This module is composed of the following layers:
    1. A convolutional layer that sends messages from r-cells to r-cells.
    2. A convolutional layer that sends messages from (r-1)-cells to r-cells.
    3. A layer that creates representations in r-cells based on the received messages.
    4. A layer that updates representations in r-cells.

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on (r-1)-cells (nodes in case r = 1).

    in_channels_1 : int
        Dimension of input features on r-cells (edges in case r = 1).

    in_channels_2 : int
        Dimension of input features on (r+1)-cells (faces in case r = 1).

    out_channels : int
        Dimension of output features on r-cells.

    conv_1_to_1 : torch.nn.Module, optional
        A module that convolves the representations of upper-adjacent neighbors of r-cells
        and their corresponding co-boundary (r+1) cells.

        If None is passed, a default implementation of this module is used
        (check the docstring of _CWNDefaultFirstConv for more detail).

    conv_0_to_1 : torch.nn.Module, optional
        A module that convolves the representations of (r-1)-cells on the boundary of r-cells.

        If None is passed, a default implementation of this module is used
        (check the docstring of _CWNDefaultSecondConv for more detail).

    aggregate_fn : torch.nn.Module, optional
        A module that aggregates the representations of r-cells obtained by convolutional layers.

        If None is passed, a default implementation of this module is used
        (check the docstring of _CWNDefaultAggregate for more detail).

    update_fn : torch.nn.Module, optional
        A module that updates the aggregated representations of r-cells.

        If None is passed, a default implementation of this module is used
        (check the docstring of _CWNDefaultUpdate for more detail).

    **kwargs : optional
        Additional arguments for the modules of the CWN layer.

    References
    ----------
    .. [1] Bodnar, et al.
        Weisfeiler and Lehman go cellular: CW networks.
        NeurIPS 2021.
        https://arxiv.org/abs/2106.12575
    """

    def __init__(
        self,
        in_channels_0,
        in_channels_1,
        in_channels_2,
        out_channels,
        conv_1_to_1=None,
        conv_0_to_1=None,
        aggregate_fn=None,
        update_fn=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv_1_to_1 = (
            conv_1_to_1
            if conv_1_to_1 is not None
            else _CWNDefaultFirstConv(in_channels_1, in_channels_2, out_channels)
        )
        self.conv_0_to_1 = (
            conv_0_to_1
            if conv_0_to_1 is not None
            else _CWNDefaultSecondConv(in_channels_0, in_channels_1, out_channels)
        )
        self.aggregate_fn = (
            aggregate_fn if aggregate_fn is not None else _CWNDefaultAggregate()
        )
        self.update_fn = (
            update_fn
            if update_fn is not None
            else _CWNDefaultUpdate(out_channels, out_channels)
        )

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        adjacency_0,
        incidence_2,
        incidence_1_t,
    ):
        r"""Forward pass.

        The forward pass was initially proposed in [1]_.
        Its equations are given in [2]_ and graphically illustrated in [3]_.

        The forward pass of this layer is composed of two convolutional steps
        that are followed by an aggregation step and a final update step.

        1. The first convolution between r-cells through (r+1)-cells exploits
        upper-adjacency neighborhood matrix and co-boundary matrix:

        ..  math::
            \begin{align*}
            &🟥 \quad m_{y \rightarrow \{z\} \rightarrow x}^{(r \rightarrow r' \rightarrow r)}
                = M_{\mathcal{L}\uparrow}(h_x^{t,(r)}, h_y^{t,(r)}, h_z^{t,(r')})\\
            &🟧 \quad m_x^{(r \rightarrow r' \rightarrow r)}
                = \text{AGG}_{y \in \mathcal{L}(x)} m_{y \rightarrow \{z\} \rightarrow x}^{(r \rightarrow r' \rightarrow r)}
            \end{align*}

        2. The second convolution from (r-1)-cells to r-cells exploits
        boundary neighborhood matrix:

        .. math::
            \begin{align*}
            &🟥 m_{y \rightarrow x}^{(r'' \rightarrow r)} = M_{\mathcal{B}}(h_x^{t,(r)}, h_y^{t,(r'')})\\
            &🟧 \quad m_x^{(r'' \rightarrow r)}
                = \text{AGG}_{y \in \mathcal{B}(x)} m_{y \rightarrow x}^{(r'' \rightarrow r)}
            \end{align*}

        3. Then, an aggregation step is applied:

        .. math::
            \begin{align*}
            &🟧 \quad m_x^{(r)} = AGG_{\mathcal{N}\_k \in \mathcal{N}} (m_x^k)
            \end{align*}

        4. Finally, an update step is applied:

        .. math::
            \begin{align*}
            &🟦 \quad h_x^{t+1,(r)} = U\left(h_x^{t,(r)}, m_x^{(r)}\right)
            \end{align*}

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_{r-1}_cells, in_channels_{r-1})
            Input features on the (r-1)-cells.
        x_1 : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            Input features on the r-cells.
        x_2 : torch.Tensor, shape = (n_{r+1}_cells, in_channels_{r+1})
            Input features on the (r+1)-cells.
        adjacency_0 : torch.sparse, shape = (n_{r}_cells, n_{r}_cells)
            Neighborhood matrix mapping r-cells to r-cells (A_{up,r}).
        incidence_2 : torch.sparse, shape = (n_{r}_cells, n_{r+1}_cells)
            Neighborhood matrix mapping (r+1)-cells to r-cells (B_{r+1}).
        incidence_1_t : torch.sparse, shape = (n_{r}_cells, n_{r-1}_cells)
            Neighborhood matrix mapping (r-1)-cells to r-cells (B^T_r).

        Returns
        -------
        torch.Tensor, shape = (n_{r}_cells, out_channels)
            Updated representations of the r-cells.

        References
        ----------
        .. [2] Papillon, Sanborn, Hajij, Miolane.
            Equations of topological neural networks (2023).
            https://github.com/awesome-tnns/awesome-tnns/
        .. [3] Papillon, Sanborn, Hajij, Miolane.
            Architectures of topological deep learning: a survey on topological neural networks (2023).
            https://arxiv.org/abs/2304.10031.
        """
        x_convolved_1_to_1 = self.conv_1_to_1(x_1, x_2, adjacency_0, incidence_2)
        x_convolved_0_to_1 = self.conv_0_to_1(x_0, x_1, incidence_1_t)

        x_aggregated = self.aggregate_fn(x_convolved_1_to_1, x_convolved_0_to_1)
        return self.update_fn(x_aggregated, x_1)


class _CWNDefaultFirstConv(nn.Module):
    r"""
    Default implementation of the first convolutional step in CWNLayer.

    The self.forward method of this module must be treated as
    a protocol for the first convolutional step in CWN layer.

    Parameters
    ----------
    in_channels_1 : int
        Dimension of input features on r-cells (edges in case r = 1).
    in_channels_2 : int
        Dimension of input features on (r+1)-cells (faces in case r = 1).
    out_channels : int
        Dimension of output features on r-cells.
    """

    def __init__(self, in_channels_1, in_channels_2, out_channels) -> None:
        super().__init__()
        self.conv_1_to_1 = Conv(
            in_channels_1, out_channels, aggr_norm=False, update_func=None
        )
        self.conv_2_to_1 = Conv(
            in_channels_2, out_channels, aggr_norm=False, update_func=None
        )

    def forward(self, x_1, x_2, adjacency_0, incidence_2):
        r"""Forward pass.

        Parameters
        ----------
        x_1 : torch.Tensor, shape = (n_{r-1}_cells, in_channels_{r-1})
            Input features on the (r-1)-cells.
        x_2 : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            Input features on the r-cells.
        adjacency_0 : torch.sparse, shape = (n_{r}_cells, n_{r}_cells)
            Neighborhood matrix mapping r-cells to r-cells (A_{up,r}).
        incidence_2 : torch.sparse, shape = (n_{r}_cells, n_{r+1}_cells)
            Neighborhood matrix mapping (r+1)-cells to r-cells (B_{r+1}).

        Returns
        -------
        torch.Tensor, shape = (n_{r}_cells, out_channels)
            Updated representations on the r-cells.
        """
        x_up = F.elu(self.conv_1_to_1(x_1, adjacency_0))
        x_coboundary = F.elu(self.conv_2_to_1(x_2, incidence_2))
        return x_up + x_coboundary


class _CWNDefaultSecondConv(nn.Module):
    r"""
    Default implementation of the second convolutional step in CWNLayer.

    The self.forward method of this module must be treated as
    a protocol for the second convolutional step in CWN layer.

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on (r-1)-cells (nodes in case r = 1).
    in_channels_1 : int
        Dimension of input features on r-cells (edges in case r = 1).
    out_channels : int
        Dimension of output features on r-cells.
    """

    def __init__(self, in_channels_0, in_channels_1, out_channels) -> None:
        super().__init__()
        self.conv_0_to_1 = Conv(
            in_channels_0, out_channels, aggr_norm=False, update_func=None
        )

    def forward(self, x_0, x_1, incidence_1_t):
        r"""Forward pass.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_{r-1}_cells, in_channels_{r-1})
            Input features on the (r-1)-cells.
        x_1 : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            Input features on the r-cells.
        incidence_1_t : torch.sparse, shape = (n_{r}_cells, n_{r-1}_cells)
            Neighborhood matrix mapping (r-1)-cells to r-cells (B^T_r).

        Returns
        -------
        torch.Tensor, shape = (n_{r}_cells, out_channels)
            Updated representations on the r-cells.
        """
        return F.elu(self.conv_0_to_1(x_0, incidence_1_t))


class _CWNDefaultAggregate(nn.Module):
    r"""
    Default implementation of an aggregation step in CWNLayer.

    The self.forward method of this module must be treated as
    a protocol for the aggregation step in CWN layer.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        r"""Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            Representations on the r-cells produced by the first convolutional step.
        y : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            Representations on the r-cells produced by the second convolutional step.

        Returns
        -------
        torch.Tensor, shape = (n_{r}_cells, out_channels)
            Aggregated representations on the r-cells.
        """
        return x + y


class _CWNDefaultUpdate(nn.Module):
    r"""Default implementation of an update step in CWNLayer.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    """

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.transform = nn.Linear(in_channels, out_channels)

    def forward(self, x, x_prev=None):
        r"""Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            New representations on the r-cells obtained after the aggregation step.
        x_prev : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            Original representations on the r-cells passed into the CWN layer.

        Returns
        -------
        torch.Tensor, shape = (n_{r}_cells, out_channels)
            Updated representations on the r-cells.
        """
        return F.elu(self.transform(x))



"""UniGCNII class."""



class UniGCNII(torch.nn.Module):
    """Hypergraph neural network utilizing the UniGCNII layer [1]_ for node-level classification.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    n_layers : int, default=2
        Number of UniGCNII message passing layers.
    alpha : float, default=0.5
        Parameter of the UniGCNII layer.
    beta : float, default=0.5
        Parameter of the UniGCNII layer.
    input_drop : float, default=0.2
        Dropout rate for the input features.
    layer_drop : float, default=0.2
        Dropout rate for the hidden features.
    use_norm : bool, default=False
        Whether to apply row normalization after every layer.
    **kwargs : optional
        Additional arguments for the inner layers.

    References
    ----------
    .. [1] Huang and Yang.
        UniGNN: a unified framework for graph and hypergraph neural networks.
        IJCAI 2021.
        https://arxiv.org/pdf/2105.00956.pdf
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        n_layers=2,
        alpha=0.5,
        beta=0.5,
        input_drop=0.2,
        layer_drop=0.2,
        use_norm=False,
        **kwargs,
    ):
        super().__init__()
        layers = []

        self.input_drop = torch.nn.Dropout(input_drop)
        self.layer_drop = torch.nn.Dropout(layer_drop)

        self.initial_linear_layer = torch.nn.Linear(in_channels, hidden_channels)

        for i in range(n_layers):
            beta = math.log(alpha / (i + 1) + 1)
            layers.append(
                UniGCNIILayer(
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    alpha=alpha,
                    beta=beta,
                    use_norm=use_norm,
                    **kwargs,
                )
            )

        self.layers = torch.nn.ModuleList(layers)


    def forward(self, x_0, incidence_1):
        """Forward pass through the model.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (num_nodes, in_channels)
            Input features of the nodes of the hypergraph.
        incidence_1 : torch.Tensor, shape = (num_nodes, num_edges)
            Incidence matrix of the hypergraph.
            It is expected that the incidence matrix contains self-loops for all nodes.

        Returns
        -------
        x_0 : torch.Tensor
            Output node features.
        x_1 : torch.Tensor
            Output hyperedge features.
        """
        x_0 = self.input_drop(x_0)
        x_0 = self.initial_linear_layer(x_0)
        x_0 = torch.nn.functional.relu(x_0)
        x_0_skip = x_0
        for layer in self.layers:
            x_0, x_1 = layer(x_0, incidence_1, x_0_skip)
            x_0 = self.layer_drop(x_0)
            x_0 = torch.nn.functional.relu(x_0)

        return x_0, x_1



"""UniGCNII layer implementation."""
import torch

from topomodelx.base.conv import Conv



class UniGCNIILayer(torch.nn.Module):
    r"""
    Implementation of the UniGCNII layer [1]_.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    alpha : float
        The alpha parameter determining the importance of the self-loop (\theta_2).
    beta : float
        The beta parameter determining the importance of the learned matrix (\theta_1).
    use_norm : bool, default=False
        Whether to apply row normalization after the layer.
    **kwargs : optional
        Additional arguments for the layer modules.

    References
    ----------
    .. [1] Huang and Yang.
        UniGNN: a unified framework for graph and hypergraph neural networks.
        IJCAI 2021.
        https://arxiv.org/pdf/2105.00956.pdf
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        alpha: float,
        beta: float,
        use_norm=False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.linear = torch.nn.Linear(in_channels, hidden_channels, bias=False)
        self.conv = Conv(
            in_channels=in_channels,
            out_channels=in_channels,
            with_linear_transform=False,
        )
        self.use_norm = use_norm


    def reset_parameters(self) -> None:
        """Reset the parameters of the layer."""
        self.linear.reset_parameters()



    def forward(self, x_0, incidence_1, x_skip=None):
        r"""Forward pass of the UniGCNII layer.

        The forward pass consists of:
        - two messages, and
        - a skip connection with a learned update function.

        First every hyper-edge sums up the features of its constituent edges:

        .. math::
            \begin{align*}
            & 🟥 \quad m_{y \rightarrow z}^{(0 \rightarrow 1)} = (B^T_1)\_{zy} \cdot h^{t,(0)}_y \\
            & 🟧 \quad m_z^{(0\rightarrow1)} = \sum_{y \in \mathcal{B}(z)} m_{y \rightarrow z}^{(0 \rightarrow 1)}
            \end{align*}

        Second, the second message is normalized with the node and edge degrees:

        .. math::
            \begin{align*}
            & 🟥 \quad m_{z \rightarrow x}^{(1 \rightarrow 0)}  = B_1 \cdot m_z^{(0 \rightarrow 1)} \\
            & 🟧 \quad m_{x}^{(1\rightarrow0)}  = \frac{1}{\sqrt{d_x}}\sum_{z \in \mathcal{C}(x)} \frac{1}{\sqrt{d_z}}m_{z \rightarrow x}^{(1\rightarrow0)} \\
            \end{align*}

        Third, the computed message is combined with skip connections and a linear transformation using hyperparameters alpha and beta:

        .. math::
            \begin{align*}
            & 🟩 \quad m_x^{(0)}  = m_x^{(1 \rightarrow 0)} \\
            & 🟦 \quad m_x^{(0)}  = ((1-\beta)I + \beta W)((1-\alpha)m_x^{(0)} + \alpha \cdot h_x^{t,(0)}) \\
            \end{align*}

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (num_nodes, in_channels)
            Input features of the nodes of the hypergraph.
        incidence_1 : torch.Tensor, shape = (num_nodes, num_edges)
            Incidence matrix of the hypergraph.
            It is expected that the incidence matrix contains self-loops for all nodes.
        x_skip : torch.Tensor, shape = (num_nodes, in_channels)
            Original node features of the hypergraph used for the skip connections.
            If not provided, the input to the layer is used as a skip connection.

        Returns
        -------
        x_0 : torch.Tensor
            Output node features.
        x_1 : torch.Tensor
            Output hyperedge features.
        """
        x_skip = x_0 if x_skip is None else x_skip
        incidence_1_transpose = incidence_1.transpose(0, 1)

        x_1 = self.conv(x_0, incidence_1_transpose)

        node_degree = torch.sum(incidence_1.to_dense(), dim=1)

        epsilon = 1e-8
        node_degree = node_degree + epsilon

        edge_degree = torch.sum(torch.diag(node_degree) @ incidence_1, dim=0)

        edge_degree = edge_degree + epsilon

        x_0 = (1 / torch.sqrt(node_degree).unsqueeze(-1)) * self.conv(
            x_1, incidence_1 @ torch.diag(1 / torch.sqrt(edge_degree))
        )

        x_combined = ((1 - self.alpha) * x_0) + (self.alpha * x_skip)
        x_0 = ((1 - self.beta) * x_combined) + self.beta * self.linear(x_combined)

        if self.use_norm:
            rownorm = x_0.detach().norm(dim=1, keepdim=True)
            scale = rownorm.pow(-1)
            scale[torch.isinf(scale)] = 0.0
            x_0 = x_0 * scale

        return x_0, x_1


