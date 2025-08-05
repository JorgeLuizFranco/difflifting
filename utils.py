import argparse
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DiffLifting for GNN tasks")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--gnn", type=str, default="GIN", choices=["GIN", "GPS"])
    parser.add_argument("--tnn", type=str, default="TOPOTUNE", choices=["CWN", "SCN2", "CXN", "UniGCNII", "UniGIN", "UniGCN", "HyperGAT", "TOPOTUNE"])
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        choices=[
            "Cora", "Citeseer", "Pubmed", "CS", "Physics", "Cornell", "Texas", "Wisconsin",
            "chameleon", "crocodile", "squirrel", "ogbg-molhiv", "NCI1", "NCI109", "IMDB-BINARY",
            "REDDIT-BINARY", "ENZYMES", "PROTEINS", "DD", "MUTAG", "ZINC"
        ]
    )
    parser.add_argument(
        "--lifting",
        type=str,
        default="diffLifting",
        choices=["SimplicialCliqueLifting", "SimplicialKHopLifting","CellCycleLifting", "DiscreteConfigurationComplexLifting",  "diffLifting", "HypergraphKHopLifting", "HypergraphKNNLifting", "HypergraphKernelLifting"],
    )

    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of TNN layers.")
    parser.add_argument("--num_layers_gnn", type=int, default=2, help="Number of GNN layers.")
    parser.add_argument("--max_epochs", type=int, default=1000, help="Max number of training epochs.")
    parser.add_argument("--number_of_mask", type=int, default=1, help="Mask number for heterophilic datasets.")
    parser.add_argument("--early_stop_patience", type=int, default=50, help="Early stopping patience.")
    parser.add_argument("--lr_decay_patience", type=int, default=10, help="LR scheduler patience.")
    parser.add_argument("--logdir", type=str, default="results/", help="Log directory.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden layer size.")
    parser.add_argument("--gnn_embedding_dim", type=int, default=32, help="GNN embedding dimension.")
    parser.add_argument("--k_max", type=int, default=3, help="Maximum k value.")
    parser.add_argument("--k", type=int, default=3, help="Value of k.")
    parser.add_argument("--graph_transformer_n_heads", type=int, default=4, help="Number of heads in transformer.")
    parser.add_argument("--positional_encoder_dim", type=int, default=4, help="Dimension of positional encoder.")
    parser.add_argument("--positional_walking_len", type=int, default=20, help="Length of positional walk.")
    parser.add_argument("--depth", type=int, default=2, help="Network depth.")

    parser.add_argument("--no_readout", action="store_true", help="Disable readout layer.")
    parser.add_argument("--signed", type=bool, default=False, help="Use signed Laplacian.")
    parser.add_argument("--use_dcm_split", action="store_true", help="Use DCM data split.")
    parser.add_argument("--no-bn", dest="bn", action="store_false", help="Disable batch norm.")

    parser.add_argument("--deepset_aggr_type", type=str, default="sum", choices=["sum", "cat", "mean"], help="Aggregation type for DeepSet.")
    parser.add_argument("--sub_gccn_model", type=str, default="GAT", choices=["GAT", "GCN", "GIN", "GPS"], help="Sub-GCCN model.")
    parser.add_argument(
        "--topo_tune_neighboors",
        type=str,
        nargs='+',
        default=["up_laplacian-0","down_laplacian-1","down_incidence-1","up_laplacian-1","down_incidence-2", "down_laplacian-2"],
        help="TopoTune neighborhood types"
    )
    parser.add_argument("--sub_gccn_model_n_layers", type=int, default=4, help="Sub-GCCN model layers.")
    parser.add_argument("--global_pooling", type=str, default="mean", choices=["sum", "mean"], help="Global pooling type.")
    parser.add_argument("--t", type=float, default=5, help="Temperature parameter for heat kernel.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic computations.")

    return parser.parse_args()
