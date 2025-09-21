import argparse
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Set the seed for random number generation to ensure reproducibility.

    This function sets the seed for Python's `random` module, NumPy,
    and PyTorch, ensuring that experiments can be reproduced. It also
    configures PyTorch's CUDA backend to ensure deterministic behavior
    when GPU is available.

    Parameters:
        seed (int): The seed value to be used for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for configuring the DiffLifting GNN tasks.

    Returns:
        argparse.Namespace: An object containing all the command-line arguments
        and their values.

    Available command-line arguments:
        --seed (int): Random seed for reproducibility. Default is 42.
        --gnn (str): Type of GNN to use, options are 'gcn', 'gin', 'linear'. Default is 'gcn'.
        --tnn (str): Type of TNN to use, options are 'san', 'scn', 'sccn'. Default is 'san'.
        --dataset (str): Dataset to use for the task, options include 'ogbg-molhiv',
            'ogbg-molpcba', 'NCI1', 'NCI109', 'IMDB-BINARY', 'ENZYMES', 'CORA', 
            'CITESEER', 'PUBMED'. Default is 'ogbg-molhiv'.
        --lr (float): Learning rate for training. Default is 0.001.
        --batch_size (int): Batch size for training. Default is 32.
        --max_epochs (int): Maximum number of epochs to train the model. Default is 1000.
        --early_stop_patience (int): Patience for early stopping. Default is 40.
        --lr_decay_patience (int): Patience for learning rate decay. Default is 10.
        --hidden_dim (int): Dimension of hidden layers. Default is 64.
        --depth (int): Depth of the network. Default is 2.
        --no-bn: Disable batch normalization if this flag is present.
        --deepset_aggr_type (str): Aggregation type for DeepSet, options are 'sum', 'cat', 'mean'. Default is 'sum'.
        --global_pooling (str): Global pooling method, options are 'sum', 'mean'. Default is 'mean'.
    """
    parser = argparse.ArgumentParser(description="DiffLifting for GNN tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--gnn", type=str, default="GIN", choices=["GIN", "GPS"])
    parser.add_argument("--tnn", type=str, default="TOPOTUNE", choices=["CWN","CIN", "SCN2", "CXN", "UniGCNII", "UniGIN", "UniGCN", "HyperGAT", "TOPOTUNE"])
    parser.add_argument(
        "--dataset",
        type=str,
        default="PROTEINS",
        choices=["Cora", "Citeseer", "Pubmed",   #Classic Node classification datasets
             "CS", "Physics","Cornell", "Texas", "Wisconsin","chameleon", "crocodile", "squirrel", #Heterophilous Graph dataset
                 "ogbg-molhiv", "NCI1", "NCI109", "IMDB-BINARY",  # Graph Classification datasets
                 "REDDIT-BINARY", "ENZYMES", "PROTEINS", "DD", "MUTAG", "ZINC"],
    )
    parser.add_argument(
        "--lifting",
        type=str,
        default="diffLifting",
        choices=["SimplicialCliqueLifting", "SimplicialKHopLifting","CellCycleLifting", "DiscreteConfigurationComplexLifting",  "diffLifting", "HypergraphKHopLifting", "HypergraphKNNLifting", "HypergraphKernelLifting"],
    )
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight Decay.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of tnn layers.")
    parser.add_argument("--num_layers_gnn", type=int, default=2, help="Number of gnn layers ")

    parser.add_argument(
        "--max_epochs", type=int, default=1000, help="Number of epochs to train."
    )
    parser.add_argument(
        "--number_of_mask", type=int, default=1, help="if the dataset is heterophyllic you have to choose from 0 to 9"
    )
    parser.add_argument("--early_stop_patience", type=int, default=50)
    parser.add_argument("--lr_decay_patience", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="results/", help="Log directory")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--gnn_embedding_dim", type=int, default=32)
    parser.add_argument("--k_max", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--graph_transformer_n_heads", type=int, default=4)
    parser.add_argument("--positional_encoder_dim", type=int, default=4)
    parser.add_argument("--positional_walking_len", type=int, default=20)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--no_readout", action='store_true')
    parser.add_argument("--signed", type=bool, default=False)
    parser.add_argument("--use_dcm_split", action='store_true')
    parser.add_argument("--no-bn", dest="bn", action="store_false")
    parser.add_argument(
        "--deepset_aggr_type", type=str, default="sum", choices=["sum", "cat", "mean"]
    )
    parser.add_argument(
        "--sub_gccn_model", type=str, default="GIN", choices=["GAT", "GCN" , "GIN"]
    )
    parser.add_argument(
        "--global_pooling", type=str, default="mean", choices=["sum", "mean"]
    )

    parser.add_argument("--t", type=float, default=5, help="Temperature parameter for the heat kernel.")
    parser.add_argument("--deterministic", action="store_true", help="Run without sampling, use deterministic neighbor and inclusion selection.")

    return parser.parse_args()