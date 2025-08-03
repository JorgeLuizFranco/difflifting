import os.path as osp

import torch
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.data import Batch
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.utils import degree, to_undirected
from torch_geometric.datasets import ZINC, TUDataset
import torch_geometric.transforms as T
from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import Planetoid, Coauthor, WikipediaNetwork, WebKB
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.loader import DataLoader

from dataset.interrank_positional_encoding import AddRandomWalkPEInterrank
from preprocessing.equal_gauss_features.equal_gaus_features import EqualGausFeatures
from preprocessing.one_hot_degree_features.transforms import OneHotDegreeFeatures, NodeDegrees
from tools.collate import collate_fn
from tools.lifting.clique_lifting import SimplicialCliqueLifting
from tools.lifting.khop import SimplicialKHopLifting
from tools.lifting.hypergraph import HypergraphKHopLifting
from tools.lifting.hypergraph import HypergraphKNNLifting
from tools.lifting.neighboorhood_complex import NeighborhoodComplexLifting
from tools.lifting.cycle_lifting import CellCycleLifting
from tools.lifting.discrete_lifting import DiscreteConfigurationComplexLifting
from tools.lifting.kernel import HypergraphKernelLifting
from tools.normalize import normalize_matrix

NODES_PREDICTION_DATASET = ["Cora", "Citeseer", "Pubmed", "karate",]
COAUTHOR_DATASETS = ["CS", "Physics"]
WEBKBDatasets = ["Cornell", "Texas", "Wisconsin"]
WIKIPEDIADatasets= ["chameleon", "crocodile", "squirrel"]
HETEROPHILIC_DATASETS = WEBKBDatasets + WIKIPEDIADatasets
NODES_PREDICTION_DATASET = NODES_PREDICTION_DATASET + COAUTHOR_DATASETS + WEBKBDatasets + WIKIPEDIADatasets

LIFTINGS = {
    "SimplicialCliqueLifting":SimplicialCliqueLifting,
    "NeighborhoodComplexLifting": NeighborhoodComplexLifting,
    "SimplicialKHopLifting":SimplicialKHopLifting,
    "CellCycleLifting": CellCycleLifting,
    "DiscreteConfigurationComplexLifting": DiscreteConfigurationComplexLifting,
    "HypergraphKHopLifting": HypergraphKHopLifting,
    "HypergraphKNNLifting": lambda **kwargs: HypergraphKNNLifting(k_value=kwargs.get("k", 1), **kwargs),
    "HypergraphKernelLifting": lambda **kwargs: HypergraphKernelLifting(**kwargs),


}
PATH = "../DATA/DATASETS"

class FilterConstant(object):
  def __init__(self, dim):
    """Initializes the FilterConstant class.

    Parameters
    ----------
    dim : int
        The number of features to output for each node.
    """
    
    self.dim = dim

  def __call__(self, data):
    """Replace node features with a constant vector of ones.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The input data object.

    Returns
    -------
    torch_geometric.data.Data
        The modified data object with node features replaced by a constant vector of ones.
    """
    data.x = torch.ones(data.num_nodes, self.dim)
    return data


def get_ogb_data(name: str) -> PygGraphPropPredDataset:
    """Loads the OGB dataset specified by name and ensures features are float.

    Args:
        name (str): The name of the OGB dataset to load.

    Returns:
        PygGraphPropPredDataset: The loaded dataset object.
    """
    path = osp.join(osp.dirname(osp.realpath(__file__)), PATH, name)
    dataset = PygGraphPropPredDataset(name=name, root=path)



    return dataset
def get_data_loaders(train_set, val_set=None, test_set=None, batch_size=1):
    """Returns three DataLoaders from the given datasets.

    Args:
        train_set: The dataset to use for the training DataLoader.
        val_set: The dataset to use for the validation DataLoader.
        test_set: The dataset to use for the testing DataLoader.
        batch_size: The batch size to use for the training DataLoader.

    Returns:
        A tuple containing the DataLoaders for training, validation, and testing.
    """
    from torch.utils.data import DataLoader

    train_loader = DataloadDataset(
        train_set,
    )
    train_loader = DataLoader(
        train_loader,
        batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataloadDataset(
        val_set
    )
    valid_loader = DataLoader(
        valid_loader,
        batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataloadDataset(
        test_set
    )
    test_loader = DataLoader(
        test_loader,
        batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader, test_loader


def divide_train_val_test_split(dataset: PygGraphPropPredDataset, args):
    """Returns three DataLoaders for training, validation, and testing from the given dataset.

    Args:
        dataset: The PygGraphPropPredDataset to use.
        batch_size: The batch size to use for the training DataLoader.

    Returns:
        A tuple containing the DataLoaders for training, validation, and testing.
    """
    if dataset.name.startswith("ogbg"):
        split_idx = dataset.get_idx_split()

        train_data = dataset[split_idx["train"]]
        data_val = dataset[split_idx["valid"]]
        data_test = dataset[split_idx["test"]]
        if args.lifting != "diffLifting":
            dataset =  lift_topology(dataset, args)
            train_data = dataset[split_idx["train"]]
            data_val = dataset[split_idx["valid"]]
            data_test = dataset[split_idx["test"]]

        return get_data_loaders(train_data, data_val, data_test, args.batch_size)



def get_graph_classification_dataset(dataset: str, batch_size, args, device, seed=42):
    """Returns DataLoaders for the given dataset.

    Args:
        dataset: The name of the dataset to use.
        batch_size: The batch size for the DataLoader.
        dim: The dimension of the node features. If None, the default is used.
        seed: The random seed for splitting the dataset.

    Returns:
        A tuple containing the DataLoaders for the training, validation, and testing sets.
    """
    if dataset.startswith("ogbg"):
        dataset = get_ogb_data(dataset)
        if args.gnn == "GPS":
            dataset = add_positional_encoding(args, dataset)
        train_loader, val_loader, test_loader = divide_train_val_test_split(dataset, args)
        dataloaders = (train_loader, val_loader, test_loader)

    elif dataset == "ZINC":
        train_set, val_set, test_set = get_zinc(args)
        if args.gnn == "GPS":
            train_set = add_positional_encoding(args, train_set)
            val_set = add_positional_encoding(args, val_set)
            test_set = add_positional_encoding(args, test_set)
        num_nodes_features = train_set.x.shape[1]
        dataloaders = get_data_loaders(train_set,val_set, test_set, batch_size)
        return  dataloaders, num_nodes_features, 1
    else:
        dataset = tu_datasets(dataset, args)
        if args.gnn == "GPS":
            dataset = add_positional_encoding(args, dataset)
        train_set, val_set, test_set = data_split(dataset, seed)
        dataloaders = get_data_loaders(train_set,val_set, test_set, batch_size)

    return dataloaders, dataset.num_features, dataset.num_classes


def ensure_float_features(dataset):
    """
    Converts all features in the dataset to float tensors.

    Args:
        dataset (torch_geometric.data.Dataset): The dataset to process.

    Returns:
        torch_geometric.data.Dataset: The dataset with float features.
    """
    for data in dataset:
        if hasattr(data, 'x') and data.x is not None:
            data.x = data.x.long()
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.long()
    return dataset

def get_zinc(args):
    """Loads the ZINC dataset and returns the training, validation, and test sets.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets.
    """
    path = osp.join(osp.dirname(osp.realpath(__file__)), PATH, "ZINC")
    train_data = ZINC(path, subset=True, split="train")
    data_val = ZINC(path, subset=True, split="val" )
    data_test = ZINC(path, subset=True, split="test")

    if args.lifting != "diffLifting":
        train_data =  lift_topology(train_data, args)
        data_val = lift_topology(data_val, args)
        data_test = lift_topology(data_test, args)
    return train_data, data_val, data_test


def tu_datasets(name,args, no_feat_replacement='constant'):
    """Loads a TUDataset and applies feature replacement if necessary.

    Args:
        name (str): The name of the TU dataset to load.
        no_feat_replacement (str, optional): How to replace missing node features.
            Defaults to 'constant'. Options:
            - 'constant': Replace with a constant vector.
            - 'degree': Replace with one-hot encoding of node degrees.

    Returns:
        TUDataset: The loaded dataset, potentially with transformed features.
    """
    path = osp.join(osp.dirname(osp.realpath(__file__)), PATH, name)
    if name == "IMDB-BINARY":
        dataset = TUDataset(name=name, root=path, transform= T.Compose([NodeDegrees(), OneHotDegreeFeatures()]),use_node_attr=False,)
    elif name == "REDDIT-BINARY":
        dataset = TUDataset(name=name, root=path, transform= T.Compose([EqualGausFeatures(**{"mean": 0, "std": 0.1, "num_features": 10})]),use_node_attr=False,)

    else:
        dataset = TUDataset(name=name, root=path,
                            use_node_attr=False, )

    if args.lifting != "diffLifting":
        return lift_topology(dataset, args)
    return dataset


def lift_topology(dataset, args):
        data_list = []
        max_dim = 0
        for i, d in enumerate(dataset):
            lift_fn = LIFTINGS[args.lifting](signed=args.signed, t=args.t, k=args.k)
            new_data = lift_fn(d)
            for key, value in new_data.items():
                if key.startswith("hodge_laplacian_"):
                    setattr(d, key,normalize_matrix(value, int(key[-1])))
                setattr(d, key,value)
            data_list.append(d)
        dataset.data, dataset.slices = dataset.collate(data_list)
        return dataset

def data_split(dataset, seed):
    """Splits a dataset into training, validation, and test sets using stratified shuffle split.

    Args:
        dataset: The dataset to be split.
        seed: The random seed to ensure reproducibility.

    Returns:
        A tuple containing the training, validation, and test datasets.
    """
    skf_train = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_test_idx = list(skf_train.split(torch.zeros(len(dataset)), dataset.y))[0]
    skf_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = list(skf_val.split(torch.zeros(val_test_idx.size), dataset.y[val_test_idx]))[0]
    train_data = dataset[train_idx]
    val_data = dataset[val_test_idx[val_idx]]
    test_data = dataset[val_test_idx[test_idx]]
    return train_data, val_data, test_data

def remove_duplicated_edges(edge_index):
    """Removes duplicated edges from an edge_index tensor.

    Args:
        edge_index: A tensor of shape (2, num_edges) containing the edges of a graph.

    Returns:
        A tensor of shape (2, num_edges) containing the edges of the graph without duplicates.
    """
    arestas = set()
    for i in range(edge_index.size(1)):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        arestas.add((min(u, v), max(u, v)))
    return torch.tensor(list(arestas), dtype=torch.long).T


def get_node_prediction_dataset(dataset, args,dim=None, seed=42):
    """Loads a dataset for node-level prediction tasks.

    Args:
        dataset (str): The name of the dataset to load. Options: KARATECLUB, Cora, CiteSeer, PubMed.
        dim (int, optional): The dimension of the node features. Defaults to None.
        seed (int, optional): The random seed for splitting the dataset. Defaults to 42.

    Returns:
        A tuple containing the DataLoaders for the training, validation, and testing sets.
    """
    if dataset == "karate":
        dataset = KarateClub()
        if args.lifting == "diffLifting":
            data = dataset[0]
        else:
            data = lift_topology(dataset, args)[0]
        num_train_nodes = int(0.8 * data.num_nodes)
        data.train_mask = torch.zeros(data.num_nodes, dtype=bool)
        data.train_mask[:num_train_nodes] = True
        data.test_mask = ~data.train_mask


    elif dataset=="Cora":
        dataset = Planetoid(root='data', name='cora', split="full", transform=T.NormalizeFeatures())
        if args.gnn == "GPS" or args.sub_gccn_model == "GPS":
            dataset = add_positional_encoding(args, dataset)
        if args.lifting == "diffLifting":
            data = dataset[0]
        else:
            data = lift_topology(dataset, args)[0]

    elif dataset=="Citeseer":
        dataset = Planetoid(root='data', name='CiteSeer', split="full", transform=T.NormalizeFeatures())
        if args.gnn == "GPS" or args.sub_gccn_model == "GPS":
            dataset = add_positional_encoding(args, dataset)
        if args.lifting == "diffLifting":
            data = dataset[0]
        else:
            data = lift_topology(dataset, args)[0]

    elif dataset=="Pubmed":
        dataset = Planetoid(root='data', name='pubmed', split="full", transform=T.NormalizeFeatures())
        if args.gnn == "GPS" or args.sub_gccn_model == "GPS":
            dataset = add_positional_encoding(args, dataset)
        if args.lifting == "diffLifting":
            data = dataset[0]
        else:
            data = lift_topology(dataset, args)[0]
    elif dataset in COAUTHOR_DATASETS:
        dataset = Coauthor(root='data', name=dataset, transform=T.NormalizeFeatures())

        if args.gnn == "GPS" or args.sub_gccn_model == "GPS":
            dataset = add_positional_encoding(args, dataset)
        if args.lifting == "diffLifting":
            data = dataset[0]
        else:
            data = lift_topology(dataset, args)[0]
        data = random_coauthor_amazon_splits(data, dataset.num_classes, None)

    elif dataset in HETEROPHILIC_DATASETS:
        mask_nr = 0
        if dataset in WEBKBDatasets:
            dataset = WebKB(root='data', name=dataset, transform=T.NormalizeFeatures())

            if args.gnn == "GPS" or args.sub_gccn_model == "GPS":
                dataset = add_positional_encoding(args, dataset)
            if args.lifting == "diffLifting":
                data = dataset[0]
            else:

                data = lift_topology(dataset, args)[0]


            data.train_mask = data.train_mask[:, mask_nr]
            data.val_mask = data.val_mask[:, mask_nr]
            data.test_mask = data.test_mask[:, mask_nr]
        elif dataset in WIKIPEDIADatasets:
            dataset = WikipediaNetwork(root='data', name=dataset, transform=T.NormalizeFeatures())
            if args.gnn == "GPS" or args.sub_gccn_model == "GPS":
                dataset = add_positional_encoding(args, dataset)
            if args.lifting == "diffLifting":
                data = dataset[0]
            else:
                data = lift_topology(dataset, args)[0]


            data.train_mask = data.train_mask[:, mask_nr]
            data.val_mask = data.val_mask[:, mask_nr]
            data.test_mask = data.test_mask[:, mask_nr]

    if args.use_dcm_split:
        mask_nr = torch.randint(0, 10, (1,)).item()
        data = cross_validation_split(
            data, dataset_name=args.dataset, curr_seed=mask_nr
        )


    dataloaders = get_data_loaders([data], [data], [data])
    return dataloaders, dataset.num_features, dataset.num_classes

def choose_dataset(args, device):
    """Chooses the appropriate dataset function based on the input data.

    Args:
        data (str): The name of the dataset to load.

    Returns:
        A function that loads the dataset. The function is either
        `get_node_prediction_dataset` or `get_graph_classification_dataset`.
    """
    if args.dataset in NODES_PREDICTION_DATASET:
        return get_node_prediction_dataset(args.dataset, args)
    else:
        return get_graph_classification_dataset(args.dataset, args.batch_size, args, device)

def add_positional_encoding(args, dataset):
    if isinstance(args, dict):
        positional_encoder = AddRandomWalkPEInterrank(walk_length=args.get("positional_walking_len"), attr_name='pe')
        return positional_encoder(dataset)
    else:
        positional_encoder = AddRandomWalkPE(walk_length=args.positional_walking_len, attr_name='pe')

    graph_with_positional_encoder = []
    for graph in dataset:
        graph_with_positional_encoder.append(positional_encoder(graph))
    dataset.data, dataset.slices = dataset.collate(graph_with_positional_encoder)
    return dataset



import torch_geometric

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_coauthor_amazon_splits(data, num_classes, lcc_mask):
    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data


class DataloadDataset(torch_geometric.data.Dataset):
    """Custom dataset to return all the values added to the dataset object.

    Parameters
    ----------
    data_lst : list[torch_geometric.data.Data]
        List of torch_geometric.data.Data objects.
    """

    def __init__(self, data_lst):
        super().__init__()
        self.data_lst = data_lst

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self.data_lst)})"

    def get(self, idx):
        """Get data object from data list.

        Parameters
        ----------
        idx : int
            Index of the data object to get.

        Returns
        -------
        tuple
            Tuple containing a list of all the values for the data and the corresponding keys.
        """
        data = self.data_lst[idx]
        keys = list(data.keys())
        return ([data[key] for key in keys], keys)

    def len(self):
        """Return the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data_lst)

import os.path as osp
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, download_url, InMemoryDataset
from torch_sparse import coalesce


class WikipediaNetworkDCM(InMemoryDataset):
    r"""

    The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average daily traffic of the web page.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"chameleon"`,
            :obj:`"crocodile"`, :obj:`"squirrel"`).
        geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
            pre-processed data as introduced in the `"Geom-GCN: Geometric
            Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
            in which the average monthly traffic of the web page is converted
            into five categories to predict.
            If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
            available.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower()
        assert self.name in ["chameleon", "squirrel"]
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> Union[str, List[str]]:
        return ["out1_node_feature_label.txt", "out1_graph_edges.txt"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], "r") as f:
            data = f.read().split("\n")[1:-1]
        x = [[float(v) for v in r.split("\t")[1].split(",")] for r in data]
        x = torch.tensor(x, dtype=torch.float)
        y = [int(r.split("\t")[2]) for r in data]
        y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], "r") as f:
            data = f.read().split("\n")[1:-1]
            data = [[int(v) for v in r.split("\t")] for r in data]
        edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()

        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
#
#
# class WebKBDCM(InMemoryDataset):
#     r"""
#     The WebKB datasets used in the
#     `"Geom-GCN: Geometric Graph Convolutional Networks"
#     <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
#     Nodes represent web pages and edges represent hyperlinks between them.
#     Node features are the bag-of-words representation of web pages.
#     The task is to classify the nodes into one of the five categories, student,
#     project, course, staff, and faculty.
#     Args:
#         root (string): Root directory where the dataset should be saved.
#         name (string): The name of the dataset (:obj:`"Cornell"`,
#             :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
#         transform (callable, optional): A function/transform that takes in an
#             :obj:`torch_geometric.data.Data` object and returns a transformed
#             version. The data object will be transformed before every access.
#             (default: :obj:`None`)
#         pre_transform (callable, optional): A function/transform that takes in
#             an :obj:`torch_geometric.data.Data` object and returns a
#             transformed version. The data object will be transformed before
#             being saved to disk. (default: :obj:`None`)
#     """
#
#     url = (
#         "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/"
#         "1c4c04f93fa6ada91976cda8d7577eec0e3e5cce/new_data"
#     )
#
#     def __init__(self, root, name, transform=None, pre_transform=None):
#         self.name = name.lower()
#         assert self.name in ["cornell", "texas", "washington", "wisconsin"]
#
#         super(WebKB, self).__init__(root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])
#
#     @property
#     def raw_dir(self):
#         return osp.join(self.root, self.name, "raw")
#
#     @property
#     def processed_dir(self):
#         return osp.join(self.root, self.name, "processed")
#
#     @property
#     def raw_file_names(self):
#         return ["out1_node_feature_label.txt", "out1_graph_edges.txt"]
#
#     @property
#     def processed_file_names(self):
#         return "data.pt"
#
#     def download(self):
#         for name in self.raw_file_names:
#             download_url(f"{self.url}/{self.name}/{name}", self.raw_dir)
#
#     def process(self):
#         with open(self.raw_paths[0], "r") as f:
#             data = f.read().split("\n")[1:-1]
#             x = [[float(v) for v in r.split("\t")[1].split(",")] for r in data]
#             x = torch.tensor(x, dtype=torch.float32)
#
#             y = [int(r.split("\t")[2]) for r in data]
#             y = torch.tensor(y, dtype=torch.long)
#
#         with open(self.raw_paths[1], "r") as f:
#             data = f.read().split("\n")[1:-1]
#             data = [[int(v) for v in r.split("\t")] for r in data]
#             edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
#             edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
#
#         data = Data(x=x, edge_index=edge_index, y=y)
#         data = data if self.pre_transform is None else self.pre_transform(data)
#         torch.save(self.collate([data]), self.processed_paths[0])
#
#     def __repr__(self):
#         return "{}()".format(self.name)
#

def get_hetero_dataset(name):
    if name in ["texas", "wisconsin"]:
        dataset = WebKB(root="data/Hetero", name=name, transform=T.NormalizeFeatures())
    elif name in ["chameleon", "squirrel"]:
        dataset = WikipediaNetwork(
            root="data/Hetero", name=name, transform=T.NormalizeFeatures()
        )
    else:
        raise ValueError(f"dataset {name} not supported in dataloader")

    return dataset


def cross_validation_split(data, dataset_name=None, curr_seed=0):

    loaded_data = np.load(f"./data/{dataset_name}/splits.npz", allow_pickle=True)
    final_splits = loaded_data["splits"].item()

    n_nodes = data.y.shape[0]
    train_indices = torch.as_tensor(final_splits[curr_seed]["Train_idx"])
    val_indices = torch.as_tensor(final_splits[curr_seed]["Test_idx"])
    test_indices = torch.as_tensor(final_splits[curr_seed]["Test_idx"])

    device = data.y.device
    train_mask = torch.zeros(n_nodes, dtype=torch.bool).to(device)
    train_mask[train_indices] = True

    valid_mask = torch.zeros(n_nodes, dtype=torch.bool).to(device)
    valid_mask[val_indices] = True

    test_mask = torch.zeros(n_nodes, dtype=torch.bool).to(device)
    test_mask[test_indices] = True

    data.train_mask = train_mask
    data.val_mask = valid_mask
    data.test_mask = test_mask

    return data