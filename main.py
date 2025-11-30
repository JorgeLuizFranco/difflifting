import argparse
import numpy
import torch
from torch import tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ogb.graphproppred import Evaluator
from torchinfo import summary

from model.dcm import DCMModule
from model.tnn_with_lifiting import TNN_KNN_MLP_N
from model.tnn_with_lifting_graph_classific import TNN_KNN_MLP_G
from node_classification.train_node_classification import train_node, evaluate_node
from train import train, evaluate
from dataset.dataset_handler import choose_dataset

from utils import parse_args, set_seed
import torch.nn as nn
import os

train_losses = []
test_accuracies = []
train_accuracies = []
triangle_counts = []  # Add this list to store triangle counts

torch.autograd.set_detect_anomaly(True)
import tempfile

# torch.set_default_tensor_type("torch.cuda.FloatTensor")
# torch.set_float32_matmul_precision("high")
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()
    set_seed(args.seed)
    print(args.__dict__)
    data, num_features, num_classes = choose_dataset(args, device)
    train_loader = data[0]
    val_loader = data[1]
    test_loader = data[2]

    diff_lifting = True if args.lifting == "diffLifting" else False
    avg_accuracy=None
    if args.lifting=="DCMLifting":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        torch.set_float32_matmul_precision("high")
        config = {
            "metric": {"name": "val_acc", "goal": "maximize"},
            "seed": args.seed,
            "data_seed": 0,
            "hsize": 32,
            "n_pre": 1,
            "n_post": 1,
            "n_conv": 1,
            "n_dgm_layers": 2,
            "dropout": 0.5,
            "lr": 0.01,
            "use_gcn": True,
            "k": 4,
            "graph_loss_reg": 1,
            "poly_loss_reg": 1,
        }
        hsize = config["hsize"]
        gamma = 50
        std = 0
        hyperparams = {
            "num_features": num_features,
            "num_classes": num_classes,
            "pre_layers": [num_features]
                          + [hsize for _ in range(config["n_pre"])],
            "post_layers": [hsize for _ in range(config["n_post"])]
                           + [num_classes],
            "dgm_layers": [hsize for _ in range(config["n_dgm_layers"] + 1)],
            "conv_layers": [hsize for _ in range(config["n_conv"])],
            "lr": config["lr"],
            "use_gcn": config["use_gcn"],
            "dropout": config["dropout"],
            "k": config["k"],
            "gamma": gamma,
            "std": std,
            "graph_loss_reg": config["graph_loss_reg"],
            "poly_loss_reg": config["poly_loss_reg"],
            "ensemble_steps": 1,
        }
        model = DCMModule(hyperparams)
    else:
        model = TNN_KNN_MLP_N(num_features, args, hidden_dim=args.hidden_dim, num_classes=num_classes,
                              k=6, diff_lifting=diff_lifting, global_pool=args.global_pooling, device=device,
                              tnn_type=args.tnn,
                              num_layers_tnn=args.num_layers, num_layers_gnn=args.num_layers_gnn,
                              embedding_dim=args.gnn_embedding_dim,
                              k_max=args.k_max)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    def train_eval(model, loss_fn, optimizer, evaluator, args, device):
        train_loss = train_node(train_loader, model, loss_fn, optimizer, device, args=args, **config)
        val_loss, val_acc = evaluate_node(model, val_loader, loss_fn, device, "val_mask",args=args, evaluator=evaluator)
        test_loss, test_acc = evaluate_node(model, test_loader, loss_fn, device, "test_mask", args=args, evaluator=evaluator)
        return train_loss, val_loss, val_acc, test_loss, test_acc


    train_losses = []
    test_losses = []
    test_accuracies = []
    val_losses = []
    val_accuracies = []
    epochs_no_improve = 0
    print(
        "Number of parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    summary(model)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        min_lr=1e-6,
        patience=args.lr_decay_patience,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    if args.dataset == "ZINC":
        loss_fn = torch.nn.L1Loss(reduction='sum')
    evaluator = None
    if args.dataset == "ogbg-molhiv":
        evaluator = Evaluator(args.dataset)

    k_vs = []  # list to track chosen k_v for each epoch

    for epoch in range(1, args.max_epochs):
        train_loss, val_loss, val_acc, test_loss, test_acc = train_eval(
            model,
            loss_fn,
            optimizer,
            evaluator,
            args,
            device
        )

        test_accuracies.append(test_acc)
        test_losses.append(test_loss)  # test losses

        val_accuracies.append(val_acc)
        val_losses.append(val_loss)  # test losses

        train_losses.append(train_loss)  # train losses

        # if hasattr(model, 'k_v'):
        #     k_vs.append(torch.mean(model.k_v))
        # else:
        #     k_vs.append(None)
        print(
            f"{epoch:3d}: Train Loss: {train_loss:.3f},"
            f" Val Loss: {val_loss:.3f}, Val Acc: {val_accuracies[-1]:.3f}, "
            f"Test Loss: {test_loss:.3f}, Test Acc: {test_accuracies[-1]:.3f}"
        )

        scheduler.step(val_acc)

        if epoch > 2 and val_accuracies[-1] <= val_accuracies[-2 - epochs_no_improve]:
            epochs_no_improve = epochs_no_improve + 1

        else:
            epochs_no_improve = 0

        if epochs_no_improve >= args.early_stop_patience:
            print("Early stopping!")
            break

    results = {
        "train_losses": tensor(train_losses),
        "test_accuracies": tensor(test_accuracies),
        "test_losses": tensor(test_losses),
        "val_accuracies": tensor(val_accuracies),
        "val_losses": tensor(val_losses),

        "params": {
            "gnn": args.gnn,
            "num_layers_gnn": args.num_layers_gnn,
            "gnn_embedding_dim": args.gnn_embedding_dim,
            "k_max": args.k_max,
        },
    }
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    torch.save(
        results, f"{args.logdir}/{args.dataset}_{args.lifting}_{args.gnn}_{args.tnn}_{args.seed}_k_adaptative.results"
    )
