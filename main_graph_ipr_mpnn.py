import argparse
import time

import torch
from torch import tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ogb.graphproppred import Evaluator
from torchinfo import summary

from ipr_mpnn.models.get_model import get_model
from model.tnn_with_lifting_graph_classific import TNN_KNN_MLP_G
from train import train, evaluate
from dataset.dataset_handler import choose_dataset


from utils import parse_args, set_seed, args_canonize, args_unify, Config
import torch.nn as nn
import os

import psutil
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
# import os
# os.environ["TORCH_COMPILE"] = "0"
train_losses = []
test_accuracies = []
train_accuracies = []
triangle_counts = []  # Add this list to store triangle counts

train_times = []
test_times = []
torch.autograd.set_detect_anomaly(True)
import tempfile

def define_path(dataset):
    if dataset in ["nci1", "nci109", "proteins","mutag"]:
        return "tudatasets"
    elif dataset == "zinc":
        return "zinc"
    elif dataset == "molhiv":
        return "ogb"
if __name__ == '__main__':

    process = psutil.Process()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.reset_peak_memory_stats()
    _, parser = parse_args()
    args, opts = parser.parse_known_args()
    config = Config()
    config_path = f"ipr_mpnn/configs/{define_path(args.dataset.lower())}/{args.dataset.lower()}.yaml"
    config.load(config_path, recursive=True)
    config.update(opts)
    args = args_unify(args_canonize(config))
    args.update(parser.parse_args().__dict__)
    data, num_features, num_classes = choose_dataset(args, device)
    train_loader = data[0]
    val_loader = data[1]
    test_loader = data[2]


    diff_lifting = True if args.lifting == "diffLifting" else False
    model = get_model(args, device)
    # model = torch.compile(model, backend="eager")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)




    def train_eval(model, train_loader, val_loader, test_loader, loss_fn, optimizer, evaluator, device):
        train_loss = train(train_loader, model, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device, evaluator)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device, evaluator)
        return train_loss, val_loss, val_acc, test_loss, test_acc


    train_losses = []
    test_losses = []
    test_accuracies = []
    val_losses = []
    val_accuracies = []
    epochs_no_improve=0
    # print(
    #     "Number of parameters:",
    #     sum(p.numel() for p in model.parameters() ),
    # )
    summary(model)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        min_lr=1e-6,
        patience=args.lr_decay_patience,
    )
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    if args.dataset == "ZINC":
        loss_fn = torch.nn.L1Loss(reduction='sum')
    evaluator = None
    if args.dataset == "ogbg-molhiv":
        evaluator = Evaluator(args.dataset)

    
    k_vs = []  # list to track chosen k_v for each epoch

    for epoch in range(1, args.max_epochs):
        if epoch <= 30:
            start_train = time.time()
            train_loss, val_loss, val_acc, test_loss, test_acc = train_eval(
                model,
                train_loader,
                val_loader,
                test_loader,
                loss_fn,
                optimizer,
                evaluator,
                device
            )
            end_train = time.time()
            train_times.append(end_train - start_train)

            # For test time, measure only the test phase
            start_test = time.time()
            _, test_acc_only = evaluate(model, test_loader, loss_fn, device, evaluator)
            end_test = time.time()
            test_times.append(end_test - start_test)
        else:
            train_loss, val_loss, val_acc, test_loss, test_acc = train_eval(
                model,
                train_loader,
                val_loader,
                test_loader,
                loss_fn,
                optimizer,
                evaluator,
                device
            )

        test_accuracies.append(test_acc)
        test_losses.append(test_loss)  # test losses

        val_accuracies.append(val_acc)
        val_losses.append(val_loss)  # test losses

        train_losses.append(train_loss)  # train losses


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

    gpu_allocated = torch.cuda.memory_allocated() / 1024 ** 2  # MB
    gpu_reserved = torch.cuda.memory_reserved() / 1024 ** 2  # MB
    gpu_peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(process.memory_info().rss / 1024 ** 3, "GB")  # memória real usada
    results = {
        "train_losses": tensor(train_losses),
        "test_accuracies": tensor(test_accuracies),
        "test_losses": tensor(test_losses),
        "val_accuracies": tensor(val_accuracies),
        "val_losses": tensor(val_losses),
        "train_times": train_times,
        "test_times": test_times,
        "cpu_memory": process.memory_info().rss,
        "gpu_stats": {
            "memory_allocated_MB": torch.cuda.memory_allocated(),
            "memory_reserved_MB": torch.cuda.memory_reserved(),
            "memory_peak_MB": torch.cuda.max_memory_allocated(),
        },
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