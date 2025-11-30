import torch
from tqdm import tqdm


def train_node_with_grad_monitor(loader, model, loss_fn, optimizer, device, print_grads=False):
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out[batch.train_mask], batch.y[batch.train_mask]) / batch.num_graphs
        loss.backward()

        if print_grads:
            # Monitor gradients for key components
            total_grad_norm = 0.0
            param_count = 0
            
            print("\n=== GRADIENT MONITORING ===")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm += grad_norm ** 2
                    param_count += 1
                    
                    # Print gradients for key components
                    if any(key in name for key in ['gnn', 'k_mlp', 'edge_mlp', 'tnn']):
                        print(f"{name}: grad_norm = {grad_norm:.6f}, param_norm = {param.data.norm(2).item():.6f}")
                else:
                    if any(key in name for key in ['gnn', 'k_mlp', 'edge_mlp', 'tnn']):
                        print(f"{name}: NO GRADIENT!")
            
            total_grad_norm = (total_grad_norm ** 0.5)
            print(f"Total gradient norm: {total_grad_norm:.6f}")
            print(f"Loss: {loss.item():.6f}")
            print("========================\n")

        optimizer.step()
        total_loss = loss.item()
    return total_loss / len(loader)


def train_node(loader, model, loss_fn, optimizer, device):
    return train_node_with_grad_monitor(loader, model, loss_fn, optimizer, device, print_grads=False)


@torch.no_grad()
def evaluate_node(model, loader, loss_fn, device, mask, evaluator=None):
    model.eval()
    total_loss = 0
    accuracy = 0
    y_pred = []
    y_true = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)[batch[mask]]
        if evaluator is not None:
            y_pred.append(out[:, 1].unsqueeze(-1))
            y_true.append(batch.y)

        loss = loss_fn(out, batch.y[batch[mask]]) / batch.num_graphs
        total_loss += loss.item()
        pred = out.argmax(-1)
        accuracy = pred.eq(batch.y[batch[mask]]).sum().item() / batch[mask].sum().item()

    return total_loss / len(loader), accuracy