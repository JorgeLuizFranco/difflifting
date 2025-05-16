import torch
from tqdm import tqdm


def train(loader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y
                       ) / batch.num_graphs
        loss.backward()
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(f"No gradient for {name}")
        #     elif param.grad.abs().mean() < 1e-10:
        #         print(f"Near-zero gradient for {name}")
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, evaluator=None):
    model.eval()
    total_loss = 0
    total_correct = 0
    y_pred = []
    y_true = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        if evaluator is not None:
            y_pred.append(out[:, 1].unsqueeze(-1))
            y_true.append(batch.y)

        loss = loss_fn(out.squeeze(), batch.y.squeeze()) / batch.num_graphs
        total_loss += loss.item()
        if not isinstance(loss_fn, torch.nn.L1Loss):
            total_correct += (out.argmax(dim=-1) == batch.y.squeeze()).sum().item()
    if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
        accuracy = total_correct / loader.dataset.len()
    else:
        accuracy = -total_loss / len(loader)
    if evaluator is not None:
        accuracy = evaluator.eval({"y_pred": torch.cat(y_pred, dim = 0), "y_true": torch.cat(y_true, dim = 0)})[evaluator.eval_metric]
    return total_loss / len(loader), accuracy
