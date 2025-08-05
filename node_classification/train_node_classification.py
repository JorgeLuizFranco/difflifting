import torch
from tqdm import tqdm


def train_node(loader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out[batch.train_mask], batch.y[batch.train_mask]) / batch.num_graphs
        loss.backward()

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} gradient: {param.grad}")
        #     else:
        #         print(f"{name} has no gradient")
        optimizer.step()

        total_loss = loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_node(model, loader, loss_fn, device, mask,evaluator=None, is_web_kb=False):
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
        if is_web_kb:
            accuracy = pred.eq(batch.y[batch[mask]]).sum().item() / batch[mask].sum().item()
        else:
            accuracy = pred.eq(batch.y[batch[mask]]).sum().item() / batch[mask].shape[0]

    return total_loss / len(loader), accuracy
