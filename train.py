import torch
from tqdm import tqdm


def train(loader, model, loss_fn, optimizer, device):
    model.train()
    train_losses = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)

        loss = loss_fn(out.squeeze().float(), batch.y.squeeze())
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    return train_losses

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, evaluator=None):
    model.eval()
    for batch in tqdm(loader):
        batch = batch.to(device)
        out = model(batch)
        loss = loss_fn(out.squeeze().float(), batch.y.squeeze())
        acc = -loss
        if not isinstance(loss_fn, torch.nn.L1Loss):
            acc = (out.argmax(dim=-1) == batch.y.squeeze()).float().mean()

    if evaluator is not None:
        acc = evaluator.eval({"y_pred": out[:, 1].unsqueeze(dim=1), "y_true": batch.y})[
            evaluator.eval_metric
        ]

    return loss, acc