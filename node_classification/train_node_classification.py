import torch
from tqdm import tqdm
import torch.nn.functional as F

def train_node(loader, model, loss_fn, optimizer, device, args=None,avg_accuracy=None, **kwargs):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        if args.lifting=="DCMLifting":
            pred, edgelprobs, polylprobs = model(batch)
            pred = pred[batch.train_mask]
            train_lab = batch.y[batch.train_mask]
            tr_loss = loss_fn(pred, train_lab)

            corr_pred = (pred.argmax(-1) == train_lab.argmax(-1)).float().detach()
            if avg_accuracy is None:
               avg_accuracy = torch.ones_like(corr_pred) * 0.5

            tredgelprobs = edgelprobs[batch.train_mask]
            point_w = avg_accuracy - corr_pred
            graph_loss = kwargs["graph_loss_reg"] * (point_w * tredgelprobs).mean()
            tr_loss = tr_loss + graph_loss

            if polylprobs is not None:
                trpolylprobs = polylprobs[batch.train_mask]
                poly_loss = kwargs["poly_loss_reg"] * (point_w * trpolylprobs).mean()
                tr_loss = tr_loss + poly_loss
            tr_loss.backward()
            optimizer.step()

            return tr_loss.item()/ len(loader)

        out = model(batch)
        loss = loss_fn(out[batch.train_mask], batch.y[batch.train_mask]) / batch.num_graphs
        loss.backward()
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(f"No gradient for {name}")
        #     elif param.grad.abs().mean() < 1e-10:
        #         print(f"Near-zero gradient for {name}")
        optimizer.step()
        total_loss = loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_node(model, loader, loss_fn, device, mask,args,evaluator=None):
    model.eval()
    total_loss = 0
    accuracy = 0
    y_pred = []
    y_true = []
    for batch in loader:
        batch = batch.to(device)
        if args.lifting == "DCMLifting":
            out, _, _ = model(batch)
            out = out[batch[mask]]


        else:
            out = model(batch)[batch[mask]]
        if evaluator is not None:
            y_pred.append(out[:, 1].unsqueeze(-1))
            y_true.append(batch.y)

        loss = loss_fn(out, batch.y[batch[mask]]) / batch.num_graphs
        total_loss += loss.item()
        pred = out.argmax(-1)
        accuracy = pred.eq(batch.y[batch[mask]]).sum().item() / batch[mask].sum().item()

    return total_loss / len(loader), accuracy

