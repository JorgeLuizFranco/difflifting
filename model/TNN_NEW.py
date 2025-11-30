import torch
import torch.nn.functional as F
from torch import nn

class CINLayer(nn.Module):
    """CWN layer with ε‑residual update (in the spirit of CIN/GIN)."""
    def __init__(
        self,
        in_channels_0: int,
        in_channels_1: int,
        in_channels_2: int,
        out_channels: int,
        conv_1_to_1: nn.Module = None,
        conv_0_to_1: nn.Module = None,
        aggregate_fn: nn.Module = None,
        update_fn: nn.Module = None,
        eps: float = 0.0,
        train_eps: bool = False,
        **kwargs,
    ):
        super().__init__()
        # convolution from r→r via (r+1) and r→r via (r-1):
        self.conv_1_to_1 = conv_1_to_1 or _CWNDefaultFirstConv(in_channels_1, in_channels_2, out_channels)
        self.conv_0_to_1 = conv_0_to_1 or _CWNDefaultSecondConv(in_channels_0, in_channels_1, out_channels)
        self.aggregate_fn  = aggregate_fn  or _CWNDefaultAggregate()
        self.update_fn     = update_fn     or _CWNDefaultUpdate(out_channels, out_channels)

        # ε parameter (fixed or trainable)
        self.initial_eps = eps
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        # reset ε
        self.eps.data.fill_(self.initial_eps)
        # if your submodules have their own reset, call them here:
        for module in (self.conv_1_to_1, self.conv_0_to_1, self.aggregate_fn, self.update_fn):
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        adjacency_0: torch.sparse.Tensor,
        incidence_2: torch.sparse.Tensor,
        incidence_1_t: torch.sparse.Tensor,
    ) -> torch.Tensor:
        # 1) conv from r-cells via (r+1)-cells
        m1 = self.conv_1_to_1(x_1, x_2, adjacency_0, incidence_2)   # shape: [n_r, out_channels]
        # 2) conv from (r-1)-cells
        m2 = self.conv_0_to_1(x_0, x_1, incidence_1_t)              # shape: [n_r, out_channels]
        # 3) aggregate
        m = self.aggregate_fn(m1, m2)                              # shape: [n_r, out_channels]
        # 4) ε‑residual update akin to GIN/CIN
        return self.update_fn(m + (1.0 + self.eps) * x_1)
