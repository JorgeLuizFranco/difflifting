from abc import abstractmethod


import torch
import torch_geometric
from torch_geometric.nn.norm import GraphNorm

class AbstractFeatureEncoder(torch.nn.Module):
    r"""Abstract class to define a custom feature encoder."""

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Forward pass of the feature encoder model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object which should contain x features.
        """



class AllCellFeatureEncoder(AbstractFeatureEncoder):
    r"""Encoder class to apply BaseEncoder.

    The BaseEncoder is applied to the features of higher order
    structures. The class creates a BaseEncoder for each dimension specified in
    selected_dimensions. Then during the forward pass, the BaseEncoders are
    applied to the features of the corresponding dimensions.

    Parameters
    ----------
    in_channels : list[int]
        Input dimensions for the features.
    out_channels : list[int]
        Output dimensions for the features.
    proj_dropout : float, optional
        Dropout for the BaseEncoders (default: 0).
    selected_dimensions : list[int], optional
        List of indexes to apply the BaseEncoders to (default: None).
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        proj_dropout=0,
        selected_dimensions=None,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimensions = (
            selected_dimensions
            if (
                selected_dimensions is not None
            )
            else range(len(self.in_channels))
        )
        for i in self.dimensions:
            setattr(
                self,
                f"encoder_{i}",
                BaseEncoder(
                    self.in_channels[i],
                    self.out_channels,
                    dropout=proj_dropout,
                ),
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, dimensions={self.dimensions})"

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Forward pass.

        The method applies the BaseEncoders to the features of the selected_dimensions.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object which should contain x_{i} features for each i in the selected_dimensions.

        Returns
        -------
        torch_geometric.data.Data
            Output data object with updated x_{i} features.
        """
        if not hasattr(data, "x_0"):
            data.x_0 = data.x

        for i in self.dimensions:
            if hasattr(data, f"x_{i}") and hasattr(self, f"encoder_{i}"):
                batch = None
                if hasattr(data, f"batch_{i}"):
                    batch = getattr(data, f"batch_{i}")
                data[f"x_{i}"] = getattr(self, f"encoder_{i}")(
                    data[f"x_{i}"], batch
                )
        return data


class BaseEncoder(torch.nn.Module):
    r"""Base encoder class used by AllCellFeatureEncoder.

    This class uses two linear layers with GraphNorm, Relu activation function, and dropout between the two layers.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimensions of output features.
    dropout : float, optional
        Percentage of channels to discard between the two linear layers (default: 0).
    """

    def __init__(self, in_channels, out_channels, dropout=0):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_channels, out_channels)
        self.linear2 = torch.nn.Linear(out_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.BN = GraphNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.linear1.in_features}, out_channels={self.linear1.out_features})"

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        r"""Forward pass of the encoder.

        It applies two linear layers with GraphNorm, Relu activation function, and dropout between the two layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of dimensions [N, in_channels].
        batch : torch.Tensor
            The batch vector which assigns each element to a specific example.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [N, out_channels].
        """
        x = self.linear1(x)
        if batch is not None:
            if (batch.shape[0] > 0):
                x = self.BN(x, batch=batch)
        else:
          x = self.BN(x)
        x = self.dropout(self.relu(x))
        x = self.linear2(x)
        return x
