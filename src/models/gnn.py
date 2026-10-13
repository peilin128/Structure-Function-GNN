from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool


def gin_conv(in_channels: int, out_channels: int) -> GINConv:
    """Create the GIN convolution used by the structural and PGH models."""
    mlp = nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Linear(out_channels, out_channels),
    )
    return GINConv(mlp)


class GNN(nn.Module):
    """Structural-only graph classifier with GCN or GIN convolutions."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        conv_layer=GCNConv,
        num_conv_layers: int = 1,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            conv_layer(
                in_channels if layer_index == 0 else hidden_channels,
                hidden_channels,
            )
            for layer_index in range(num_conv_layers)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x = data.x.float()
        for conv in self.convs:
            x = self.dropout(F.relu(conv(x, data.edge_index)))
        return self.classifier(global_mean_pool(x, data.batch))
