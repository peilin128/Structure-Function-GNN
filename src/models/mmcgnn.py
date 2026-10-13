from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from src.models.graph_processing import connectivity_module, pgh_output_dim


class MMCGNN(nn.Module):
    """Fuse a one-layer structural embedding with its PGH representation."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_interp_nodes: int,
        conv_layer=GCNConv,
        dropout_rate: float = 0.5,
        tau: float = 0.5,
    ):
        super().__init__()
        self.num_interp_nodes = int(num_interp_nodes)
        self.tau = float(tau)

        self.conv = conv_layer(in_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout_rate)

        self.pgh_encoder = nn.Sequential(
            nn.Linear(pgh_output_dim(self.num_interp_nodes), hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, data):
        hidden = F.relu(self.conv(data.x.float(), data.edge_index))
        structural_nodes = self.dropout(hidden)

        pgh_vector = connectivity_module(
            hidden,
            data.batch,
            self.num_interp_nodes,
            self.tau,
            training=self.training,
        )
        pgh_embedding = self.pgh_encoder(pgh_vector)
        structural_embedding = global_mean_pool(structural_nodes, data.batch)
        return self.classifier(
            torch.cat([structural_embedding, pgh_embedding], dim=1)
        )
