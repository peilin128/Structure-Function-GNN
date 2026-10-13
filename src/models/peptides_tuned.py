from __future__ import annotations


import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import GCNConv, GINEConv, Linear as PyGLinear, global_mean_pool

from src.models.graph_processing import connectivity_module, pgh_output_dim


VALID_BACKBONES = {"gcn", "gine"}
VALID_DATASETS = {"Peptides-func", "Peptides-struct"}

# Exact trainable parameter counts implied by the released configs and model
# implementation. They reproduce the rounded 486k/491k/488k/492k values in
# the paper. PGH models intentionally exceed these counts and are logged rather
# than constrained because PGH is not part of the reference architecture.
REFERENCE_VANILLA_PARAMETER_COUNTS = {
    ("Peptides-func", "gcn"): 485_741,
    ("Peptides-func", "gine"): 490_966,
    ("Peptides-struct", "gcn"): 488_061,
    ("Peptides-struct", "gine"): 492_301,
}


class RWSEEncoder(nn.Module):
    """BatchNorm over 20 raw RWSE values followed by Linear(20, 28)."""

    def __init__(self):
        super().__init__()
        self.raw_norm = nn.BatchNorm1d(20)
        self.linear = nn.Linear(20, 28)

    def forward(self, rwse: torch.Tensor) -> torch.Tensor:
        if rwse.ndim != 2 or rwse.size(1) != 20:
            raise ValueError(
                f"Expected pestat_RWSE with shape [num_nodes, 20], got {tuple(rwse.shape)}."
            )
        return self.linear(self.raw_norm(rwse.float()))


class LapPEEncoder(nn.Module):
    """2-layer DeepSet LapPE encoder with output dimension 16."""

    def __init__(self):
        super().__init__()
        self.linear_a = nn.Linear(2, 32)
        self.pe_encoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

    def forward(self, eigvals: torch.Tensor, eigvecs: torch.Tensor) -> torch.Tensor:
        if eigvals.ndim != 3 or eigvals.size(1) != 10 or eigvals.size(2) != 1:
            raise ValueError(
                "Expected EigVals with shape [num_nodes, 10, 1], "
                f"got {tuple(eigvals.shape)}."
            )
        if eigvecs.ndim != 2 or eigvecs.size(1) != 10:
            raise ValueError(
                f"Expected EigVecs with shape [num_nodes, 10], got {tuple(eigvecs.shape)}."
            )

        eigvals = eigvals.float()
        eigvecs = eigvecs.float()

        # Match the reference implementation: one random sign per frequency,
        # shared by all nodes in the current mini-batch.
        if self.training:
            sign_flip = torch.rand(eigvecs.size(1), device=eigvecs.device)
            sign_flip = torch.where(sign_flip >= 0.5, 1.0, -1.0)
            eigvecs = eigvecs * sign_flip.unsqueeze(0)

        pos_enc = torch.cat((eigvecs.unsqueeze(2), eigvals), dim=2)
        empty_mask = torch.isnan(pos_enc)
        pos_enc = pos_enc.masked_fill(empty_mask, 0.0)
        pos_enc = self.linear_a(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)
        pos_enc = pos_enc.masked_fill(empty_mask[:, :, 0].unsqueeze(2), 0.0)
        return pos_enc.sum(dim=1)


class PaperGCNBlock(nn.Module):
    """GCNConv -> GELU -> Dropout -> residual."""

    def __init__(self, hidden_channels: int, dropout_rate: float):
        super().__init__()
        self.conv = GCNConv(hidden_channels, hidden_channels, bias=True)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        return residual + x


class PaperGINEBlock(nn.Module):
    """Reference GINE MLP followed by a full-layer residual connection."""

    def __init__(self, hidden_channels: int, dropout_rate: float):
        super().__init__()
        mlp = nn.Sequential(
            PyGLinear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            PyGLinear(hidden_channels, hidden_channels),
            nn.Dropout(dropout_rate),
        )
        self.conv = GINEConv(mlp)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return x + self.conv(x, edge_index, edge_attr)


class PaperGraphHead(nn.Module):
    """Configurable MLP head with paper-style pre-linear dropout and GELU."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int = 3,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        depth = int(depth)
        if depth < 1:
            raise ValueError(f"head depth must be >= 1, got {depth}.")

        layers: list[nn.Module] = []
        current_dim = in_channels
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Dropout(dropout_rate),
                    nn.Linear(current_dim, hidden_channels, bias=True),
                    nn.GELU(),
                ]
            )
            current_dim = hidden_channels
        layers.extend(
            [
                nn.Dropout(dropout_rate),
                nn.Linear(current_dim, out_channels, bias=True),
            ]
        )
        self.mlp = nn.Sequential(*layers)

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        return self.mlp(graph_embedding)


class PaperTunedPeptidesModel(nn.Module):
    def __init__(
        self,
        *,
        dataset_name: str,
        backbone: str,
        hidden_channels: int,
        out_channels: int,
        num_conv_layers: int,
        dropout_rate: float = 0.1,
        head_depth: int = 3,
        use_pgh: bool = False,
        pgh_layer: int | None = None,
        num_interp_nodes: int = 16,
        tau: float = 0.5,
    ):
        super().__init__()

        dataset_name = str(dataset_name)
        backbone = str(backbone).lower()
        if dataset_name not in VALID_DATASETS:
            raise ValueError(f"Unsupported dataset_name={dataset_name!r}.")
        if backbone not in VALID_BACKBONES:
            raise ValueError(f"Unsupported backbone={backbone!r}.")
        dropout_rate = float(dropout_rate)
        head_depth = int(head_depth)
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                f"dropout_rate must satisfy 0 <= dropout_rate < 1, got {dropout_rate}."
            )
        if head_depth < 1:
            raise ValueError(f"head_depth must be >= 1, got {head_depth}.")


        self.dataset_name = dataset_name
        self.backbone = backbone
        self.hidden_channels = int(hidden_channels)
        self.num_conv_layers = int(num_conv_layers)
        self.use_pgh = bool(use_pgh)
        self.tau = float(tau)

        pe_dim = 28 if dataset_name == "Peptides-func" else 16
        atom_dim = self.hidden_channels - pe_dim
        if atom_dim <= 0:
            raise ValueError(
                f"hidden_channels={self.hidden_channels} must exceed PE dim={pe_dim}."
            )

        self.atom_encoder = AtomEncoder(atom_dim)
        # The reference configs enable a Bond encoder for both GCN and GINE.
        # GCN does not consume edge_attr, but retaining this module mirrors the
        # released architecture/parameterization.
        self.bond_encoder = BondEncoder(self.hidden_channels)
        self.position_encoder = (
            RWSEEncoder() if dataset_name == "Peptides-func" else LapPEEncoder()
        )

        block_cls = PaperGCNBlock if backbone == "gcn" else PaperGINEBlock
        self.layers = nn.ModuleList(
            block_cls(self.hidden_channels, dropout_rate)
            for _ in range(self.num_conv_layers)
        )

        if self.use_pgh:
            if pgh_layer is None:
                raise ValueError("pgh_layer is required when use_pgh=True.")
            self.pgh_layer = int(pgh_layer)
            if self.pgh_layer < 1 or self.pgh_layer > self.num_conv_layers:
                raise ValueError(
                    f"pgh_layer={self.pgh_layer} is invalid for "
                    f"num_conv_layers={self.num_conv_layers}."
                )

            self.adj_feature_dim = pgh_output_dim(num_interp_nodes)
            self.pgh_encoder = nn.Sequential(
                nn.Linear(self.adj_feature_dim, self.hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
            self.pgh_feature_dim = self.hidden_channels
            head_input_dim = self.hidden_channels + self.pgh_feature_dim
        else:
            self.pgh_layer = None
            self.pgh_encoder = None
            self.pgh_feature_dim = 0
            head_input_dim = self.hidden_channels

        self.num_interp_nodes = int(num_interp_nodes)
        self.head = PaperGraphHead(
            in_channels=head_input_dim,
            hidden_channels=self.hidden_channels,
            out_channels=int(out_channels),
            depth=head_depth,
            dropout_rate=dropout_rate,
        )

        # Enforce the published parameter count only for the exact vanilla
        # reference architecture. Task-tuned models use head_depth=2, so their
        # parameter counts are intentionally different.
        is_exact_reference_architecture = (
            not self.use_pgh and dropout_rate == 0.1 and head_depth == 3
        )

        if is_exact_reference_architecture:
            actual_parameters = count_trainable_parameters(self)
            expected_parameters = REFERENCE_VANILLA_PARAMETER_COUNTS[
                (self.dataset_name, self.backbone)
            ]
            if actual_parameters != expected_parameters:
                raise RuntimeError(
                "Paper-tuned vanilla parameter count mismatch: "
                f"expected {expected_parameters:,}, got {actual_parameters:,}. "
                "Check OGB/PyG encoder and layer implementations."
                )

    def _encode_input(self, data):
        atom_features = self.atom_encoder(data.x.long())
        edge_features = self.bond_encoder(data.edge_attr.long())

        if self.dataset_name == "Peptides-func":
            if not hasattr(data, "pestat_RWSE"):
                raise ValueError(
                    "Peptides-func requires precomputed data.pestat_RWSE."
                )
            positional = self.position_encoder(data.pestat_RWSE)
        else:
            if not (hasattr(data, "EigVals") and hasattr(data, "EigVecs")):
                raise ValueError(
                    "Peptides-struct requires precomputed EigVals and EigVecs."
                )
            positional = self.position_encoder(data.EigVals, data.EigVecs)

        x = torch.cat((atom_features, positional), dim=1)
        if x.size(1) != self.hidden_channels:
            raise RuntimeError(
                f"Encoded node width is {x.size(1)}, expected {self.hidden_channels}."
            )
        return x, edge_features

    def forward(self, data):
        x, edge_attr = self._encode_input(data)
        feature_for_pgh = None

        for layer_index, layer in enumerate(self.layers, start=1):
            if self.backbone == "gcn":
                x = layer(x, data.edge_index)
            else:
                x = layer(x, data.edge_index, edge_attr)

            if self.use_pgh and layer_index == self.pgh_layer:
                feature_for_pgh = x

        structural_graph = global_mean_pool(x, data.batch)
        if not self.use_pgh:
            return self.head(structural_graph)

        if feature_for_pgh is None:
            raise RuntimeError("Failed to capture the configured PGH extraction layer.")

        pgh = connectivity_module(
            feature_for_pgh,
            data.batch,
            self.num_interp_nodes,
            self.tau,
            training=self.training,
        )
        pgh = self.pgh_encoder(pgh)
        return self.head(torch.cat((structural_graph, pgh), dim=1))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
