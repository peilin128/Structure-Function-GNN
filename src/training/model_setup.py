from __future__ import annotations

import torch
from torch_geometric.nn import GCNConv

from src.models.gnn import GNN, gin_conv
from src.models.mmcgnn import MMCGNN


TU_MODEL_NAMES = {
    "gcn_1l",
    "gin_1l",
    "mmcgcn_1l_layer1",
    "mmcgin_1l_layer1",
}

PEPTIDES_MODEL_NAMES = {
    "peptuned_gcn_6l",
    "peptuned_gine_8l",
    "peptuned_gine_10l",
    "peptuned_mmcgcn_6l_layer3",
    "peptuned_mmcgine_8l_layer4",
    "peptuned_mmcgine_10l_layer5",
}


def _unwrap_dataset(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        return dataset.dataset
    return dataset


def _infer_out_channels(dataset) -> int:
    if hasattr(dataset, "num_classes") and int(dataset.num_classes) > 0:
        return int(dataset.num_classes)
    target = dataset[0].y
    return int(target.shape[-1]) if target.ndim > 0 else 1


def _build_peptides_model(
    *,
    dataset,
    model_name: str,
    hidden_channels: int,
    num_interp_nodes: int,
    tau: float,
    dataset_name: str | None,
    dropout_rate: float,
    head_depth: int,
):
    tokens = model_name.split("_")
    family = tokens[1]
    use_pgh = family.startswith("mmc")
    backbone = family[3:] if use_pgh else family
    num_conv_layers = int(tokens[2].removesuffix("l"))
    pgh_layer = int(tokens[3].removeprefix("layer")) if use_pgh else None

    resolved_dataset_name = dataset_name or getattr(dataset, "name", None)
    if resolved_dataset_name not in {"Peptides-func", "Peptides-struct"}:
        raise ValueError(
            "Peptides models require dataset_name='Peptides-func' or "
            "'Peptides-struct'."
        )

    expected_layers, expected_hidden = {
        ("Peptides-func", "gcn"): (6, 235),
        ("Peptides-func", "gine"): (8, 160),
        ("Peptides-struct", "gcn"): (6, 235),
        ("Peptides-struct", "gine"): (10, 145),
    }[(resolved_dataset_name, backbone)]
    if num_conv_layers != expected_layers:
        raise ValueError(
            f"{resolved_dataset_name}/{backbone} requires {expected_layers} layers, "
            f"got {num_conv_layers}."
        )
    if int(hidden_channels) != expected_hidden:
        raise ValueError(
            f"{resolved_dataset_name}/{backbone} requires hidden={expected_hidden}, "
            f"got {hidden_channels}."
        )

    from src.models.peptides_tuned import PaperTunedPeptidesModel

    return PaperTunedPeptidesModel(
        dataset_name=resolved_dataset_name,
        backbone=backbone,
        hidden_channels=int(hidden_channels),
        out_channels=10 if resolved_dataset_name == "Peptides-func" else 11,
        num_conv_layers=num_conv_layers,
        dropout_rate=float(dropout_rate),
        head_depth=int(head_depth),
        use_pgh=use_pgh,
        pgh_layer=pgh_layer,
        num_interp_nodes=int(num_interp_nodes),
        tau=float(tau),
    )


def get_model(
    dataset,
    model_name: str,
    hidden_channels: int,
    num_interp_nodes: int,
    tau: float = 0.5,
    dataset_name: str | None = None,
    dropout_rate: float = 0.5,
    head_depth: int = 1,
):
    """Construct one model used by a built-in main-table preset."""
    dataset = _unwrap_dataset(dataset)
    name = str(model_name).lower()

    if name in PEPTIDES_MODEL_NAMES:
        return _build_peptides_model(
            dataset=dataset,
            model_name=name,
            hidden_channels=hidden_channels,
            num_interp_nodes=num_interp_nodes,
            tau=tau,
            dataset_name=dataset_name,
            dropout_rate=dropout_rate,
            head_depth=head_depth,
        )

    if name not in TU_MODEL_NAMES:
        allowed = ", ".join(sorted((*TU_MODEL_NAMES, *PEPTIDES_MODEL_NAMES)))
        raise ValueError(f"Unknown main-table model {model_name!r}. Use one of: {allowed}.")

    out_channels = _infer_out_channels(dataset)
    in_channels = int(dataset.num_node_features)
    hidden_channels = int(hidden_channels)
    dropout_rate = float(dropout_rate)

    if name == "gcn_1l":
        return GNN(
            in_channels,
            hidden_channels,
            out_channels,
            conv_layer=GCNConv,
            num_conv_layers=1,
            dropout_rate=dropout_rate,
        )
    if name == "gin_1l":
        return GNN(
            in_channels,
            hidden_channels,
            out_channels,
            conv_layer=gin_conv,
            num_conv_layers=1,
            dropout_rate=dropout_rate,
        )
    if name == "mmcgcn_1l_layer1":
        return MMCGNN(
            in_channels,
            hidden_channels,
            out_channels,
            num_interp_nodes=int(num_interp_nodes),
            conv_layer=GCNConv,
            dropout_rate=dropout_rate,
            tau=float(tau),
        )
    if name == "mmcgin_1l_layer1":
        return MMCGNN(
            in_channels,
            hidden_channels,
            out_channels,
            num_interp_nodes=int(num_interp_nodes),
            conv_layer=gin_conv,
            dropout_rate=dropout_rate,
            tau=float(tau),
        )

    raise RuntimeError(f"Unreachable model branch for {name!r}.")
