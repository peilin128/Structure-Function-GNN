from __future__ import annotations

import argparse
import os
import time
from copy import deepcopy
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.datasets import LRGBDataset, TUDataset
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx

from configs.paper_configs import available_presets, get_paper_config
from src.data.peptides_posenc import PeptidesPositionEncodingTransform
from src.training.peptides_train import train_presplit_dataset
from src.training.training_utils import train_from_scratch
from src.training.utils import init_log_path, log_results


class AddOrConcatStructuralFeatures(BaseTransform):
    """Append five structural statistics to existing node attributes."""

    def __call__(self, data):
        graph = to_networkx(data, to_undirected=True)
        degree = dict(graph.degree())
        clustering = nx.clustering(graph)
        betweenness = nx.betweenness_centrality(graph, normalized=True)
        closeness = nx.closeness_centrality(graph)
        pagerank = nx.pagerank(graph)

        structural_features = torch.tensor(
            [
                [
                    degree[node],
                    clustering[node],
                    betweenness[node],
                    closeness[node],
                    pagerank[node],
                ]
                for node in range(data.num_nodes)
            ],
            dtype=torch.float,
        )

        if getattr(data, "x", None) is None:
            data.x = structural_features
        else:
            node_features = data.x.float()
            if node_features.dim() == 1:
                node_features = node_features.view(-1, 1)
            data.x = torch.cat([node_features, structural_features], dim=1)
        return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one GCN, GIN, GINE, or PGH main-table experiment."
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=available_presets(),
        help="Built-in main-table experiment preset.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print the available preset names and exit.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name. Required for TUDataset presets.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        help="Dataset cache directory override.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        help="Root directory for logs, result tables, and optional checkpoints.",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save validation-selected checkpoints.",
    )
    return parser.parse_args()


def _load_experiment_config(args: argparse.Namespace) -> dict:
    if args.list_presets:
        print("\n".join(available_presets()))
        raise SystemExit(0)
    if args.preset is None:
        raise SystemExit("Provide --preset. Use --list-presets to inspect presets.")

    config = get_paper_config(args.preset, dataset=args.dataset)
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.save_checkpoints:
        config["save_checkpoints"] = True

    if args.output_root is not None:
        output_root = Path(args.output_root).expanduser()
        config["log_dir"] = str(output_root / "logs")
        config["csv_dir"] = str(output_root / "results")
        config["checkpoint_dir"] = str(output_root / "checkpoints")

    return config


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_datasets(config: dict):
    dataset_name = str(config["dataset"])
    data_root = Path(config.get("data_root", "data")).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)

    if dataset_name in {"Peptides-func", "Peptides-struct"}:
        pe_mode = "rwse" if dataset_name == "Peptides-func" else "lappe"
        pre_transform = PeptidesPositionEncodingTransform(mode=pe_mode)
        dataset_kwargs = {
            "root": str(data_root),
            "name": dataset_name,
            "pre_transform": pre_transform,
        }
        return True, (
            LRGBDataset(split="train", **dataset_kwargs),
            LRGBDataset(split="val", **dataset_kwargs),
            LRGBDataset(split="test", **dataset_kwargs),
        )

    dataset = TUDataset(
        root=str(data_root),
        name=dataset_name,
        pre_transform=AddOrConcatStructuralFeatures(),
    )
    return False, dataset


def main() -> None:
    args = _parse_args()
    config = _load_experiment_config(args)
    device = _select_device()

    log_dir = Path(config.get("log_dir", "outputs/logs")).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(config.get("csv_dir", "outputs/results")).expanduser().mkdir(
        parents=True, exist_ok=True
    )

    dataset_name = str(config["dataset"])
    pre_split, datasets = _load_datasets(config)

    num_repetitions = int(config.get("num_repetitions", 10))
    train_size_ratio = float(config.get("train_split", 0.8))
    val_size_ratio = float(config.get("val_split", 0.1))
    hyperparams = config["hyperparams"]

    for model_name in hyperparams["model_name"]:
        current_hyperparams = deepcopy(hyperparams)
        current_hyperparams["model_name"] = [model_name]

        log_filename = (
            f"{dataset_name}_{model_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )
        init_log_path(str(log_dir / log_filename))
        log_results("Experiment metadata")
        log_results(f"Dataset: {dataset_name}")
        log_results(f"Device type: {device.type}")
        log_results(f"Number of repetitions: {num_repetitions}")
        log_results(f"Hyperparameters: {current_hyperparams}")

        if pre_split:
            train_dataset, val_dataset, test_dataset = datasets
            train_presplit_dataset(
                config,
                device,
                train_dataset,
                val_dataset,
                test_dataset,
                current_hyperparams,
                num_repetitions,
            )
        else:
            train_from_scratch(
                config,
                datasets,
                device,
                num_repetitions,
                train_size_ratio,
                val_size_ratio,
                current_hyperparams,
            )


if __name__ == "__main__":
    main()
