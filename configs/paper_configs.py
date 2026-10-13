from __future__ import annotations

from copy import deepcopy


TU_DATASETS = {
    "MUTAG": {"batch_size": 32},
    "ENZYMES": {"batch_size": 32},
    "PROTEINS": {"batch_size": 32},
    "COLLAB": {"batch_size": 32},
    "REDDIT-BINARY": {"batch_size": 8},
    "IMDB-BINARY": {"batch_size": 32},
}

TU_PRESET_MODELS = {
    "tu_gcn_1l": "gcn_1l",
    "tu_gin_1l": "gin_1l",
    "tu_gcn_pgh_1l": "mmcgcn_1l_layer1",
    "tu_gin_pgh_1l": "mmcgin_1l_layer1",
}

PEPTIDES_PRESETS = {
    "peptides_func_gcn": {
        "dataset": "Peptides-func",
        "model_name": "peptuned_gcn_6l",
        "hidden_channels": 235,
        "dropout_rate": 0.3,
        "weight_decay": 5e-4,
        "head_depth": 2,
        "use_pgh": False,
    },
    "peptides_func_gcn_pgh": {
        "dataset": "Peptides-func",
        "model_name": "peptuned_mmcgcn_6l_layer3",
        "hidden_channels": 235,
        "dropout_rate": 0.3,
        "weight_decay": 5e-4,
        "head_depth": 2,
        "use_pgh": True,
    },
    "peptides_func_gine": {
        "dataset": "Peptides-func",
        "model_name": "peptuned_gine_8l",
        "hidden_channels": 160,
        "dropout_rate": 0.3,
        "weight_decay": 5e-4,
        "head_depth": 2,
        "use_pgh": False,
    },
    "peptides_func_gine_pgh": {
        "dataset": "Peptides-func",
        "model_name": "peptuned_mmcgine_8l_layer4",
        "hidden_channels": 160,
        "dropout_rate": 0.3,
        "weight_decay": 5e-4,
        "head_depth": 2,
        "use_pgh": True,
    },
    "peptides_struct_gcn": {
        "dataset": "Peptides-struct",
        "model_name": "peptuned_gcn_6l",
        "hidden_channels": 235,
        "dropout_rate": 0.2,
        "weight_decay": 1e-4,
        "head_depth": 2,
        "use_pgh": False,
    },
    "peptides_struct_gcn_pgh": {
        "dataset": "Peptides-struct",
        "model_name": "peptuned_mmcgcn_6l_layer3",
        "hidden_channels": 235,
        "dropout_rate": 0.2,
        "weight_decay": 1e-4,
        "head_depth": 2,
        "use_pgh": True,
    },
    "peptides_struct_gine": {
        "dataset": "Peptides-struct",
        "model_name": "peptuned_gine_10l",
        "hidden_channels": 145,
        "dropout_rate": 0.2,
        "weight_decay": 1e-4,
        "head_depth": 2,
        "use_pgh": False,
    },
    "peptides_struct_gine_pgh": {
        "dataset": "Peptides-struct",
        "model_name": "peptuned_mmcgine_10l_layer5",
        "hidden_channels": 145,
        "dropout_rate": 0.2,
        "weight_decay": 1e-4,
        "head_depth": 2,
        "use_pgh": True,
    },
}


def available_presets() -> tuple[str, ...]:
    """Return the supported main-table preset names."""
    return tuple(sorted((*TU_PRESET_MODELS, *PEPTIDES_PRESETS)))


def _tu_config(preset: str, dataset: str) -> dict:
    dataset = str(dataset).upper()
    if dataset not in TU_DATASETS:
        allowed = ", ".join(TU_DATASETS)
        raise ValueError(f"Unknown TU dataset {dataset!r}. Choose one of: {allowed}.")

    model_name = TU_PRESET_MODELS[preset]
    batch_size = int(TU_DATASETS[dataset]["batch_size"])
    use_pgh = model_name.startswith("mmc")

    hyperparams = {
        "model_name": [model_name],
        "hidden_channels": [64],
        "num_interp_nodes": [16, 32] if use_pgh else [16],
        "tau": [0.5, 1.0] if use_pgh else [0.5],
        "batch_size": [batch_size],
        "epochs": [100],
        "learning_rate": [0.001],
        "weight_decay": [0.0005],
        "dropout_rate": [0.5],
    }

    model_tag = model_name.replace("_", "-")
    return {
        "dataset": dataset,
        "data_root": "data/tu_struct_features_v1",
        "feature_cache_tag": "tu_struct_features_v1",
        "num_repetitions": 10,
        "train_split": 0.8,
        "val_split": 0.1,
        "base_seed": 42,
        "seed_stride": 2,
        "log_every_epoch": 10,
        "save_checkpoints": False,
        "checkpoint_dir": f"outputs/checkpoints/{dataset.lower()}/{model_tag}",
        "hyperparams": hyperparams,
        "log_dir": "outputs/logs",
        "csv_dir": "outputs/results",
    }


def _peptides_config(preset: str) -> dict:
    settings = PEPTIDES_PRESETS[preset]
    dataset = str(settings["dataset"])
    model_name = str(settings["model_name"])
    use_pgh = bool(settings["use_pgh"])
    dataset_tag = dataset.lower().replace("-", "_")
    model_tag = model_name.replace("_", "-")

    return {
        "dataset": dataset,
        "data_root": "data/peptides_paper_tuned_v1",
        "peptides_recipe": "tonshoff_2023",
        "optimization_profile": "task_tuned_v1",
        "num_repetitions": 10,
        "total_repetitions": 10,
        "train_split": -1,
        "val_split": -1,
        "base_seed": 42,
        "seed_stride": 2,
        "log_every_epoch": 10,
        "save_checkpoints": False,
        "checkpoint_dir": f"outputs/checkpoints/{dataset_tag}/{model_tag}",
        "hyperparams": {
            "model_name": [model_name],
            "hidden_channels": [int(settings["hidden_channels"])],
            "num_interp_nodes": [16],
            "tau": [0.1, 0.5, 1.0, 2.0] if use_pgh else [0.5],
            "batch_size": [200],
            "epochs": [250],
            "learning_rate": [0.001],
            "weight_decay": [float(settings["weight_decay"])],
            "dropout_rate": [float(settings["dropout_rate"])],
            "head_depth": [int(settings["head_depth"])],
        },
        "optim": {
            "optimizer": "adamW",
            "weight_decay": float(settings["weight_decay"]),
            "base_lr": 0.001,
            "max_epoch": 250,
            "scheduler": "cosine_with_warmup",
            "num_warmup_epochs": 5,
            "clip_grad_norm": True,
            "clip_grad_norm_value": 1.0,
        },
        "log_dir": "outputs/logs",
        "csv_dir": "outputs/results",
    }


def get_paper_config(preset: str, dataset: str | None = None) -> dict:
    """Build a fresh configuration for one main-table experiment."""
    preset = str(preset).strip().lower()
    if preset in TU_PRESET_MODELS:
        if dataset is None:
            raise ValueError(f"Preset {preset!r} requires --dataset.")
        config = _tu_config(preset, dataset)
    elif preset in PEPTIDES_PRESETS:
        if dataset is not None:
            expected = PEPTIDES_PRESETS[preset]["dataset"]
            if str(dataset) != expected:
                raise ValueError(
                    f"Preset {preset!r} is fixed to dataset {expected!r}; "
                    f"received {dataset!r}."
                )
        config = _peptides_config(preset)
    else:
        allowed = ", ".join(available_presets())
        raise ValueError(f"Unknown preset {preset!r}. Available presets: {allowed}.")
    return deepcopy(config)
