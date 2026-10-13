from __future__ import annotations

import csv
from datetime import datetime
import os
import random
from typing import Any

import numpy as np
import torch


log_path: str | None = None


def init_log_path(path: str) -> None:
    """Set the log file used by ``log_results``."""
    global log_path
    log_path = path


def log_results(message: str) -> None:
    """Write a timestamped message to the active log and standard output."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    if log_path is None:
        raise ValueError("log_path has not been initialized.")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(full_message + "\n")
    print(full_message)


def _get_best_value(
    best_hyperparams: dict[str, Any] | None,
    key: str,
    default: Any = None,
) -> Any:
    if best_hyperparams is None:
        return default
    return best_hyperparams.get(key, default)


def _get_config_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    value = config.get("hyperparams", {}).get(key, default)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def write_results_to_csv(
    config,
    avg_final_test_acc=None,
    std_final_test_acc=None,
    avg_final_test_precision=None,
    std_final_test_precision=None,
    avg_final_test_recall=None,
    std_final_test_recall=None,
    avg_final_test_f1=None,
    std_final_test_f1=None,
    avg_final_test_roc_auc=None,
    std_final_test_roc_auc=None,
    avg_final_test_avg_precision=None,
    std_final_test_avg_precision=None,
    avg_final_test_mae=None,
    std_final_test_mae=None,
    avg_final_test_r2=None,
    std_final_test_r2=None,
    avg_best_epoch=None,
    std_best_epoch=None,
    avg_epochs_ran=None,
    std_epochs_ran=None,
    best_hyperparams=None,
):
    """Append one main-table experiment summary row to a CSV file."""
    csv_dir = os.path.expanduser(str(config.get("csv_dir", "results")))
    os.makedirs(csv_dir, exist_ok=True)

    csv_filename = f"{config['dataset']}_results.csv"
    csv_path = os.path.join(csv_dir, csv_filename)

    header = [
        "timestamp",
        "dataset",
        "num_repetitions",
        "train_split",
        "val_split",
        "model_name",
        "hidden_channels",
        "num_interp_nodes",
        "batch_size",
        "epochs",
        "avg_best_epoch",
        "std_best_epoch",
        "avg_epochs_ran",
        "std_epochs_ran",
        "learning_rate",
        "weight_decay",
        "dropout_rate",
        "head_depth",
        "tau",
        "avg_final_test_acc",
        "std_final_test_acc",
        "avg_final_test_precision",
        "std_final_test_precision",
        "avg_final_test_recall",
        "std_final_test_recall",
        "avg_final_test_f1",
        "std_final_test_f1",
        "avg_final_test_roc_auc",
        "std_final_test_roc_auc",
        "avg_final_test_avg_precision",
        "std_final_test_avg_precision",
        "avg_final_test_mae",
        "std_final_test_mae",
        "avg_final_test_r2",
        "std_final_test_r2",
    ]

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        config["dataset"],
        config.get("num_repetitions"),
        config.get("train_split"),
        config.get("val_split"),
        _get_best_value(best_hyperparams, "model_name"),
        _get_best_value(
            best_hyperparams,
            "hidden_channels",
            _get_config_value(config, "hidden_channels"),
        ),
        _get_best_value(
            best_hyperparams,
            "num_interp_nodes",
            _get_config_value(config, "num_interp_nodes"),
        ),
        _get_best_value(
            best_hyperparams,
            "batch_size",
            _get_config_value(config, "batch_size"),
        ),
        _get_best_value(
            best_hyperparams,
            "epochs",
            _get_config_value(config, "epochs"),
        ),
        avg_best_epoch,
        std_best_epoch,
        avg_epochs_ran,
        std_epochs_ran,
        _get_best_value(
            best_hyperparams,
            "learning_rate",
            _get_config_value(config, "learning_rate"),
        ),
        _get_best_value(
            best_hyperparams,
            "weight_decay",
            _get_config_value(config, "weight_decay"),
        ),
        _get_best_value(
            best_hyperparams,
            "dropout_rate",
            _get_config_value(config, "dropout_rate"),
        ),
        _get_best_value(
            best_hyperparams,
            "head_depth",
            _get_config_value(config, "head_depth"),
        ),
        _get_best_value(
            best_hyperparams,
            "tau",
            _get_config_value(config, "tau"),
        ),
        avg_final_test_acc,
        std_final_test_acc,
        avg_final_test_precision,
        std_final_test_precision,
        avg_final_test_recall,
        std_final_test_recall,
        avg_final_test_f1,
        std_final_test_f1,
        avg_final_test_roc_auc,
        std_final_test_roc_auc,
        avg_final_test_avg_precision,
        std_final_test_avg_precision,
        avg_final_test_mae,
        std_final_test_mae,
        avg_final_test_r2,
        std_final_test_r2,
    ]

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

    log_results(f"Results saved to {csv_filename}")


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and available PyTorch accelerators."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
