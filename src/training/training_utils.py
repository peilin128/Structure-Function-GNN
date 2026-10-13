from __future__ import annotations

import gc
import os
import time
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch_geometric.loader import DataLoader

from src.training.model_setup import get_model
from src.training.utils import log_results, set_seed, write_results_to_csv


def _count_hyperparameter_combinations(hyperparams) -> int:
    """Count the Cartesian-product size of the hyperparameter grid."""
    num_combinations = 1
    for key, values in hyperparams.items():
        if not isinstance(values, (list, tuple)):
            raise TypeError(
                f"hyperparams[{key!r}] must be a list/tuple, got {type(values).__name__}."
            )
        if len(values) == 0:
            raise ValueError(f"hyperparams[{key!r}] is empty.")
        num_combinations *= len(values)
    return num_combinations


def _first_hyperparameter_combination(hyperparams) -> dict:
    """Return the only/first hyperparameter combination."""
    return {key: values[0] for key, values in hyperparams.items()}


def _build_model_from_hparams(dataset, hparams: dict):
    """Build one main-table model from one hyperparameter dictionary."""
    return get_model(
        dataset,
        hparams["model_name"],
        hparams["hidden_channels"],
        hparams["num_interp_nodes"],
        tau=hparams["tau"],
        dropout_rate=float(hparams.get("dropout_rate", 0.5)),
    )


def _model_log_string(hparams: dict) -> str:
    """Make the selected main-table settings visible in log files."""
    fields = [
        f"model_name: {hparams.get('model_name')}",
        f"num_interp_nodes: {hparams.get('num_interp_nodes')}",
        f"tau: {hparams.get('tau')}",
    ]

    return ", ".join(fields)


def _clone_state_dict_to_cpu(model):
    """
    Clone the current model checkpoint to CPU.

    This avoids keeping multiple best checkpoints on GPU memory.
    """
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def train(model, train_loader, optimizer, device):
    total_loss = 0.0
    correct = 0
    total = 0
    model.train()
    start_time = time.time()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        # model(data) returns raw logits.
        loss = F.cross_entropy(out, data.y)

        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    elif torch.device(device).type == "mps":
        torch.mps.empty_cache()

    gc.collect()

    acc = correct / total if total > 0 else 0.0
    return total_loss, acc, time.time() - start_time


def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

    acc = correct / total if total > 0 else 0.0
    return acc


def evaluate_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)

            # model(data) returns raw logits, not log-probabilities.
            # Convert logits to normalized class probabilities for AP/AUC metrics.
            probs = F.softmax(out, dim=1)

            preds = probs.argmax(dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    all_probs_np = np.asarray(all_probs)
    labels_np = np.asarray(all_labels)

    num_classes = all_probs_np.shape[1]
    unique_labels = np.unique(labels_np)

    try:
        if num_classes == 2:
            if len(unique_labels) < 2:
                roc_auc = np.nan
                avg_precision = np.nan
            else:
                roc_auc = roc_auc_score(labels_np, all_probs_np[:, 1])
                avg_precision = average_precision_score(labels_np, all_probs_np[:, 1])

        else:
            classes = np.arange(num_classes)

            if len(unique_labels) < num_classes:
                roc_auc = np.nan
                avg_precision = np.nan
            else:
                labels_bin = label_binarize(labels_np, classes=classes)

                roc_auc = roc_auc_score(
                    labels_np,
                    all_probs_np,
                    labels=classes,
                    average="macro",
                    multi_class="ovr",
                )

                avg_precision = average_precision_score(
                    labels_bin,
                    all_probs_np,
                    average="macro",
                )

    except Exception as e:
        print(f"Caught metric error: {e!r}")
        roc_auc = np.nan
        avg_precision = np.nan

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
    }


def train_from_scratch(
    config, dataset, device, num_repetitions, train_ratio, val_ratio, hyperparams
):
    """Train and evaluate one TUDataset main-table configuration."""

    best_hyperparams_list = []
    final_test_accuracies = []
    final_test_precision = []
    final_test_recall = []
    final_test_f1 = []
    final_test_roc_auc = []
    final_test_avg_precision = []
    best_epoch_values = []
    epochs_ran_values = []

    n = len(dataset)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    test_size = n - train_size - val_size

    num_combinations = _count_hyperparameter_combinations(hyperparams)
    log_every_epoch = int(config.get("log_every_epoch", 1))
    base_seed = int(config.get("base_seed", 42))
    seed_stride = int(config.get("seed_stride", 2))

    save_checkpoints = bool(config.get("save_checkpoints", False))
    checkpoint_dir = os.path.expanduser(
        str(config.get("checkpoint_dir", "checkpoints"))
    )

    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_results("Best-trial checkpoint saving is enabled.")


    for i in range(num_repetitions):
        iteration = i + 1
        trial_seed = base_seed + i * seed_stride
        set_seed(trial_seed)

        log_results(
            f"Trial {iteration}/{num_repetitions}: "
            "Starting random train-validation-test split"
        )

        split_generator = torch.Generator().manual_seed(trial_seed)

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=split_generator,
        )

        # Save the exact graph indices for optional checkpoint metadata.
        train_indices = list(train_dataset.indices)
        val_indices = list(val_dataset.indices)
        test_indices = list(test_dataset.indices)

        log_results(
            f"Sizes -> Train: {len(train_dataset)}, "
            f"Validation: {len(val_dataset)}, Test: {len(test_dataset)}"
        )

        hyperparam_keys = list(hyperparams.keys())
        hyperparam_values = [hyperparams[k] for k in hyperparam_keys]

        if num_combinations == 1:
            candidate_hparams = [
                _first_hyperparameter_combination(hyperparams)
            ]
            log_results(
                "Only one hyperparameter combination; validation checkpoint "
                f"selection remains enabled: {candidate_hparams[0]}"
            )
        else:
            candidate_hparams = [
                dict(zip(hyperparam_keys, params))
                for params in product(*hyperparam_values)
            ]

        best_val_acc = -np.inf
        best_hparams = None
        best_state_dict = None

        for param_dict in candidate_hparams:
            batch_size = int(param_dict["batch_size"])
            candidate_train_dataset = train_dataset
            candidate_val_dataset = val_dataset

            # A dedicated generator keeps mini-batch permutations reproducible.
            loader_generator = torch.Generator().manual_seed(
                trial_seed + 100_000
            )

            train_loader = DataLoader(
                candidate_train_dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=loader_generator,
                num_workers=0,
            )
            val_loader = DataLoader(
                candidate_val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )

            # Re-seed before each candidate so initialization is reproducible.
            set_seed(trial_seed)

            # Use the original TUDataset object for model metadata.
            model = _build_model_from_hparams(dataset, param_dict)
            model.to(device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(param_dict["learning_rate"]),
                weight_decay=float(param_dict["weight_decay"]),
            )

            epochs = int(param_dict["epochs"])
            best_config_val_acc = -np.inf
            best_config_state_dict = None
            best_config_epoch = None
            epochs_completed = 0
            metrics_val = None

            log_results(
                f"Trial {iteration}/{num_repetitions}: Starting hyperparameter configuration "
                f"{param_dict}"
            )

            for epoch in range(epochs):
                current_epoch = epoch + 1

                loss, train_acc, duration = train(
                    model,
                    train_loader,
                    optimizer,
                    device,
                )

                epochs_completed = current_epoch

                metrics_val = evaluate_metrics(model, val_loader, device)
                val_acc = metrics_val["accuracy"]

                if val_acc > best_config_val_acc:
                    best_config_val_acc = val_acc
                    best_config_state_dict = _clone_state_dict_to_cpu(model)
                    best_config_epoch = current_epoch

                should_log = (
                    current_epoch == 1
                    or current_epoch % log_every_epoch == 0
                    or current_epoch == epochs
                )
                if should_log:
                    log_results(
                        f"Trial {iteration}/{num_repetitions}, Epoch {current_epoch}/{epochs}: "
                        f"{_model_log_string(param_dict)}, "
                        f"Train Acc: {train_acc:.6f}, "
                        f"Val Acc: {val_acc:.6f}, "
                        f"Best Val Acc So Far: {best_config_val_acc:.6f} at Epoch {best_config_epoch}, "
                        f"Loss: {loss:.6f}, "
                        f"Time per epoch: {duration:.2f} sec"
                    )

            if best_config_state_dict is None:
                raise RuntimeError(
                    f"No best checkpoint was saved for hyperparameter configuration {param_dict}. "
                    "Check that epochs > 0 and the validation loader is non-empty."
                )

            if best_config_val_acc > best_val_acc:
                best_val_acc = best_config_val_acc
                best_hparams = dict(param_dict)
                best_hparams["best_epoch"] = best_config_epoch
                best_hparams["epochs_ran"] = epochs_completed
                best_state_dict = best_config_state_dict

            log_results(
                f"Trial {iteration}/{num_repetitions}: Completed hyperparameter configuration "
                f"{param_dict} with best Val Acc: {best_config_val_acc:.6f} "
                f"at Epoch {best_config_epoch}; epochs_ran={epochs_completed}"
            )

        if best_hparams is None or best_state_dict is None:
            raise RuntimeError(
                "Validation search finished but no best checkpoint was selected. "
                "Check that epochs > 0 and the validation loader is non-empty."
            )

        log_results(
            f"Trial {iteration}/{num_repetitions}: Best hyperparameters: "
            f"{best_hparams} with best Validation Accuracy: {best_val_acc:.6f}; "
            f"best_epoch={best_hparams['best_epoch']}; "
            f"epochs_ran={best_hparams['epochs_ran']}"
        )

        best_hyperparams_list.append(best_hparams)
        best_epoch_values.append(best_hparams["best_epoch"])
        epochs_ran_values.append(best_hparams["epochs_ran"])

        final_test_dataset = test_dataset

        test_loader = DataLoader(
            final_test_dataset,
            batch_size=int(best_hparams["batch_size"]),
            shuffle=False,
            num_workers=0,
        )

        final_model = _build_model_from_hparams(dataset, best_hparams)
        final_model.to(device)
        final_model.load_state_dict(best_state_dict)

        final_metrics = evaluate_metrics(
            final_model,
            test_loader,
            device,
        )

        if save_checkpoints:
            dataset_name = str(config["dataset"])
            model_name = str(best_hparams["model_name"])

            checkpoint_name = (
                f"{dataset_name}_{model_name}_"
                f"trial_{iteration:02d}_seed_{trial_seed}.pt"
            )
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            temporary_path = checkpoint_path + ".tmp"

            checkpoint_payload = {
                "checkpoint_version": 2,
                "dataset": dataset_name,
                "model_name": model_name,
                "trial": iteration,
                "trial_seed": trial_seed,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "test_indices": test_indices,
                "best_hparams": dict(best_hparams),
                "best_val_accuracy": float(best_val_acc),
                "final_test_metrics": {
                    key: float(value)
                    for key, value in final_metrics.items()
                },
                "model_state_dict": best_state_dict,
            }

            torch.save(checkpoint_payload, temporary_path)
            os.replace(temporary_path, checkpoint_path)

            log_results(
                f"Trial {iteration}/{num_repetitions}: "
                f"Saved checkpoint {checkpoint_name}"
            )

        final_test_accuracies.append(final_metrics["accuracy"])
        final_test_precision.append(final_metrics["precision"])
        final_test_recall.append(final_metrics["recall"])
        final_test_f1.append(final_metrics["f1"])
        final_test_roc_auc.append(final_metrics["roc_auc"])
        final_test_avg_precision.append(final_metrics["average_precision"])

        log_results(
            f"Trial {iteration}/{num_repetitions}: Final Test Metrics: "
            f"{_model_log_string(best_hparams)}, "
            f"Acc: {final_metrics['accuracy']:.6f}, "
            f"Precision: {final_metrics['precision']:.6f}, "
            f"Recall: {final_metrics['recall']:.6f}, "
            f"F1: {final_metrics['f1']:.6f}, "
            f"ROC-AUC: {final_metrics['roc_auc']:.6f}, "
            f"Avg Precision: {final_metrics['average_precision']:.6f}, "
            f"Best Epoch: {best_hparams['best_epoch']}, "
            f"Epochs Ran: {best_hparams['epochs_ran']}"
        )

    avg_final_test_acc = np.mean(final_test_accuracies)
    std_final_test_acc = np.std(final_test_accuracies)
    avg_final_test_precision = np.mean(final_test_precision)
    std_final_test_precision = np.std(final_test_precision)
    avg_final_test_recall = np.mean(final_test_recall)
    std_final_test_recall = np.std(final_test_recall)
    avg_final_test_f1 = np.mean(final_test_f1)
    std_final_test_f1 = np.std(final_test_f1)
    avg_final_test_roc_auc = np.mean(final_test_roc_auc)
    std_final_test_roc_auc = np.std(final_test_roc_auc)
    avg_final_test_avg_precision = np.mean(final_test_avg_precision)
    std_final_test_avg_precision = np.std(final_test_avg_precision)
    avg_best_epoch = np.mean(best_epoch_values)
    std_best_epoch = np.std(best_epoch_values)
    avg_epochs_ran = np.mean(epochs_ran_values)
    std_epochs_ran = np.std(epochs_ran_values)

    log_results(
        f"Final Average Test Accuracy across {num_repetitions} trials: "
        f"{avg_final_test_acc:.6f} +/- {std_final_test_acc:.6f}"
    )

    log_results(
        f"Final Average Precision: {avg_final_test_precision:.6f} +/- {std_final_test_precision:.6f}, "
        f"Recall: {avg_final_test_recall:.6f} +/- {std_final_test_recall:.6f}, "
        f"F1: {avg_final_test_f1:.6f} +/- {std_final_test_f1:.6f}, "
        f"ROC-AUC: {avg_final_test_roc_auc:.6f} +/- {std_final_test_roc_auc:.6f}, "
        f"Avg Precision: {avg_final_test_avg_precision:.6f} +/- {std_final_test_avg_precision:.6f}, "
        f"Best Epoch: {avg_best_epoch:.2f} +/- {std_best_epoch:.2f}, "
        f"Epochs Ran: {avg_epochs_ran:.2f} +/- {std_epochs_ran:.2f}"
    )

    write_results_to_csv(
        config,
        avg_final_test_acc,
        std_final_test_acc,
        best_hyperparams=best_hyperparams_list[-1],
        avg_final_test_precision=avg_final_test_precision,
        std_final_test_precision=std_final_test_precision,
        avg_final_test_recall=avg_final_test_recall,
        std_final_test_recall=std_final_test_recall,
        avg_final_test_f1=avg_final_test_f1,
        std_final_test_f1=std_final_test_f1,
        avg_final_test_roc_auc=avg_final_test_roc_auc,
        std_final_test_roc_auc=std_final_test_roc_auc,
        avg_final_test_avg_precision=avg_final_test_avg_precision,
        std_final_test_avg_precision=std_final_test_avg_precision,
        avg_best_epoch=avg_best_epoch,
        std_best_epoch=std_best_epoch,
        avg_epochs_ran=avg_epochs_ran,
        std_epochs_ran=std_epochs_ran,
    )
