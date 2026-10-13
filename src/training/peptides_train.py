from __future__ import annotations

import csv
import gc
import json
import math
import os
import time
from itertools import product

import numpy as np
import torch
from sklearn.metrics import average_precision_score, r2_score
from torch_geometric.loader import DataLoader

from src.training.model_setup import get_model
from src.training.utils import log_results, set_seed, write_results_to_csv


def evaluate(model, loader, device, dataset_name):
    """Evaluate AP for Peptides-func or MAE/R2 for Peptides-struct."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)

            if dataset_name == "Peptides-func":
                all_preds.append(torch.sigmoid(out).cpu().numpy())
                all_labels.append(data.y.cpu().numpy().astype(float))
            elif dataset_name == "Peptides-struct":
                all_preds.append(out.cpu().numpy())
                all_labels.append(data.y.cpu().numpy())
            else:
                raise ValueError(
                    f"Unsupported dataset_name={dataset_name!r}. "
                    "Expected 'Peptides-func' or 'Peptides-struct'."
                )

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)

    if dataset_name == "Peptides-func":
        per_label_ap = average_precision_score(y_true, y_pred, average=None)
        return float(np.mean(per_label_ap))

    mae = float(np.abs(y_pred - y_true).mean())
    r2_values = [
        r2_score(y_true[:, target], y_pred[:, target])
        for target in range(y_true.shape[1])
    ]
    return mae, float(np.mean(r2_values))


def _build_model_from_hparams(
    dataset,
    hparams: dict,
    dataset_name: str,
):
    return get_model(
        dataset,
        hparams["model_name"],
        int(hparams["hidden_channels"]),
        int(hparams["num_interp_nodes"]),
        tau=float(hparams["tau"]),
        dataset_name=dataset_name,
        dropout_rate=float(hparams["dropout_rate"]),
        head_depth=int(hparams["head_depth"]),
    )


def _model_log_string(hparams: dict) -> str:
    return (
        f"model_name={hparams.get('model_name')}, "
        f"hidden_channels={hparams.get('hidden_channels')}, "
        f"learning_rate={hparams.get('learning_rate')}, "
        f"weight_decay={hparams.get('weight_decay')}, "
        f"dropout_rate={hparams.get('dropout_rate')}, "
        f"head_depth={hparams.get('head_depth')}, "
        f"num_interp_nodes={hparams.get('num_interp_nodes')}, "
        f"tau={hparams.get('tau')}"
    )


def _clone_state_dict_to_cpu(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _is_better_metric(current, best, is_classification: bool):
    return current > best if is_classification else current < best


def _make_train_loader(dataset, batch_size: int, seed: int):
    # A separate generator gives every hyperparameter candidate the same graph
    # order within a repetition, making tau comparisons less confounded.
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    clip_grad_norm_value: float,
):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    start_time = time.time()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(data)
        loss = criterion(out, data.y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(clip_grad_norm_value),
        )
        optimizer.step()

        num_graphs = int(data.num_graphs)
        total_loss += float(loss.item()) * num_graphs
        total_graphs += num_graphs

    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    elif torch.device(device).type == "mps":
        torch.mps.empty_cache()
    gc.collect()

    mean_loss = total_loss / max(total_graphs, 1)
    return mean_loss, time.time() - start_time


def _cosine_with_warmup_scheduler(
    optimizer,
    num_warmup_epochs: int,
    max_epoch: int,
):
    """Match the reference LambdaLR schedule (half cosine after warmup)."""

    def lr_lambda(current_step: int):
        if current_step < num_warmup_epochs:
            return max(
                1e-6,
                float(current_step) / float(max(1, num_warmup_epochs)),
            )
        progress = float(current_step - num_warmup_epochs) / float(
            max(1, max_epoch - num_warmup_epochs)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _summarize_selected_hparams(selected_hparams_per_trial):
    """
    Produce one CSV-compatible summary. Fixed values stay scalar; values selected
    differently across repetitions are stored as a JSON list in trial order.
    """
    keys = set().union(*(params.keys() for params in selected_hparams_per_trial))
    summary = {}

    for key in sorted(keys):
        values = [params.get(key) for params in selected_hparams_per_trial]
        if all(value == values[0] for value in values):
            summary[key] = values[0]
        else:
            summary[key] = json.dumps(values, ensure_ascii=True)
    return summary


def _write_trial_results(config, model_name: str, rows):
    csv_dir = os.path.expanduser(str(config.get("csv_dir", "results")))
    os.makedirs(csv_dir, exist_ok=True)
    path = os.path.join(
        csv_dir,
        f"{config['dataset']}_{model_name}_trial_results.csv",
    )

    fieldnames = [
        "dataset",
        "model_name",
        "trial",
        "seed",
        "selection_metric",
        "best_val_metric",
        "best_epoch",
        "epochs_ran",
        "selected_hidden_channels",
        "selected_learning_rate",
        "selected_weight_decay",
        "selected_dropout_rate",
        "selected_head_depth",
        "num_trainable_parameters",
        "selected_num_interp_nodes",
        "selected_tau",
        "test_average_precision",
        "test_mae",
        "test_r2",
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log_results(f"Per-trial results saved to {os.path.basename(path)}")


def _validate_paper_tuned_optimization(config, hyperparams):
    """Validate the task-tuned optimization recipe used in Table 2."""
    optim_cfg = config["optim"]

    if str(config.get("peptides_recipe", "")).lower() != "tonshoff_2023":
        raise ValueError(
            "Paper-tuned runs require peptides_recipe='tonshoff_2023' so the "
            "task-specific RWSE/LapPE preprocessing is enabled."
        )

    dataset_name = str(config.get("dataset", ""))
    allowed_models = {
        "Peptides-func": {
            "peptuned_gcn_6l",
            "peptuned_gine_8l",
            "peptuned_mmcgcn_6l_layer3",
            "peptuned_mmcgine_8l_layer4",
        },
        "Peptides-struct": {
            "peptuned_gcn_6l",
            "peptuned_gine_10l",
            "peptuned_mmcgcn_6l_layer3",
            "peptuned_mmcgine_10l_layer5",
        },
    }
    if dataset_name not in allowed_models:
        raise ValueError(f"Unsupported paper-tuned dataset: {dataset_name!r}.")
    model_values = hyperparams.get("model_name")
    if not isinstance(model_values, (list, tuple)) or not model_values:
        raise ValueError("hyperparams.model_name must be a non-empty list.")
    invalid_models = set(model_values).difference(allowed_models[dataset_name])
    if invalid_models:
        raise ValueError(
            f"Invalid paper-tuned model(s) for {dataset_name}: "
            f"{sorted(invalid_models)}."
        )

    profile = str(config.get("optimization_profile", "task_tuned_v1")).strip().lower()
    if profile != "task_tuned_v1":
        raise ValueError(
            "Main-table Peptides runs require optimization_profile='task_tuned_v1'."
        )

    if str(optim_cfg.get("optimizer", "")).lower() != "adamw":
        raise ValueError("Paper-tuned Peptides runs require optimizer='adamW'.")
    if float(optim_cfg["base_lr"]) != 0.001:
        raise ValueError("Paper-tuned Peptides runs require base_lr=0.001.")
    if int(optim_cfg["max_epoch"]) != 250:
        raise ValueError("Paper-tuned Peptides runs require max_epoch=250.")
    if str(optim_cfg.get("scheduler", "")).lower() != "cosine_with_warmup":
        raise ValueError(
            "Paper-tuned Peptides runs require scheduler='cosine_with_warmup'."
        )
    if int(optim_cfg.get("num_warmup_epochs", -1)) != 5:
        raise ValueError("Paper-tuned Peptides runs require 5 warmup epochs.")
    if not bool(optim_cfg.get("clip_grad_norm", False)):
        raise ValueError("Paper-tuned Peptides runs require gradient clipping.")
    if float(optim_cfg.get("clip_grad_norm_value", 1.0)) != 1.0:
        raise ValueError("The reference implementation uses clip_grad_norm_value=1.0.")

    task_regularization = {
        "Peptides-func": {
            "weight_decay": 5e-4,
            "dropout_rate": 0.3,
            "head_depth": 2,
        },
        "Peptides-struct": {
            "weight_decay": 1e-4,
            "dropout_rate": 0.2,
            "head_depth": 2,
        },
    }[dataset_name]
    required_fixed_values = {
        "epochs": 250,
        "batch_size": 200,
        "learning_rate": 0.001,
        **task_regularization,
    }

    expected_weight_decay = float(required_fixed_values["weight_decay"])
    if float(optim_cfg["weight_decay"]) != expected_weight_decay:
        raise ValueError(
            f"{profile}/{dataset_name} requires optim.weight_decay="
            f"{expected_weight_decay}; got {optim_cfg['weight_decay']}."
        )

    for key, required in required_fixed_values.items():
        values = hyperparams.get(key)
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"hyperparams.{key} must be a non-empty list.")
        if any(float(value) != float(required) for value in values):
            raise ValueError(
                f"{profile}/{dataset_name} requires {key}={required}; got {values}."
            )


def train_presplit_dataset(
    config,
    device,
    train_dataset,
    val_dataset,
    test_dataset,
    hyperparams,
    num_repetitions,
):
    """
    Train on the official LRGB splits.

    Search protocol:
      1. Within each repetition, train every configured hyperparameter candidate
         for the paper-tuned full 250-epoch schedule.
      2. Keep the best-validation checkpoint for each candidate.
      3. Select the candidate with the best validation AP/MAE.
      4. Evaluate the test set exactly once using that selected checkpoint.

    PGH-only candidates (for example tau) are selected by validation performance
    within the same fixed paper-tuned backbone and optimization regime.
    """
    _validate_paper_tuned_optimization(config, hyperparams)

    optim_cfg = config["optim"]
    dataset_name = config["dataset"]
    is_classification = dataset_name == "Peptides-func"
    selection_metric = "average_precision" if is_classification else "mae"

    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if is_classification
        else torch.nn.L1Loss()
    )

    max_epoch = int(optim_cfg["max_epoch"])
    base_lr = float(optim_cfg["base_lr"])
    default_weight_decay = float(optim_cfg["weight_decay"])
    num_warmup_epochs = int(optim_cfg["num_warmup_epochs"])
    clip_grad_norm_value = float(optim_cfg.get("clip_grad_norm_value", 1.0))
    log_every_epoch = int(config.get("log_every_epoch", 10))

    keys = list(hyperparams.keys())
    combinations = [
        dict(zip(keys, values))
        for values in product(*(hyperparams[key] for key in keys))
    ]
    if not combinations:
        raise ValueError("The hyperparameter grid is empty.")

    final_metrics = []
    selected_hparams_per_trial = []
    best_epoch_values = []
    epochs_ran_values = []
    trial_rows = []

    base_seed = int(config.get("base_seed", 42))
    seed_stride = int(config.get("seed_stride", 2))
    total_repetitions = int(config.get("total_repetitions", num_repetitions))

    configured_indices = config.get("repetition_indices")
    if configured_indices is None:
        repetition_indices = list(range(num_repetitions))
    else:
        if not isinstance(configured_indices, (list, tuple)):
            raise TypeError(
                "config['repetition_indices'] must be a list of zero-based indices."
            )
        repetition_indices = [int(index) for index in configured_indices]

    if len(repetition_indices) != int(num_repetitions):
        raise ValueError(
            "len(repetition_indices) must equal num_repetitions: "
            f"got {len(repetition_indices)} and {num_repetitions}."
        )
    if any(index < 0 or index >= total_repetitions for index in repetition_indices):
        raise ValueError(
            "Every repetition index must satisfy "
            f"0 <= index < total_repetitions ({total_repetitions})."
        )
    if len(set(repetition_indices)) != len(repetition_indices):
        raise ValueError("repetition_indices must not contain duplicates.")

    for repetition_index in repetition_indices:
        trial = repetition_index + 1
        trial_seed = base_seed + seed_stride * repetition_index
        log_results(
            f"Trial {trial}/{total_repetitions}: seed={trial_seed}, "
            f"candidates={len(combinations)}"
        )

        trial_best_metric = -np.inf if is_classification else np.inf
        trial_best_state = None
        trial_best_hparams = None
        trial_best_epoch = None

        for candidate_index, param_dict in enumerate(combinations, start=1):
            # Reset all RNGs so candidates within the same repetition use the same
            # initialization and data-order seed. Tau remains the intended change.
            set_seed(trial_seed)

            batch_size = int(param_dict["batch_size"])
            train_loader = _make_train_loader(
                train_dataset,
                batch_size=batch_size,
                seed=trial_seed,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
            )

            model = _build_model_from_hparams(
                train_dataset,
                param_dict,
                dataset_name,
            ).to(device)
            candidate_num_parameters = _count_trainable_parameters(model)

            candidate_lr = float(
                param_dict.get("learning_rate", base_lr)
            )
            candidate_weight_decay = float(
                param_dict.get("weight_decay", default_weight_decay)
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=candidate_lr,
                weight_decay=candidate_weight_decay,
            )
            scheduler = _cosine_with_warmup_scheduler(
                optimizer,
                num_warmup_epochs=num_warmup_epochs,
                max_epoch=max_epoch,
            )

            candidate_best_metric = -np.inf if is_classification else np.inf
            candidate_best_state = None
            candidate_best_epoch = None

            log_results(
                f"Trial {trial}/{total_repetitions}, candidate "
                f"{candidate_index}/{len(combinations)}: "
                f"{_model_log_string(param_dict)}"
            )

            for epoch in range(1, max_epoch + 1):
                loss, duration = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    clip_grad_norm_value,
                )

                val_result = evaluate(model, val_loader, device, dataset_name)
                val_metric = val_result if is_classification else val_result[0]
                scheduler.step()

                if _is_better_metric(
                    val_metric,
                    candidate_best_metric,
                    is_classification,
                ):
                    candidate_best_metric = val_metric
                    candidate_best_state = _clone_state_dict_to_cpu(model)
                    candidate_best_epoch = epoch

                should_log = (
                    epoch == 1
                    or epoch % log_every_epoch == 0
                    or epoch == max_epoch
                )
                if should_log:
                    log_results(
                        f"Trial {trial}/{total_repetitions}, candidate "
                        f"{candidate_index}/{len(combinations)}, "
                        f"Epoch {epoch}/{max_epoch}: "
                        f"Val {selection_metric}={val_metric:.6f}, "
                        f"Best={candidate_best_metric:.6f} "
                        f"at epoch {candidate_best_epoch}, "
                        f"Loss={loss:.6f}, "
                        f"LR={optimizer.param_groups[0]['lr']:.2e}, "
                        f"Time={duration:.2f}s"
                    )

            if candidate_best_state is None:
                raise RuntimeError(
                    f"No validation checkpoint was saved for candidate {param_dict}."
                )

            if _is_better_metric(
                candidate_best_metric,
                trial_best_metric,
                is_classification,
            ):
                trial_best_metric = candidate_best_metric
                trial_best_state = candidate_best_state
                trial_best_hparams = dict(param_dict)
                trial_best_hparams["num_trainable_parameters"] = candidate_num_parameters
                trial_best_epoch = candidate_best_epoch

            del model, optimizer, scheduler, candidate_best_state
            gc.collect()
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()

        if trial_best_state is None or trial_best_hparams is None:
            raise RuntimeError("Hyperparameter search did not select a model.")

        test_loader = DataLoader(
            test_dataset,
            batch_size=int(trial_best_hparams["batch_size"]),
            shuffle=False,
        )
        selected_model = _build_model_from_hparams(
            train_dataset,
            trial_best_hparams,
            dataset_name,
        ).to(device)
        selected_model.load_state_dict(trial_best_state)
        test_result = evaluate(
            selected_model,
            test_loader,
            device,
            dataset_name,
        )

        trial_best_hparams["best_epoch"] = trial_best_epoch
        trial_best_hparams["epochs_ran"] = max_epoch
        trial_best_hparams["final_max_epoch"] = max_epoch

        selected_hparams_per_trial.append(dict(trial_best_hparams))
        best_epoch_values.append(trial_best_epoch)
        epochs_ran_values.append(max_epoch)
        final_metrics.append(test_result)

        checkpoint_path = ""
        checkpoint_name = ""
        if bool(config.get("save_checkpoints", False)):
            checkpoint_dir = os.path.expanduser(str(config["checkpoint_dir"]))
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_name = (
                f"{dataset_name}_{trial_best_hparams['model_name']}_"
                f"trial{trial:02d}_seed{trial_seed}.pt"
            )
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save(
                {
                    "state_dict": trial_best_state,
                    "dataset": dataset_name,
                    "model_name": trial_best_hparams["model_name"],
                    "trial": trial,
                    "seed": trial_seed,
                    "best_val_metric": trial_best_metric,
                    "best_epoch": trial_best_epoch,
                    "best_hyperparams": trial_best_hparams,
                },
                checkpoint_path,
            )

        if is_classification:
            test_ap = float(test_result)
            test_mae = ""
            test_r2 = ""
            log_results(
                f"Trial {trial}: selected tau={trial_best_hparams['tau']}, "
                f"best Val AP={trial_best_metric:.6f} at epoch {trial_best_epoch}; "
                f"Test AP={test_ap:.6f}; checkpoint={checkpoint_name or 'disabled'}"
            )
        else:
            test_ap = ""
            test_mae = float(test_result[0])
            test_r2 = float(test_result[1])
            log_results(
                f"Trial {trial}: selected tau={trial_best_hparams['tau']}, "
                f"best Val MAE={trial_best_metric:.6f} at epoch {trial_best_epoch}; "
                f"Test MAE={test_mae:.6f}, Test R2={test_r2:.6f}; "
                f"checkpoint={checkpoint_name or 'disabled'}"
            )

        trial_rows.append(
            {
                "dataset": dataset_name,
                "model_name": trial_best_hparams["model_name"],
                "trial": trial,
                "seed": trial_seed,
                "selection_metric": selection_metric,
                "best_val_metric": trial_best_metric,
                "best_epoch": trial_best_epoch,
                "epochs_ran": max_epoch,
                "selected_hidden_channels": trial_best_hparams[
                    "hidden_channels"
                ],
                "selected_learning_rate": trial_best_hparams.get(
                    "learning_rate", base_lr
                ),
                "selected_weight_decay": trial_best_hparams.get(
                    "weight_decay", default_weight_decay
                ),
                "selected_dropout_rate": trial_best_hparams["dropout_rate"],
                "selected_head_depth": trial_best_hparams["head_depth"],
                "num_trainable_parameters": trial_best_hparams[
                    "num_trainable_parameters"
                ],
                "selected_num_interp_nodes": trial_best_hparams[
                    "num_interp_nodes"
                ],
                "selected_tau": trial_best_hparams["tau"],
                "test_average_precision": test_ap,
                "test_mae": test_mae,
                "test_r2": test_r2,
            }
        )

        del selected_model, trial_best_state
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()

    model_name = str(selected_hparams_per_trial[0]["model_name"])
    _write_trial_results(config, model_name, trial_rows)

    avg_best_epoch = float(np.mean(best_epoch_values))
    std_best_epoch = float(np.std(best_epoch_values))
    avg_epochs_ran = float(np.mean(epochs_ran_values))
    std_epochs_ran = float(np.std(epochs_ran_values))
    summary_hparams = _summarize_selected_hparams(selected_hparams_per_trial)

    selected_tau_values = [params["tau"] for params in selected_hparams_per_trial]
    log_results(
        f"Selected tau values across trials: {selected_tau_values}; "
        f"best_epoch={avg_best_epoch:.2f} +/- {std_best_epoch:.2f}"
    )

    if is_classification:
        ap_values = np.asarray(final_metrics, dtype=float)
        write_results_to_csv(
            config,
            avg_final_test_avg_precision=float(ap_values.mean()),
            std_final_test_avg_precision=float(ap_values.std()),
            avg_best_epoch=avg_best_epoch,
            std_best_epoch=std_best_epoch,
            avg_epochs_ran=avg_epochs_ran,
            std_epochs_ran=std_epochs_ran,
            best_hyperparams=summary_hparams,
        )
    else:
        metric_array = np.asarray(final_metrics, dtype=float)
        mae_values = metric_array[:, 0]
        r2_values = metric_array[:, 1]
        write_results_to_csv(
            config,
            avg_final_test_mae=float(mae_values.mean()),
            std_final_test_mae=float(mae_values.std()),
            avg_final_test_r2=float(r2_values.mean()),
            std_final_test_r2=float(r2_values.std()),
            avg_best_epoch=avg_best_epoch,
            std_best_epoch=std_best_epoch,
            avg_epochs_ran=avg_epochs_ran,
            std_epochs_ran=std_epochs_ran,
            best_hyperparams=summary_hparams,
        )
