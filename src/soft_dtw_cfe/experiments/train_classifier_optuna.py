"""
Train time-series classifiers with Optuna and save checkpoints in the optuna output layout.

This script mirrors the old latent_flow pipeline in a leaner form:
- loads an aeon dataset
- z-normalizes per channel
- runs Optuna over CNN dropout/lr/weight_decay
- trains the best model for a fixed number of epochs
- writes outputs to: <output_dir>/<dataset>_optuna/
    checkpoints/best_classifier.pt
    results/classifier_trials.csv
    metadata/{experiment_info.json,classifier_optuna.db,full_classifier_metadata.json}

Example:
    uv run python -m soft_dtw_cfe.experiments.train_classifier_optuna \\
        --datasets CBF TwoLeadECG \\
        --classifier-trials 20 \\
        --clf-epochs 80 \\
        --output-dir optuna_experimentsv2
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from soft_dtw_cfe.data.get_datasets import MULTI_DATASETS, UNI_DATASETS, get_data
from soft_dtw_cfe.data.utils import min_max_normalize_train_test
from soft_dtw_cfe.experiments.classifier_optim import ClassifierOptimizer


ALL_DATASETS = sorted(set(UNI_DATASETS + MULTI_DATASETS))


def prepare_loaders(
    dataset: str,
    batch_size: int,
    test_size: float,
    seed: int,
    device: str,
) -> tuple[DataLoader, DataLoader, dict, dict]:
    """Load, normalize, and wrap a dataset into PyTorch loaders."""
    X_train, y_train, X_test, y_test = get_data(dataset, test_size=test_size, seed=seed)
    X_train, X_test, norm_stats = min_max_normalize_train_test(X_train, X_test, return_stats=True)

    Xtr_t = torch.from_numpy(X_train).float().to(device)
    Xte_t = torch.from_numpy(X_test).float().to(device)
    ytr_t = torch.from_numpy(y_train).long().to(device)
    yte_t = torch.from_numpy(y_test).long().to(device)

    train_ds = TensorDataset(Xtr_t, ytr_t)
    test_ds = TensorDataset(Xte_t, yte_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    meta = {
        "dataset": dataset,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "n_classes": int(np.unique(y_train).shape[0]),
        "n_channels": int(X_train.shape[1]),
        "time_steps": int(X_train.shape[-1]),
        "test_size": test_size,
        "seed": seed,
    }

    return train_loader, test_loader, norm_stats, meta


def _ensure_dirs(base: Path) -> dict[str, Path]:
    paths = {
        "root": base,
        "checkpoints": base / "checkpoints",
        "results": base / "results",
        "metadata": base / "metadata",
        "visualizations": base / "visualizations",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _save_metadata(
    meta_dir: Path,
    dataset_meta: dict,
    study: optuna.Study | None,
    best_params: dict,
    train_result: dict,
    args: argparse.Namespace,
    device: str,
) -> None:
    """Persist lightweight JSON summaries alongside the checkpoint."""
    experiment_info = {
        **dataset_meta,
        "batch_size": args.batch_size,
        "device": device,
        "classifier_trials": args.classifier_trials if not args.skip_clf_optim else 0,
        "final_clf_epochs": args.clf_epochs,
        "timestamp": datetime.utcnow().isoformat(),
        "experiment_name": f"{dataset_meta['dataset']}_optuna",
    }
    (meta_dir / "experiment_info.json").write_text(json.dumps(experiment_info, indent=2))

    full_meta = {
        "timestamp": datetime.utcnow().isoformat(),
        "system_info": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "dataset": dataset_meta,
        "best_classifier_params": best_params,
        "train_result": train_result,
        "classifier_optimization": None,
    }

    if study is not None:
        full_meta["classifier_optimization"] = {
            "study_name": study.study_name,
            "n_trials": len(study.trials),
            "n_completed": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial_number": study.best_trial.number,
            "direction": "MAXIMIZE",
        }

    (meta_dir / "full_classifier_metadata.json").write_text(json.dumps(full_meta, indent=2))


def run_dataset(dataset: str, args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    print(f"\n=== {dataset} (device={device}) ===")

    base = Path(args.output_dir) / f"{dataset}_optuna"
    dirs = _ensure_dirs(base)
    ckpt_path = dirs["checkpoints"] / "best_classifier.pt"

    if ckpt_path.exists() and args.skip_if_checkpoint:
        print(f"  ✓ Checkpoint exists at {ckpt_path}, skipping training (use --no-skip-if-checkpoint to override).")
        return

    train_loader, test_loader, norm_stats, dataset_meta = prepare_loaders(
        dataset=dataset,
        batch_size=args.batch_size,
        test_size=args.test_size,
        seed=args.seed,
        device=device,
    )

    clf_optimizer = ClassifierOptimizer(
        train_loader=train_loader,
        test_loader=test_loader,
        in_channels=dataset_meta["n_channels"],
        num_classes=dataset_meta["n_classes"],
        device=device,
    )

    study: optuna.Study | None = None
    best_params = {
        "dropout": args.default_dropout,
        "lr": args.default_lr,
        "weight_decay": args.default_weight_decay,
    }

    if not args.skip_clf_optim:
        storage = f"sqlite:///{(dirs['metadata'] / 'classifier_optuna.db').absolute()}"
        print(f"  Running Optuna for {args.classifier_trials} trials (storage: {storage})...")
        study = clf_optimizer.optimize(
            n_trials=args.classifier_trials,
            study_name=f"{dataset}_optuna_classifier",
            storage=storage,
        )
        best_params = study.best_params
        print(f"  Best params: {best_params} (value={study.best_value:.4f})")

        trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        trials_df.to_csv(dirs["results"] / "classifier_trials.csv", index=False)
    else:
        print("  Skipping Optuna; using default hyperparameters.")

    clf, train_result = clf_optimizer.train_best_model(best_params, epochs=args.clf_epochs)
    ClassifierOptimizer.save_checkpoint(clf, best_params, train_result, ckpt_path)

    # Persist metadata for reproducibility
    _save_metadata(dirs["metadata"], dataset_meta, study, best_params, train_result, args, device)

    # Also save normalization stats so evaluation scripts can reuse them if desired
    norm_stats_json = {k: np.asarray(v).tolist() for k, v in norm_stats.items()}
    (dirs["results"] / "norm_stats.json").write_text(json.dumps(norm_stats_json, indent=2))
    print(f"  Done. Best val acc: {train_result.get('best_val_acc', float('nan')):.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna-based classifier training for Soft-DTW CFE.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        help=f"Datasets to process (default: {', '.join(ALL_DATASETS)})",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--classifier-trials", type=int, default=30, help="Number of Optuna trials.")
    parser.add_argument("--clf-epochs", type=int, default=80, help="Epochs for the final classifier fit.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data used for test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument("--output-dir", type=str, default="optuna_experimentsv2", help="Base output directory.")
    parser.add_argument("--skip-clf-optim", action="store_true", help="Skip Optuna and use default hyperparameters.")
    parser.add_argument(
        "--skip-if-checkpoint",
        action="store_true",
        help="If a checkpoint exists, skip re-training that dataset.",
    )
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--default-dropout", type=float, default=0.1, help="Dropout used when skipping Optuna.")
    parser.add_argument("--default-lr", type=float, default=1e-3, help="LR used when skipping Optuna.")
    parser.add_argument("--default-weight-decay", type=float, default=0.0, help="Weight decay when skipping Optuna.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    for ds in args.datasets:
        try:
            run_dataset(ds, args)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {ds}: {exc}")


if __name__ == "__main__":
    main()
