"""
Evaluate the Soft-DTW counterfactual solver using trained classifier checkpoints.

This script expects checkpoints produced by train_classifier_optuna.py and writes
metrics/plots back into the same optuna-style directory.

Example:
    uv run python -m soft_dtw_cfe.experiments.evaluate_solver \\
        --datasets CBF TwoLeadECG \\
        --n-eval-samples 50 \\
        --output-dir optuna_experimentsv2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from soft_dtw_cfe.data.get_datasets import MULTI_DATASETS, UNI_DATASETS, get_data
from soft_dtw_cfe.data.utils import min_max_normalize_train_test
from soft_dtw_cfe.experiments.classifier_optim import ClassifierOptimizer
from soft_dtw_cfe.method.soft_dtw_loss import mean_soft_dtw_to_neighbors
from soft_dtw_cfe.method.solver import CounterfactualSolver, SolverConfig


ALL_DATASETS = sorted(set(UNI_DATASETS + MULTI_DATASETS))


def prepare_data(
    dataset: str, test_size: float, seed: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    X_train, y_train, X_test, y_test = get_data(dataset, test_size=test_size, seed=seed)
    X_train, X_test, norm_stats = min_max_normalize_train_test(X_train, X_test, return_stats=True)

    Xtr_t = torch.from_numpy(X_train).float().to(device)
    Xte_t = torch.from_numpy(X_test).float().to(device)
    ytr_t = torch.from_numpy(y_train).long().to(device)
    yte_t = torch.from_numpy(y_test).long().to(device)
    return Xtr_t, ytr_t, Xte_t, yte_t, norm_stats


def _plot_examples(
    x_orig: torch.Tensor,
    x_cf: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    cf_preds: torch.Tensor,
    norm_stats: dict,
    path: Path,
    max_examples: int = 6,
) -> None:
    """Simple overlay plot of originals vs. counterfactuals."""
    max_examples = min(max_examples, x_orig.shape[0])
    if max_examples == 0:
        return

    min_val = norm_stats.get("min")
    range_val = norm_stats.get("range")
    x_orig_np = x_orig.detach().cpu().numpy()
    x_cf_np = x_cf.detach().cpu().numpy()

    if min_val is not None and range_val is not None:
        x_orig_np = x_orig_np * range_val + min_val
        x_cf_np = x_cf_np * range_val + min_val

    n_cols = min(3, max_examples)
    n_rows = int(np.ceil(max_examples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    axes = axes.ravel()

    for i in range(max_examples):
        ax = axes[i]
        series_orig = x_orig_np[i]
        series_cf = x_cf_np[i]
        n_channels = series_orig.shape[0]
        for ch in range(n_channels):
            ax.plot(series_orig[ch], label=f"orig ch{ch}", alpha=0.8)
            ax.plot(series_cf[ch], label=f"cf ch{ch}", alpha=0.8, linestyle="--")
        ax.set_title(
            f"pred={int(preds[i])}→tgt={int(targets[i])} (cf={int(cf_preds[i])})",
            fontsize=9,
        )
        ax.set_xlabel("Time")
        ax.legend(fontsize=7, frameon=False)

    for j in range(max_examples, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_dataset(dataset: str, args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    print(f"\n=== {dataset} (device={device}) ===")

    base = Path(args.output_dir) / f"{dataset}_optuna"
    ckpt_path = base / "checkpoints" / "best_classifier.pt"
    if not ckpt_path.exists():
        print(f"  ! Skipping {dataset}: checkpoint not found at {ckpt_path}")
        return

    Xtr_t, ytr_t, Xte_t, _, norm_stats = prepare_data(
        dataset=dataset,
        test_size=args.test_size,
        seed=args.seed,
        device=device,
    )
    norm_stats_path = base / "results" / "norm_stats.json"
    if norm_stats_path.exists():
        loaded_norm_stats = json.loads(norm_stats_path.read_text())
        norm_stats = {k: np.asarray(v) for k, v in loaded_norm_stats.items()}
    num_classes = int(torch.unique(ytr_t).numel())

    clf, _, _ = ClassifierOptimizer.load_checkpoint(ckpt_path, device=device)
    clf.eval()

    solver = CounterfactualSolver(clf, device=device)
    solver.compute_class_samples(Xtr_t, ytr_t, num_classes=num_classes)

    cfg = SolverConfig(
        steps=args.steps,
        lr=args.lr,
        lambda_proximity=args.lambda_proximity,
        lambda_sparsity=args.lambda_sparsity,
        lambda_validity=args.lambda_validity,
        k_neighbors=args.k_neighbors,
        dtw_gamma=args.dtw_gamma,
        dtw_normalize=not args.disable_dtw_normalize,
        p_target_min=args.p_target_min,
        valid_objective=args.valid_objective,
        weight_decay=args.weight_decay,
    )

    if args.n_eval_samples is None or args.n_eval_samples <= 0:
        n_eval = Xte_t.shape[0]
    else:
        n_eval = min(args.n_eval_samples, Xte_t.shape[0])
    idxs = torch.arange(n_eval, device=device)

    with torch.no_grad():
        logits = clf(Xte_t[idxs])
        preds = logits.argmax(dim=1)
    targets = (preds + 1) % num_classes

    neighbors = solver.precompute_neighbors(Xte_t[idxs], targets, cfg)
    result = solver.solve(
        Xte_t[idxs],
        y_target=targets,
        config=cfg,
        target_neighbors=neighbors,
    )

    x_cf = result["x_cf"].detach()
    logits_cf = result["logits"].detach()
    cf_preds = result["y_pred"].detach()
    probs_cf = torch.softmax(logits_cf, dim=1)
    target_probs = probs_cf.gather(1, targets.view(-1, 1)).squeeze(1)

    validity = float((target_probs >= cfg.p_target_min).float().mean().item())
    proximity_l2 = torch.norm(x_cf - Xte_t[idxs], dim=(1, 2)).mean().item()
    sparsity_l1 = torch.norm(x_cf - Xte_t[idxs], p=1, dim=(1, 2)).mean().item()
    plausibility = float(
        mean_soft_dtw_to_neighbors(
            x_cf.detach().cpu(),
            neighbors.detach().cpu(),
            gamma=cfg.dtw_gamma,
            normalize=cfg.dtw_normalize,
        ).mean().item()
    )

    metrics = {
        "dataset": dataset,
        "n_eval": int(n_eval),
        "lambda_validity": cfg.lambda_validity,
        "lambda_proximity": cfg.lambda_proximity,
        "lambda_sparsity": cfg.lambda_sparsity,
        "k_neighbors": cfg.k_neighbors,
        "dtw_gamma": cfg.dtw_gamma,
        "p_target_min": cfg.p_target_min,
        "validity": validity,
        "proximity_l2": proximity_l2,
        "sparsity_l1": sparsity_l1,
        "plausibility_dtw": plausibility,
        "outlier_factor_all": result.get("outlier_factor_all"),
        "outlier_factor_valid": result.get("outlier_factor_valid"),
    }

    # Persist results
    results_dir = base / "results"
    vis_dir = base / "visualizations"
    meta_dir = base / "metadata"
    results_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        results_dir / "counterfactuals.npz",
        x_orig=Xte_t[idxs].detach().cpu().numpy(),
        x_cf=x_cf.detach().cpu().numpy(),
        target_classes=targets.detach().cpu().numpy(),
        cf_probs=probs_cf.detach().cpu().numpy(),
        norm_min=norm_stats.get("min"),
        norm_range=norm_stats.get("range"),
    )

    _plot_examples(
        x_orig=Xte_t[idxs],
        x_cf=x_cf,
        preds=preds,
        targets=targets,
        cf_preds=cf_preds,
        norm_stats=norm_stats,
        path=vis_dir / "counterfactual_examples.png",
        max_examples=args.max_examples,
    )

    meta = {
        "solver_config": cfg.__dict__,
        "metrics": metrics,
    }
    (meta_dir / "evaluation_metadata.json").write_text(json.dumps(meta, indent=2))

    print(
        f"  validity={validity:.3f} prox_l2={proximity_l2:.4f} "
        f"sparsity_l1={sparsity_l1:.4f} dtw={plausibility:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CounterfactualSolver using Optuna-trained classifiers.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        help=f"Datasets to evaluate (default: {', '.join(ALL_DATASETS)})",
    )
    parser.add_argument("--steps", type=int, default=300, help="Solver steps.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Solver learning rate.")
    parser.add_argument("--lambda-validity", type=float, default=10.0, help="Validity weight.")
    parser.add_argument("--lambda-proximity", type=float, default=1.0, help="Proximity weight.")
    parser.add_argument("--lambda-sparsity", type=float, default=1.0, help="Sparsity weight.")
    parser.add_argument("--k-neighbors", type=int, default=10, help="Number of DTW neighbors.")
    parser.add_argument("--dtw-gamma", type=float, default=1.0, help="DTW gamma.")
    parser.add_argument("--p-target-min", type=float, default=0.55, help="Target probability threshold.")
    parser.add_argument(
        "--valid-objective",
        type=str,
        default="hinge",
        choices=["hinge", "ce"],
        help="Validity objective for the solver.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Solver weight decay.")
    parser.add_argument(
        "--n-eval-samples",
        type=int,
        default=None,
        help="Number of test samples to evaluate (None or <=0 uses the full test set).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="optuna_experimentsv2", help="Optuna output directory.")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--disable-dtw-normalize", action="store_true", help="Disable DTW normalization.")
    parser.add_argument("--max-examples", type=int, default=6, help="Max examples to plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    for ds in args.datasets:
        try:
            evaluate_dataset(ds, args)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {ds}: {exc}")


if __name__ == "__main__":
    main()
