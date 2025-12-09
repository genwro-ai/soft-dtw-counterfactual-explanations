"""
Run the core CounterfactualSolver across multiple lambda_validity and k_neighbors values for
selected datasets, collecting standard metrics (validity, proximity L2,
sparsity L1, DTW plausibility, IsolationForest inlier rates).

Usage:
    python scripts/run_core_lambda_sweep.py \
        --datasets CBF TwoLeadECG GunPoint \
        --lambdas 1 2 5 \
        --k-neighbors-list 5 10 20 \
        --n-samples 50 \
        --optuna-dir optuna_experimentsv2 \
        --output-dir hyperparams_search
"""
import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from soft_dtw_cfe.experiments.classifier_optim import ClassifierOptimizer
from soft_dtw_cfe.data.get_datasets import get_data
from soft_dtw_cfe.data.utils import min_max_normalize_train_test
from soft_dtw_cfe.method.soft_dtw_loss import mean_soft_dtw_to_neighbors
from soft_dtw_cfe.method.solver import CounterfactualSolver, SolverConfig


def _to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(arr).float().to(device)


def run_hyperparams_search_for_dataset(
    dataset: str,
    lambdas: Iterable[float],
    k_neighbors_list: Iterable[int],
    n_samples: int | None,
    args: argparse.Namespace,
) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    print(f"\n=== {dataset} === (device={device})")

    # Load and normalize data
    X_train, y_train, X_test, _ = get_data(dataset, test_size=args.test_size, seed=args.seed)
    X_train, X_test, norm_stats = min_max_normalize_train_test(X_train, X_test, return_stats=True)
    X_train_t = _to_tensor(X_train, device)
    X_test_t = _to_tensor(X_test, device)
    y_train_t = torch.from_numpy(y_train).long().to(device)

    print(X_train.shape, X_test.shape)

    # Load classifier
    ckpt = Path(args.optuna_dir) / f"{dataset}_optuna" / "checkpoints" / "best_classifier.pt"
    if not ckpt.exists():
        print(f"  Skipping {dataset}: classifier checkpoint not found at {ckpt}")
        return
    clf, _, _ = ClassifierOptimizer.load_checkpoint(ckpt, device=device)
    clf.eval()

    # Prepare solver
    solver = CounterfactualSolver(clf, device=device)
    num_classes = int(np.unique(y_train).shape[0])
    solver.compute_class_samples(X_train_t, y_train_t, num_classes=num_classes)

    # Evaluation indices
    n_eval = X_test.shape[0] if n_samples is None else min(n_samples, X_test.shape[0])
    idxs = np.arange(n_eval)

    # Base targets (next class)
    with torch.no_grad():
        logits = clf(X_test_t[idxs])
        preds = logits.argmax(dim=1)
    targets = (preds + 1) % num_classes

    out_base = Path(args.output_dir) / dataset
    out_base.mkdir(parents=True, exist_ok=True)

    # Sweep over k_neighbors first to precompute neighbors efficiently
    for k_neighbors in k_neighbors_list:
        print(f"  k_neighbors={k_neighbors}")

        # Precompute neighbors once per k_neighbors value
        cfg_temp = SolverConfig(
            steps=args.steps,
            lr=args.lr,
            lambda_proximity=1.0,
            lambda_sparsity=1.0,
            lambda_validity=1.0,
            k_neighbors=k_neighbors,
            dtw_gamma=args.dtw_gamma,
            dtw_normalize=True,
            p_target_min=args.p_target_min,
            valid_objective="hinge",
            weight_decay=0.0,
        )
        neighbors = solver.precompute_neighbors(X_test_t[idxs], targets, cfg_temp)

        for lam in lambdas:
            print(f"    lambda_validity={lam}")
            cfg = SolverConfig(
                steps=args.steps,
                lr=args.lr,
                lambda_proximity=1.0,
                lambda_sparsity=1.0,
                lambda_validity=float(lam),
                k_neighbors=k_neighbors,
                dtw_gamma=args.dtw_gamma,
                dtw_normalize=True,
                p_target_min=args.p_target_min,
                valid_objective="hinge",
                weight_decay=0.0,
            )

            result = solver.solve(
                X_test_t[idxs],
                y_target=targets,
                config=cfg,
                target_neighbors=neighbors,
            )

            x_cf = result["x_cf"].detach().cpu()
            x_orig = X_test_t[idxs].detach().cpu()
            logits_cf = result["logits"].detach().cpu()
            probs_cf = torch.softmax(logits_cf, dim=1)
            target_probs = probs_cf.gather(1, targets.view(-1, 1).cpu()).squeeze(1)
            validity = float((target_probs >= cfg.p_target_min).float().mean().item())
            proximity_l2 = torch.norm(x_cf - x_orig, dim=(1, 2)).mean().item()
            sparsity_l1 = torch.norm(x_cf - x_orig, p=1, dim=(1, 2)).mean().item()
            plausibility = float(
                mean_soft_dtw_to_neighbors(
                    x_cf, neighbors.cpu(), gamma=cfg.dtw_gamma, normalize=cfg.dtw_normalize
                ).mean().item()
            )
            outlier_all = result.get("outlier_factor_all")
            outlier_valid = result.get("outlier_factor_valid")

            metrics = {
                "dataset": dataset,
                "lambda_validity": lam,
                "k_neighbors": k_neighbors,
                "n": int(n_eval),
                "validity": validity,
                "proximity_l2": proximity_l2,
                "sparsity_l1": sparsity_l1,
                "plausibility_dtw": plausibility,
                "outlier_factor_all": outlier_all,
                "outlier_factor_valid": outlier_valid,
            }

            out_dir = out_base / f"k_{k_neighbors}" / f"lambda_{lam}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "metrics.json").open("w") as f:
                json.dump(metrics, f, indent=2)

            np.savez_compressed(
                out_dir / "counterfactuals.npz",
                x_orig=x_orig.numpy(),
                x_cf=x_cf.numpy(),
                target_classes=targets.cpu().numpy(),
                cf_probs=probs_cf.numpy(),
                norm_min=norm_stats["min"],
                norm_range=norm_stats["range"],
            )
            print(
                f"      validity={validity:.3f} prox_l2={proximity_l2:.4f} "
                f"sparsity_l1={sparsity_l1:.4f} dtw={plausibility:.4f} outlier_all={outlier_all}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lambda and k-neighbors sweep for core CounterfactualSolver.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["CBF", "TwoLeadECG", "GunPoint"],
        help="Datasets to run.",
    )
    parser.add_argument(
        "--lambdas",
        nargs="*",
        type=float,
        default=[1.0, 2.0, 5.0],
        help="lambda_validity values to sweep.",
    )
    parser.add_argument(
        "--k-neighbors-list",
        nargs="*",
        type=int,
        default=[5, 10, 20],
        help="k_neighbors values to sweep for DTW.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of test samples per dataset (default: 50, use None for all).",
    )
    parser.add_argument("--steps", type=int, default=300, help="Solver steps.")
    parser.add_argument("--lr", type=float, default=0.001, help="Solver learning rate.")
    parser.add_argument("--dtw-gamma", type=float, default=1, help="DTW gamma.")
    parser.add_argument("--p-target-min", type=float, default=0.55, help="Target prob threshold.")
    parser.add_argument("--optuna-dir", type=str, default="optuna_experimentsv2", help="Optuna dir.")
    parser.add_argument("--output-dir", type=str, default="hyperparams_search", help="Output directory.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    for ds in args.datasets:
        try:
            run_hyperparams_search_for_dataset(ds, args.lambdas, args.k_neighbors_list, args.n_samples, args)
        except Exception as e:
            print(f"Error on {ds}: {e}")
