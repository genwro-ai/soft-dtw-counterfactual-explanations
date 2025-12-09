import argparse
import json
import sys
from pathlib import Path

# Ensure local package imports (nte.*) work when running from repo root
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest

from soft_dtw_cfe.reference.m_cels.nte.experiment.default_args0 import parse_arguments  # noqa: E402
from soft_dtw_cfe.reference.m_cels.nte.experiment.utils import backgroud_data_configuration  # noqa: E402
from soft_dtw_cfe.reference.m_cels.nte.models.saliency_model.counterfactual_multi_nomal import (  # noqa: E402
    CFExplainer,
)
from soft_dtw_cfe.data.get_datasets import get_data, MULTI_DATASETS, UNI_DATASETS
from soft_dtw_cfe.data.utils import min_max_normalize_train_test
from soft_dtw_cfe.method.soft_dtw_loss import (
    build_class_sample_bank,
    mean_soft_dtw_to_neighbors,
)
from soft_dtw_cfe.clf.clfs import CNN1DClassifier
from soft_dtw_cfe.experiments.classifier_optim import ClassifierOptimizer


def load_classifier(dataset: str, optuna_base_dir: str, device: str) -> CNN1DClassifier | None:
    """Load a saved PyTorch classifier checkpoint if present."""
    ckpt = Path(optuna_base_dir) / f"{dataset}_optuna" / "checkpoints" / "best_classifier.pt"
    if not ckpt.exists():
        return None
    clf, _, _ = ClassifierOptimizer.load_checkpoint(ckpt, device=device)
    return clf


def compute_metrics(
    x_orig: torch.Tensor,
    x_cf: torch.Tensor,
    target_classes: np.ndarray,
    neighbors: torch.Tensor,
    cf_probs: np.ndarray,
    tau: float,
    dtw_gamma: float = 0.1,
    iforest: IsolationForest | None = None,
) -> dict[str, float]:
    """Match Glacier metric set for fair comparison."""
    diff = x_cf - x_orig
    proximity = float(torch.norm(diff, dim=(1, 2)).mean().item())
    sparsity = float(torch.norm(diff, p=1, dim=(1, 2)).mean().item())
    plausibility = float(
        mean_soft_dtw_to_neighbors(x_cf, neighbors, gamma=dtw_gamma, normalize=True).mean().item()
    )

    cf_probs_t = torch.from_numpy(cf_probs).float().to(x_cf.device)
    targets_t = torch.from_numpy(target_classes).long().to(x_cf.device)
    target_prob = cf_probs_t.gather(1, targets_t.view(-1, 1)).squeeze(1)
    validity_mask = target_prob >= tau
    validity = float(validity_mask.float().mean().item())
    validity_loss = float(F.relu(tau - target_prob).mean().item())

    outlier_all = None
    outlier_valid = None
    if iforest is not None:
        flat = torch.nan_to_num(x_cf.detach().cpu()).reshape(x_cf.size(0), -1).numpy()
        preds_all = iforest.predict(flat)
        outlier_all = float((preds_all == 1).mean())
        if validity_mask.any():
            preds_valid = preds_all[validity_mask.cpu().numpy().astype(bool)]
            if preds_valid.size > 0:
                outlier_valid = float((preds_valid == 1).mean())

    return {
        "validity": validity,
        "validity_loss": validity_loss,
        "proximity_l2": proximity,
        "sparsity_l1": sparsity,
        "plausibility_dtw": plausibility,
        "outlier_factor_all": outlier_all,
        "outlier_factor_valid": outlier_valid,
    }


def run_mcels_on_dataset(
    dataset: str,
    args: argparse.Namespace,
    optuna_base_dir: str,
    device: str,
    n_eval_samples: int | None,
    save_dir: Path | None,
) -> dict[str, float] | None:
    """Execute M-CELS on one dataset; returns metrics or None if unavailable."""
    print(f"\n=== Dataset: {dataset} ===")
    Xtr, ytr, Xte, yte = get_data(dataset, test_size=args.test_size, seed=args.seed)
    Xtr_norm, Xte_norm, norm_stats = min_max_normalize_train_test(
        Xtr, Xte, return_stats=True
    )

    clf = load_classifier(dataset, optuna_base_dir=optuna_base_dir, device=device)
    if clf is None:
        print(f"Skipping {dataset}: classifier checkpoint not found.")
        return None
    clf.eval()

    # Background data selection mimics authors' defaults (test + 100%)
    bg_data, bg_label, bg_len = backgroud_data_configuration(
        BACKGROUND_DATA=args.background_data,
        BACKGROUND_DATA_PERC=args.background_data_perc,
        dataset=type("Obj", (), {"test_data": Xte_norm, "test_label": yte, "train_data": Xtr_norm, "train_label": ytr}),
    )
    bg_data = torch.from_numpy(bg_data[:bg_len]).float().to(device)
    bg_label = torch.from_numpy(bg_label[:bg_len]).long().to(device)

    # Build neighbor bank for plausibility
    num_classes = int(len(np.unique(ytr)))
    class_bank = build_class_sample_bank(
        torch.from_numpy(Xtr_norm).float(), torch.from_numpy(ytr), num_classes=num_classes
    )

    explainer = CFExplainer(
        background_data=bg_data.cpu().numpy(),
        background_label=bg_label.cpu().numpy(),
        predict_fn=clf,
        enable_wandb=False,
        args=args,
        use_cuda=device.startswith("cuda"),
    )

    # IsolationForest on train for outlier scoring
    iforest = IsolationForest(random_state=args.seed)
    flat_train = torch.from_numpy(Xtr_norm).reshape(len(Xtr_norm), -1).numpy()
    iforest.fit(flat_train)

    indices = np.arange(len(Xte_norm))
    if n_eval_samples is not None:
        if n_eval_samples < 0:
            pass  # negative means all
        else:
            indices = indices[:n_eval_samples]

    cf_preds = []
    cf_probs = []
    x_cfs = []
    target_classes = []
    processed_indices: list[int] = []

    softmax_fn = torch.nn.Softmax(dim=-1)

    for idx in indices:
        x = torch.from_numpy(Xte_norm[idx]).float().to(device)
        y = int(yte[idx])
        with torch.no_grad():
            orig_logits = clf(x.unsqueeze(0))
            orig_probs = softmax_fn(orig_logits)[0]
        orig_pred = int(torch.argmax(orig_probs).item())
        cf_label = int((orig_pred + 1) % num_classes)

        # Skip if no background samples for target class (avoids KNN crash)
        if (bg_label.cpu().numpy() == cf_label).sum() == 0:
            print(f"Skipping idx {idx}: no background samples for target class {cf_label}")
            continue

        mask, cf_ts, target_prob = explainer.generate_saliency(
            data=x.cpu().numpy(), label=y, save_dir="/tmp", target=orig_probs.unsqueeze(0), dataset=None
        )
        cf_tensor = torch.from_numpy(cf_ts).float().to(device)
        # Ensure shape (D, T)
        if cf_tensor.dim() == 1:
            cf_tensor = cf_tensor.view(1, -1)
        elif cf_tensor.dim() == 3 and cf_tensor.size(0) == 1:
            cf_tensor = cf_tensor.squeeze(0)
        with torch.no_grad():
            cf_logits = clf(cf_tensor.unsqueeze(0))
            cf_prob_vec = softmax_fn(cf_logits)[0]
            cf_pred = int(torch.argmax(cf_prob_vec).item())

        cf_preds.append(cf_pred)
        cf_probs.append(cf_prob_vec.cpu().numpy())
        x_cfs.append(cf_tensor.cpu())
        target_classes.append(cf_label)
        processed_indices.append(idx)

    if not x_cfs:
        print(f"No counterfactuals generated for {dataset} (skipping).")
        return None
    proc_indices_np = np.array(processed_indices)
    x_orig_batch = torch.from_numpy(Xte_norm[proc_indices_np]).float()
    x_cf_batch = torch.stack(x_cfs, dim=0)
    target_classes_np = np.array(target_classes)
    cf_probs_np = np.stack(cf_probs, axis=0)

    # Neighbor selection for targets
    neighbors = torch.stack(
        [class_bank[cls][: args.k_neighbors] for cls in target_classes_np],
        dim=0,
    )

    metrics = compute_metrics(
        x_orig=x_orig_batch,
        x_cf=x_cf_batch,
        target_classes=target_classes_np,
        neighbors=neighbors,
        cf_probs=cf_probs_np,
        tau=args.tau_value,
        dtw_gamma=args.dtw_gamma,
        iforest=iforest,
    )

    if save_dir:
        ds_dir = save_dir / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ds_dir / "counterfactuals.npz",
            x_orig=x_orig_batch.numpy(),
            x_cf=x_cf_batch.numpy(),
            y_true=yte[indices],
            cf_pred=np.array(cf_preds),
            target_classes=target_classes_np,
            cf_probs=cf_probs_np,
            norm_min=norm_stats["min"],
            norm_range=norm_stats["range"],
        )
        with open(ds_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved counterfactuals to {ds_dir}")

    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run M-CELS on aeon datasets")
    parser.add_argument(
        "--optuna-dir",
        type=str,
        default="optuna_experimentsv2",
        help="Base directory containing *_optuna checkpoints.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=UNI_DATASETS + MULTI_DATASETS,
        help="Datasets to run (defaults to Glacier set).",
    )
    parser.add_argument(
        "--n-eval-samples",
        type=int,
        default=None,
        help="Number of test samples per dataset (default: all test samples).",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--k-neighbors", type=int, default=5, help="Neighbors for DTW plausibility.")
    parser.add_argument("--tau-value", type=float, default=0.5, help="Target probability threshold.")
    parser.add_argument("--dtw-gamma", type=float, default=0.1, help="Gamma for soft-DTW.")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="mcels_outputs",
        help="Directory to save counterfactuals and metrics (empty to skip saving).",
    )
    args_cli = parser.parse_args()

    device = args_cli.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Pull default M-CELS args without consuming this script's CLI args
    _argv_backup = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        mcels_args, _unknown = parse_arguments(standalone=True)
    finally:
        sys.argv = _argv_backup
    mcels_args.background_data = "train"
    mcels_args.background_data_perc = 100
    mcels_args.max_itr = 1000
    mcels_args.enable_wandb = False
    mcels_args.k_neighbors = args_cli.k_neighbors
    mcels_args.tau_value = args_cli.tau_value
    mcels_args.dtw_gamma = args_cli.dtw_gamma
    mcels_args.test_size = args_cli.test_size
    mcels_args.seed = args_cli.seed

    all_metrics: dict[str, dict[str, float]] = {}
    save_base = Path(args_cli.save_dir) if args_cli.save_dir else None
    for ds in args_cli.datasets:
        metrics = run_mcels_on_dataset(
            dataset=ds,
            args=mcels_args,
            optuna_base_dir=args_cli.optuna_base_dir,
            device=device,
            n_eval_samples=args_cli.n_eval_samples,
            save_dir=save_base,
        )
        if metrics:
            all_metrics[ds] = metrics

    if all_metrics:
        print("\n=== Summary ===")
        for ds, mets in all_metrics.items():
            print(f"{ds}: validity={mets['validity']:.3f}, prox={mets['proximity_l2']:.4f}, sparsity={mets['sparsity_l1']:.4f}")


if __name__ == "__main__":
    main()
