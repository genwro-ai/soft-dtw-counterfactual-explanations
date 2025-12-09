from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from soft_dtw_cfe.method.soft_dtw_loss import (
    build_class_sample_bank,
    select_knn_dtw_batch,
    mean_soft_dtw_to_neighbors,
)
from soft_dtw_cfe.data.utils import ensure_batch, to_torch_tensor
from sklearn.ensemble import IsolationForest
from tqdm import tqdm


@dataclass
class SolverConfig:
    steps: int = 500
    lr: float = 0.01
    lambda_proximity: float = 1.0
    lambda_sparsity: float = 1.0
    lambda_validity: float = 10.0
    k_neighbors: int = 5
    dtw_gamma: float = 1.0
    dtw_normalize: bool = True
    p_target_min: float = 0.5
    valid_objective: str = "hinge"
    weight_decay: float = 0.0
    verbose: bool = False  # Print progress every iteration
    print_every: int = 50  # Print progress every N steps (if verbose=True)


class CounterfactualSolver:
    def __init__(
        self,
        clf: nn.Module,
        device: str = "cpu",
    ) -> None:
        self.clf = clf.to(device)
        self.device = device

        for p in self.clf.parameters():
            p.requires_grad_(False)

        self.clf.eval()
        self.class_neighbors: Optional[Dict[int, torch.Tensor]] = None
        self.iforest_model: Optional[IsolationForest] = None

    def solve(
        self,
        x: torch.Tensor,
        y_target: torch.Tensor | int,
        config: SolverConfig | None = None,
        target_neighbors: Optional[torch.Tensor] = None,
    ) -> dict:
        if config is None:
            config = SolverConfig()

        x = x.to(self.device)
        x_b = ensure_batch(x)
        x_cf = x_b.clone().detach().to(self.device)
        x_cf.requires_grad_(True)

        if not torch.is_floating_point(x):
            x = x.float()
        if isinstance(y_target, int):
            y_target = torch.tensor([y_target], device=self.device)
        else:
            y_target = to_torch_tensor(y_target, device=self.device)

        y_b = ensure_batch(y_target)

        opt = torch.optim.Adam([x_cf], lr=config.lr, weight_decay=config.weight_decay)

        if target_neighbors is not None:
            neighbors = target_neighbors.detach().to(self.device)
            print(f"[Solver] Using precomputed neighbors: {neighbors.shape}")
        else:
            if self.class_neighbors is None:
                raise ValueError(
                    "No neighbors provided. Call compute_class_samples() first or pass target_neighbors to solve()."
                )
            print(
                f"[Solver] Computing DTW neighbors (k={config.k_neighbors}) - consider precomputing for speed!"
            )
            neighbors = select_knn_dtw_batch(
                x_b,
                self.class_neighbors,
                y_b,
                k=config.k_neighbors,
                gamma=config.dtw_gamma,
                normalize=config.dtw_normalize,
            )

        # Pre-compute y_idx to avoid repeated conversions
        y_idx = y_b.long().view(-1)
        
        # Track final loss values
        final_loss_prox = None
        final_loss_sparse = None
        final_loss_valid = None
        final_loss_plaus = None
        
        for it in tqdm(range(config.steps), desc="Solving counterfactual"):
            opt.zero_grad(set_to_none=True)

            logits = self.clf(x_cf)

            # Validity
            probs = F.softmax(logits, dim=1).gather(1, y_idx.view(-1, 1)).squeeze(1)
            if config.valid_objective == "ce":
                loss_valid = F.cross_entropy(logits, y_idx)
            else:
                loss_valid = F.relu(config.p_target_min - probs).mean()

            diff = x_cf - x_b
            loss_prox = (diff**2).mean()
            loss_sparse = diff.abs().mean()

            # Plausibility: mean Soft-DTW to K target-class neighbors
            loss_plaus = mean_soft_dtw_to_neighbors(
                x_cf, neighbors, gamma=config.dtw_gamma, normalize=config.dtw_normalize
            )

            loss = (
                config.lambda_proximity * loss_prox
                + config.lambda_sparsity * loss_sparse
                + config.lambda_validity * (loss_plaus + loss_valid)
            )

            loss.backward()

            # Print progress (optional, disabled by default for speed)
            if config.verbose and (it % config.print_every == 0 or it == config.steps - 1):
                log_str = (
                    f"it={it:3d} | "
                    f"loss_prox: {loss_prox.item():.4f}, "
                    f"loss_sparse: {loss_sparse.item():.4f}, "
                    f"loss_valid: {loss_valid.item():.4f}, "
                    f"loss_plaus: {loss_plaus.item():.4f}"
                )
                print(log_str)

            opt.step()
            
            # Save final iteration losses
            if it == config.steps - 1:
                final_loss_prox = loss_prox.item()
                final_loss_sparse = loss_sparse.item()
                final_loss_valid = loss_valid.item()
                final_loss_plaus = loss_plaus.item()

        with torch.no_grad():
            logits = self.clf(x_cf)
            probs_final = F.softmax(logits, dim=1)
            y_pred = logits.argmax(dim=1)
            target_probs_final = probs_final.gather(1, y_idx.view(-1, 1)).squeeze(1)

        valid_mask = target_probs_final >= config.p_target_min
        isolation_scores = self.compute_isolation_forest_scores(x_cf, valid_mask=valid_mask)

        result = {
            "x_cf": x_cf.detach(),
            "logits": logits.detach(),
            "y_pred": y_pred.detach(),
            "loss_proximity": final_loss_prox,
            "loss_sparsity": final_loss_sparse,
            "loss_validity": final_loss_valid,
            "loss_plausibility": final_loss_plaus,
        }
        
        if isolation_scores:
            result.update(isolation_scores)
        
        return result

    def compute_class_samples(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        num_classes: int,
    ) -> Dict[int, torch.Tensor]:
        """
        Precompute and store all training samples by class for k-NN lookup.

        Args:
            X: Training data (N, D, T) where N=samples, D=dimensions, T=time steps
            y: Training labels (N,)
            num_classes: Number of classes

        Returns:
            Dictionary mapping class_idx -> Tensor of all samples from that class
        """
        X = X.to(self.device)
        y = y.to(self.device)
        bank = build_class_sample_bank(X, y, num_classes, device=self.device)
        self.class_neighbors = bank
        self._fit_isolation_forest(X)
        return bank

    def precompute_neighbors(
        self,
        X: torch.Tensor,
        y_target: torch.Tensor,
        config: SolverConfig | None = None,
    ) -> torch.Tensor:
        """
        Precompute k-NN neighbors for a batch of samples.

        This is a performance optimization: instead of computing neighbors
        in every optimization iteration (expensive!), compute them once upfront.

        Args:
            X: Input samples (B, D, T) where B=batch, D=dimensions, T=time steps
            y_target: Target classes for each sample (B,)
            config: Solver config (uses k_neighbors, dtw_gamma, dtw_normalize)

        Returns:
            Precomputed neighbors (B, K, D, T) where K=number of neighbors

        Example:
            >>> # Precompute once
            >>> neighbors = solver.precompute_neighbors(x_batch, target_batch, config)
            >>> # Use in solve (avoids recomputation)
            >>> result = solver.solve(x_batch, target_batch, config, target_neighbors=neighbors)
        """
        if config is None:
            config = SolverConfig()

        if self.class_neighbors is None:
            raise ValueError("Must call compute_class_samples() first")

        X = X.to(self.device)
        y_target = to_torch_tensor(y_target, device=self.device)

        neighbors = select_knn_dtw_batch(
            X,
            self.class_neighbors,
            y_target,
            k=config.k_neighbors,
            gamma=config.dtw_gamma,
            normalize=config.dtw_normalize,
        )

        return neighbors

    def _fit_isolation_forest(self, X: torch.Tensor) -> None:
        """Fit the Isolation Forest model on flattened training samples."""
        X_cpu = torch.nan_to_num(X.detach().float().cpu())
        n_samples = X_cpu.size(0)
        if n_samples == 0:
            self.iforest_model = None
            return

        features = X_cpu.reshape(n_samples, -1).numpy()
        self.iforest_model = IsolationForest(contamination=0.01, random_state=39)
        self.iforest_model.fit(features)

    def compute_isolation_forest_scores(
        self,
        samples: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, float | None]:
        """Compute Isolation Forest inlier rates for the provided samples."""
        if self.iforest_model is None:
            return {}

        samples_cpu = torch.nan_to_num(samples.detach().float().cpu())
        n_samples = samples_cpu.size(0)
        if n_samples == 0:
            return {}

        features = samples_cpu.reshape(n_samples, -1).numpy()
        preds = self.iforest_model.predict(features)
        scores: dict[str, float | None] = {
            "outlier_factor_all": float((preds == 1).mean()),
            "outlier_factor_valid": None,
        }

        if valid_mask is not None:
            mask = valid_mask.detach().cpu().flatten()
            if mask.numel() == n_samples and bool(mask.any()):
                mask_np = mask.numpy().astype(bool)
                preds_valid = preds[mask_np]
                scores["outlier_factor_valid"] = float((preds_valid == 1).mean())

        return scores
