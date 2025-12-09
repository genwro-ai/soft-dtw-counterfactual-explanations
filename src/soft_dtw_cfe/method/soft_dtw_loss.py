import torch

from soft_dtw_cfe.method.dtw import SoftDTW


def build_class_sample_bank(
    X: torch.Tensor, y: torch.Tensor, num_classes: int, device: str | None = None
) -> dict[int, torch.Tensor]:
    """
    Build a dictionary mapping each class to its training samples.

    Args:
        X: Training samples with shape (N, D, T) where N=samples, D=dimensions, T=time steps
        y: Class labels with shape (N,)
        num_classes: Number of classes
        device: Optional device to move tensors to

    Returns:
        Dictionary mapping class index to tensor of samples (N_c, D, T)
    """
    if device is not None:
        X = X.to(device)
        y = y.to(device)
    y = y.view(-1)
    bank: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for c in range(num_classes):
            mask = y == c
            if mask.any():
                bank[c] = X[mask].detach()
            else:
                bank[c] = torch.empty(0, *X.shape[1:], device=X.device)
    return bank


def select_knn_dtw_batch(
    x: torch.Tensor,
    bank: dict[int, torch.Tensor],
    y_target: torch.Tensor,
    k: int,
    gamma: float = 1.0,
    normalize: bool = True,
    metric: str = "dtw",
) -> torch.Tensor:
    """
    Find k nearest neighbors from target class for a batch of inputs.

    This function uses the codebase convention where time series are stored as (B, D, T)
    but internally transposes to (B, T, D) for DTW computation.

    Args:
        x: Input samples with shape (B, D, T) where B=batch, D=dimensions, T=time steps
        bank: Dictionary mapping class index to samples (N_c, D, T)
        y_target: Target classes for each sample (B,)
        k: Number of nearest neighbors to return
        gamma: Gamma parameter for soft DTW (ignored if metric="l2")
        normalize: Whether to normalize DTW distance (ignored if metric="l2")
        metric: Distance metric - "l2" (fast) or "dtw" (accurate for time series)

    Returns:
        k nearest neighbors from target_class with shape (B, k, D, T)
    """
    x = x if x.dim() == 3 else x.unsqueeze(0)
    B, D, T = x.shape
    device = x.device
    y_target = y_target.view(-1).long()

    # Prepare tensor to store results
    batch_neighbors = torch.zeros(B, k, D, T, device=device)

    unique_targets = torch.unique(y_target)

    with torch.no_grad():
        for target_cls_val in unique_targets:
            target_cls_item = target_cls_val.item()

            if target_cls_item not in bank:
                raise ValueError(f"Target class {target_cls_item} missing in bank")

            candidates = bank[target_cls_item]
            if candidates is None or candidates.numel() == 0:
                raise ValueError(f"No samples available for class {target_cls_item}")

            # Get mask for current class
            class_mask = y_target == target_cls_val
            x_class = x[class_mask]

            if metric == "l2":
                # Fast L2 distance computation
                x_flat = x_class.reshape(x_class.size(0), -1)
                candidates_flat = candidates.reshape(candidates.size(0), -1)
                distances = torch.cdist(x_flat, candidates_flat, p=2)

            elif metric == "dtw":
                # DTW distance computation - loop over candidates for memory efficiency
                # DTW expects (B, T, D) so we transpose
                dtw = SoftDTW(gamma=gamma, normalize=normalize)
                x_dtw = x_class.transpose(1, 2)  # (B_class, T, D)
                candidates_dtw = candidates.transpose(1, 2)  # (N_c, T, D)

                num_candidates = candidates.size(0)
                distances = torch.zeros(x_class.size(0), num_candidates, device=device)

                for i in range(num_candidates):
                    candidate = candidates_dtw[i : i + 1].expand(x_dtw.size(0), -1, -1)
                    dist = dtw(x_dtw, candidate)
                    distances[:, i] = dist

            else:
                raise ValueError(f"Unknown metric: {metric}. Use 'l2' or 'dtw'")

            # Get indices of k smallest distances
            k_eff = min(k, candidates.size(0))
            _, indices = torch.topk(distances, k_eff, largest=False, dim=1)

            # Gather neighbors (in original D, T layout)
            neighbors = candidates[indices]

            # Pad if we have fewer than k candidates
            if k_eff < k:
                pad = neighbors[:, -1:, :, :].expand(neighbors.size(0), k - k_eff, D, T)
                neighbors = torch.cat([neighbors, pad], dim=1)

            # Place neighbors in correct batch positions
            batch_neighbors[class_mask] = neighbors

    return batch_neighbors


def mean_soft_dtw_to_neighbors(
    x: torch.Tensor,
    neighbors: torch.Tensor,
    gamma: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute mean soft DTW distance from each sample to its neighbors.

    Args:
        x: Input samples with shape (B, D, T) where B=batch, D=dimensions, T=time steps
        neighbors: Neighbor samples with shape (B, K, D, T) where K=num neighbors
        gamma: Gamma parameter for soft DTW
        normalize: Whether to normalize DTW distance

    Returns:
        Mean distance across all samples and neighbors (scalar)
    """
    if neighbors.dim() != 4:
        raise ValueError("neighbors must be (B, K, D, T)")

    # Convert to (B, T, D) and (B, K, T, D) for DTW
    x_dtw = x.transpose(1, 2)  # (B, T, D)
    neighbors_dtw = neighbors.transpose(2, 3)  # (B, K, T, D)

    B, K, T, D = neighbors_dtw.shape
    dtw = SoftDTW(gamma=gamma, normalize=normalize)

    x_rep = x_dtw.unsqueeze(1).expand(B, K, T, D).reshape(B * K, T, D)
    y_rep = neighbors_dtw.reshape(B * K, T, D)
    dists = dtw(x_rep, y_rep).view(B, K)

    return dists.mean()
