import numpy as np
import torch
from numpy.typing import NDArray


def ensure_3d_ndt(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ensure time-series array has shape (N, D, T).

    - If input is (N, T), returns (N, 1, T) - adds channel dimension
    - If input is already (N, D, T), returns as-is
    """
    if x.ndim == 2:
        return x[:, None, :]  # (N, T) -> (N, 1, T)
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected 2D or 3D array, got shape {x.shape}")


def ensure_batch(x: torch.Tensor) -> torch.Tensor:
    return x if x.dim() >= 2 and x.size(0) > 0 else x.unsqueeze(0)


def z_normalize_train_test(
    X_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    eps: float = 1e-8,
    return_stats: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Z-normalize per-channel using train statistics.

    Expects input in (N, D, T) format.
    Computes mean and std over axes (N, T) for each channel D on the train set,
    applies normalization to both train and test. Returns normalized arrays with
    shape (N, D, T) and a stats dict containing mean and std.
    """
    if X_train.ndim == 2:
        # (N, T) -> (N, 1, T)
        X_train = X_train[:, None, :]
        X_test = X_test[:, None, :]

    # Compute stats over (N, T) axes for each channel D
    mean = X_train.mean(axis=(0, 2), keepdims=True)  # Shape: (1, D, 1)
    std = X_train.std(axis=(0, 2), keepdims=True)  # Shape: (1, D, 1)
    std = np.where(std < eps, 1.0, std)

    Xtr_z = (X_train - mean) / std
    Xte_z = (X_test - mean) / std
    stats = {"mean": mean.squeeze(), "std": std.squeeze()}

    if return_stats:
        return Xtr_z, Xte_z, stats
    return Xtr_z, Xte_z


def min_max_normalize_train_test(
    X_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    eps: float = 1e-8,
    return_stats: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Min-max normalize per-channel using train statistics.

    Expects input in (N, D, T) format.
    Computes min and max over axes (N, T) for each channel D on the train set,
    applies normalization to both train and test. Returns normalized arrays with
    shape (N, D, T) and a stats dict containing min, max, and range.
    """
    if X_train.ndim == 2:
        # (N, T) -> (N, 1, T)
        X_train = X_train[:, None, :]
        X_test = X_test[:, None, :]

    # Compute stats over (N, T) axes for each channel D
    min_val = X_train.min(axis=(0, 2), keepdims=True)  # Shape: (1, D, 1)
    max_val = X_train.max(axis=(0, 2), keepdims=True)  # Shape: (1, D, 1)
    range_val = max_val - min_val
    range_val = np.where(range_val < eps, 1.0, range_val)

    Xtr_mm = (X_train - min_val) / range_val
    Xte_mm = (X_test - min_val) / range_val
    stats = {
        "min": min_val.squeeze(),
        "max": max_val.squeeze(),
        "range": range_val.squeeze(),
    }

    if return_stats:
        return Xtr_mm, Xte_mm, stats
    return Xtr_mm, Xte_mm


def to_torch_tensor(
    x: np.ndarray | torch.Tensor,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return (
        x.to(device).to(dtype)
        if isinstance(x, torch.Tensor)
        else torch.from_numpy(x).to(device).to(dtype)
    )


def pad_time_series(
    X: NDArray[np.float64], multiple: int = 8
) -> tuple[NDArray[np.float64], int]:
    """Pad time series to a multiple along the time axis.

    Args:
        X: Array of shape (N, D, T)
        multiple: Pad time axis to nearest multiple of this value

    Returns:
        Tuple of (padded_array, original_length)
    """
    if X.ndim == 2:
        # (N, T) -> (N, 1, T)
        X = X[:, None, :]

    T = X.shape[2]
    pad_T = ((T + multiple - 1) // multiple) * multiple

    if pad_T == T:
        return X, T

    pad_amount = pad_T - T
    # Pad along time axis (axis=2) with edge values
    X_padded = np.pad(X, ((0, 0), (0, 0), (0, pad_amount)), mode="edge")
    return X_padded, T


def to_torch_ndt(X: np.ndarray, device: str) -> torch.Tensor:
    """Convert numpy array (N, D, T) to torch tensor on device.

    Args:
        X: Array of shape (N, D, T)
        device: Device to place tensor on

    Returns:
        Tensor of shape (N, D, T)
    """
    return torch.from_numpy(X).float().to(device)
