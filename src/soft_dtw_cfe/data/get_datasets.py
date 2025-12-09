import numpy as np
from aeon.datasets import load_classification
from numpy.typing import NDArray


UNI_DATASETS = [
    "TwoLeadECG",
    "GunPoint",
    "Earthquakes",
    "Coffee",
    "ItalyPowerDemand",
    "CBF",
]

MULTI_DATASETS = [
    "Cricket",
    "Epilepsy",
]


def get_data(
    name: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    "Return the data as (N, D, T) numpy arrays."
    X_train, y_train = load_classification(name, split="train", extract_path="data")
    X_test, y_test = load_classification(name, split="test", extract_path="data")

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    y_unique = np.unique(y)
    y = np.searchsorted(y_unique, y)

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    indices = np.arange(n_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_indices = indices[n_test:]
    test_indices = indices[:n_test]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test
