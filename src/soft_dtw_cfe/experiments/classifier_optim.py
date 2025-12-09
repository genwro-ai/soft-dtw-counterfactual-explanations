import torch
import optuna
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Any

from soft_dtw_cfe.clf.clfs import CNN1DClassifier
from soft_dtw_cfe.clf.train import train_classifier


class ClassifierOptimizer:
    """Optimize CNN1DClassifier hyperparameters using Optuna."""

    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        in_channels: int,
        num_classes: int,
        device: str = "cpu",
    ):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.device = device

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for classifier optimization."""
        # Suggest hyperparameters
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        batch_norm = trial.suggest_categorical("batch_norm", [True, False])

        # Create model with suggested parameters
        clf = CNN1DClassifier(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            dropout=dropout,
        ).to(self.device)

        # Train classifier
        result = train_classifier(
            clf,
            self.train_loader,
            self.test_loader,
            epochs=50,  # Fixed for optimization
            lr=lr,
            weight_decay=weight_decay,
            device=self.device,
        )

        # Return best test accuracy
        return result["best_test_acc"]

    def optimize(
        self, n_trials: int = 50, study_name: str = "classifier_optim", storage: str | None = None
    ) -> optuna.Study:
        """Run Optuna optimization study."""
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        return study

    def train_best_model(
        self, best_params: dict[str, Any], epochs: int = 80
    ) -> tuple[CNN1DClassifier, dict]:
        """Train final model with best parameters."""
        clf = CNN1DClassifier(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            dropout=best_params["dropout"],
        ).to(self.device)

        result = train_classifier(
            clf,
            self.train_loader,
            self.test_loader,
            epochs=epochs,
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
            device=self.device,
        )

        return clf, result

    @staticmethod
    def save_checkpoint(
        clf: CNN1DClassifier,
        params: dict[str, Any],
        train_result: dict,
        path: str | Path,
    ) -> None:
        """Save classifier checkpoint with metadata."""
        # Get in_channels from the first conv layer in features
        first_conv = clf.features[0]  # First layer is Conv1d
        in_channels = first_conv.in_channels
        
        # Get num_classes from classifier layer
        num_classes = clf.classifier.out_features
        
        checkpoint = {
            "model_state_dict": clf.state_dict(),
            "params": params,
            "train_result": train_result,
            "model_config": {
                "in_channels": in_channels,
                "num_classes": num_classes,
                "dropout": params.get("dropout", 0.1),
            },
        }
        torch.save(checkpoint, path)
        print(f"Saved classifier checkpoint to {path}")

    @staticmethod
    def load_checkpoint(
        path: str | Path, device: str = "cpu"
    ) -> tuple[CNN1DClassifier, dict, dict]:
        """Load classifier from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        config = checkpoint["model_config"]
        clf = CNN1DClassifier(
            in_channels=config["in_channels"],
            num_classes=config["num_classes"],
            dropout=config["dropout"],
        ).to(device)

        clf.load_state_dict(checkpoint["model_state_dict"])
        clf.eval()

        print(f"Loaded classifier from {path}")
        return clf, checkpoint["params"], checkpoint["train_result"]
