#!/usr/bin/env python
# coding: utf-8
import os
import sys
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import IsolationForest

# Ensure local module imports work when running from repository root
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from soft_dtw_cfe.reference.glacier.src.help_functions import (
    ResultWriter,
    conditional_pad,
    remove_paddings,
    evaluate,
    fit_evaluation_models,
    reset_seeds,
    time_series_normalize,
    find_best_lr,
)
from soft_dtw_cfe.reference.glacier.src.keras_models import Autoencoder, LSTMFCNClassifier
# from latent_flow_ts.reference.glacier.src._guided import get_global_weights

from soft_dtw_cfe.data.get_datasets import get_data
from soft_dtw_cfe.data.utils import min_max_normalize_train_test
from soft_dtw_cfe.method.solver import SolverConfig
from soft_dtw_cfe.method.soft_dtw_loss import (
    mean_soft_dtw_to_neighbors,
    build_class_sample_bank,
    select_knn_dtw_batch,
)
from soft_dtw_cfe.clf.clfs import CNN1DClassifier
from soft_dtw_cfe.experiments.classifier_optim import ClassifierOptimizer

# Default dataset lists (aeon, univariate vs multivariate)
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


os.environ["TF_DETERMINISTIC_OPS"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


class PyTorchClassifierWrapper:
    """
    Wrapper to make PyTorch classifier compatible with Keras-based code.
    
    The wrapper converts between TensorFlow/Keras formats (N, T, D) and PyTorch formats (N, D, T),
    and provides a predict() method that returns probabilities like Keras.
    """
    def __init__(self, pytorch_model: torch.nn.Module, device: str = "cpu"):
        self.model = pytorch_model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def predict(self, x):
        """
        Predict probabilities for input samples (Keras-like interface).
        
        Args:
            x: Input samples in Keras format (N, T, D) - numpy array or TensorFlow tensor
            
        Returns:
            Predicted probabilities (N, num_classes) - numpy array
        """
        # Convert TensorFlow tensor to numpy if needed
        if hasattr(x, 'numpy'):  # TensorFlow tensor
            x_np = x.numpy()
        elif isinstance(x, np.ndarray):
            x_np = x
        else:
            # Try to convert to numpy
            x_np = np.array(x)
        
        # Convert from Keras (N, T, D) to PyTorch (N, D, T) format
        x_transposed = x_np.transpose(0, 2, 1)
        
        # Convert to torch tensor
        x_torch = torch.from_numpy(x_transposed).float().to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(x_torch)
            probs = F.softmax(logits, dim=1)
        
        # Convert back to numpy
        return probs.cpu().numpy()
    
    def __call__(self, x):
        """Allow the wrapper to be called like a Keras model."""
        return self.predict(x)


def load_saved_classifier(dataset: str, optuna_base_dir: str = "optuna_experiments", device: str = "cpu") -> Optional[PyTorchClassifierWrapper]:
    """
    Load a saved PyTorch classifier from optuna experiments.
    
    Args:
        dataset: Dataset name (e.g., 'TwoLeadECG', 'GunPoint', etc.)
        optuna_base_dir: Base directory containing optuna experiment folders
        device: Device to load model on ('cpu' or 'cuda')
        
    Returns:
        PyTorchClassifierWrapper if checkpoint exists, None otherwise
    """
    checkpoint_path = Path(optuna_base_dir) / f"{dataset}_optuna" / "checkpoints" / "best_classifier.pt"
    
    if not checkpoint_path.exists():
        return None
    
    try:
        # Load the classifier using ClassifierOptimizer
        clf, params, train_result = ClassifierOptimizer.load_checkpoint(checkpoint_path, device=device)
        
        # Get test accuracy from train_result
        test_acc = train_result.get('best_test_acc', train_result.get('best_val_acc', 0.0))
        
        logging.info(f"Loaded saved classifier from {checkpoint_path}")
        logging.info(f"  Test accuracy: {test_acc:.4f}")
        logging.info(f"  Parameters: {params}")
        
        # Wrap the PyTorch model for compatibility with Keras-based code
        return PyTorchClassifierWrapper(clf, device=device)
        
    except Exception as e:
        logging.warning(f"Failed to load classifier from {checkpoint_path}: {e}")
        return None


def compute_solver_metrics(
    x_orig,
    x_cf,
    neighbors,
    cf_pred_probs,
    target_classes,
    tau_value,
    dtw_gamma=1.0,
    dtw_normalize=True,
    iforest_model: IsolationForest | None = None,
):
    """
    Compute proximity, sparsity, plausibility, and outlier metrics for counterfactuals.
    
    Args:
        x_orig: Original samples (B, D, T) torch tensor
        x_cf: Counterfactual samples (B, D, T) torch tensor
        neighbors: Precomputed neighbors (B, K, D, T) torch tensor
        cf_pred_probs: Counterfactual prediction probabilities (B, C) numpy array
        target_classes: Target classes for each sample (B,) numpy array
        tau_value: Target probability threshold (float)
        dtw_gamma: Gamma parameter for soft-DTW
        dtw_normalize: Whether to normalize DTW distances
        iforest_model: Pre-fitted IsolationForest instance (optional)
    
    Returns:
        dict with 'proximity', 'sparsity', 'plausibility', 'validity',
        'outlier_factor_all', and 'outlier_factor_valid' metrics
    """
    diff = x_cf - x_orig
    prox = (diff**2).mean().item()
    sparse = diff.abs().mean().item()
    plaus = mean_soft_dtw_to_neighbors(
        x_cf, neighbors, gamma=dtw_gamma, normalize=dtw_normalize
    ).item()

    # Validity loss (hinge loss on target probability)
    cf_pred_probs_t = torch.from_numpy(cf_pred_probs).float().to(x_cf.device)
    target_classes_t = torch.from_numpy(target_classes).long().to(x_cf.device)
    probs = cf_pred_probs_t.gather(1, target_classes_t.view(-1, 1)).squeeze(1)
    validity = F.relu(tau_value - probs).mean().item()

    outlier_factor_all = None
    outlier_factor_valid = None
    if iforest_model is not None:
        x_cf_cpu = torch.nan_to_num(x_cf.detach().float().cpu())
        flattened = x_cf_cpu.reshape(x_cf_cpu.size(0), -1).numpy()
        preds_all = iforest_model.predict(flattened)
        outlier_factor_all = float((preds_all == 1).mean())

        valid_mask = (probs >= tau_value).detach().cpu().flatten()
        if valid_mask.numel() == x_cf_cpu.size(0) and bool(valid_mask.any()):
            mask_np = valid_mask.numpy().astype(bool)
            preds_valid = preds_all[mask_np]
            if preds_valid.size > 0:
                outlier_factor_valid = float((preds_valid == 1).mean())


    return {
        'proximity': prox,
        'sparsity': sparse,
        'plausibility': plaus,
        'validity': validity,
        'outlier_factor_all': outlier_factor_all,
        'outlier_factor_valid': outlier_factor_valid,
    }


def main():
    parser = ArgumentParser(description="Train LatentCF with aeon datasets.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="glacier_outputs",
        help="Output directory to save results CSVs (default: glacier_outputs)",
    )
    parser.add_argument("--w-type", type=str, default="uniform", help="Weighting: local | global | uniform | unconstrained")
    parser.add_argument("--w-value", type=float, default=0.5, help="Prediction margin weight in [0,1]")
    parser.add_argument("--tau-value", type=float, default=0.5, help="Target probability threshold [0.5,1]")
    parser.add_argument("--lr-list", nargs="+", type=float, default=[0.001, 0.0001], help="Learning rates to try for CF search (default: 0.001 0.0001)")
    parser.add_argument("--optuna-dir", type=str, default="optuna_experiments", help="Base directory containing optuna experiment folders with saved classifiers")
    parser.add_argument("--use-saved-classifier", action="store_true", help="Use saved classifier from optuna experiments instead of training new one")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=UNI_DATASETS,
        choices=UNI_DATASETS,
        help="Datasets to run (default: all univariate aeon datasets).",
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}.")
    
    datasets_to_run = args.datasets
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets_to_run:
        logger.info(f"\n{'='*80}\nRunning dataset: {dataset}\n{'='*80}")
        
        output_csv = output_dir / f"{dataset}_latentcf_results.csv"
        pred_margin_weight = float(args.w_value)
        tau_value = float(args.tau_value)

        result_writer = ResultWriter(file_name=output_csv, dataset_name=dataset)
        if not os.path.isfile(output_csv):
            result_writer.write_head()

        # 1) Load data via aeon helper - use same seed as optuna experiments
        X_train, y_train, X_test, y_test = get_data(dataset, test_size=0.2, seed=42)
        norm_min = None
        norm_range = None

        # Ensure 3D shape and orientation (N, D, T) first for consistency
        if X_train.ndim == 3:
            # Aeon returns (N, D, T) for multichannel
            pass  # Keep as is for now
        else:
            raise ValueError("Expected 3D data from aeon get_data.")
        
        Ntr_orig, D, T = X_train.shape
        K = len(np.unique(y_train))
        logger.info(f"Dataset shape: N={Ntr_orig}, D={D}, T={T}, K={K}")

        # Try to load saved classifier first if requested
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_saved_clf = args.use_saved_classifier
        saved_classifier = None
        
        if use_saved_clf:
            logger.info(f"Attempting to load saved classifier from {args.optuna_dir}...")
            saved_classifier = load_saved_classifier(dataset, args.optuna_dir, device=device)
            
            if saved_classifier is not None:
                logger.info(f"✓ Successfully loaded saved classifier for {dataset}")
                # When using saved classifier, we need to match the normalization used in optuna
                # Optuna uses min-max normalization
                X_train_norm, X_test_norm, _ = min_max_normalize_train_test(
                    X_train, X_test, return_stats=True
                )
                norm_min = _["min"]
                norm_range = _["range"]
                # Convert to (N, T, D) for Keras-based code
                X_train_norm = X_train_norm.transpose(0, 2, 1)
                X_test_norm = X_test_norm.transpose(0, 2, 1)
            else:
                logger.warning(f"✗ Could not load saved classifier for {dataset}, will train new one")
                use_saved_clf = False
        
        # If not using saved classifier, use the original normalization
        if not use_saved_clf:
            # Convert to (N, T, D) for original processing
            X_train = X_train.transpose(0, 2, 1)
            X_test = X_test.transpose(0, 2, 1)
            
            # Normalize globally (flatten all values) using train statistics - matching original implementation
            X_train_norm, trained_scaler = time_series_normalize(
                data=X_train, n_timesteps=T, n_features=D
            )
            X_test_norm, _ = time_series_normalize(
                data=X_test, n_timesteps=T, n_features=D, scaler=trained_scaler
            )
            if hasattr(trained_scaler, "data_min_") and hasattr(trained_scaler, "data_range_"):
                norm_min = trained_scaler.data_min_.squeeze()
                norm_range = trained_scaler.data_range_.squeeze()

        # Split off validation from train (12.5% of train)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_norm,
            y_train,
            test_size=0.125,
            random_state=39,
            stratify=y_train,
        )

        # Conditional padding to multiple of 4 for CNN AE
        X_tr_padded, padding_size = conditional_pad(X_tr)
        X_val_padded, _ = conditional_pad(X_val)
        X_te_padded, _ = conditional_pad(X_test_norm)
        n_timesteps_padded = X_tr_padded.shape[1]
        logger.info(
            f"Data pre-processed, original #timesteps={T}, padded #timesteps={n_timesteps_padded}, D={D}, K={K}."
        )

        # 2) Evaluation models (LOF + NN) - fit one per class
        n_neighbors_lof = int(np.cbrt(X_tr.shape[0])) if X_tr.shape[0] > 1 else 1
        
        # Create dict of LOF and NN models for each class
        lof_estimators = {}
        nn_models = {}
        
        for class_label in range(K):
            class_data = X_tr[y_tr == class_label]
            if len(class_data) > 0:
                lof_est, nn_model = fit_evaluation_models(
                    n_neighbors_lof=min(n_neighbors_lof, len(class_data) - 1) if len(class_data) > 1 else 1,
                    n_neighbors_nn=1,
                    training_data=np.squeeze(class_data),
                )
                lof_estimators[class_label] = lof_est
                nn_models[class_label] = nn_model
            else:
                logger.warning(f"No samples found for class {class_label}")

        # 3) Classifier: Use saved PyTorch classifier or train LSTM-FCN (Keras)
        if use_saved_clf and saved_classifier is not None:
            logger.info("Using saved PyTorch classifier")
            classifier = saved_classifier
            
            # Evaluate classifier on test set
            y_pred_probs = classifier.predict(X_te_padded)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            
            acc = balanced_accuracy_score(y_true=y_test, y_pred=y_pred_classes)
            logger.info(f"Saved classifier test balanced accuracy: {acc:0.4f}")
        else:
            # Train new Keras classifier
            reset_seeds()
            classifier = LSTMFCNClassifier(n_timesteps_padded, D, n_output=K, n_LSTM_cells=8)
            
            if K == 2:
                classifier.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    loss="binary_crossentropy",
                    metrics=["accuracy"]
                )
            else:
                classifier.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    loss="categorical_crossentropy",
                    metrics=["accuracy"]
                )

            early_stopping_acc = keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=30, restore_best_weights=True
            )
            logger.info("Training LSTM-FCN classifier...")
            classifier.fit(
                X_tr_padded,
                to_categorical(y_tr, K),
                epochs=150,
                batch_size=32,
                shuffle=True,
                verbose=True,
                validation_data=(X_val_padded, to_categorical(y_val, K)),
                callbacks=[early_stopping_acc],
            )

            # Evaluate and prepare a fixed test subset (up to 50 samples)
            y_pred_probs = classifier.predict(X_te_padded)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)

            acc = balanced_accuracy_score(y_true=y_test, y_pred=y_pred_classes)
            logger.info(f"Classifier test balanced accuracy: {acc:0.4f}")

        # Use entire test set for CF evaluation (not a subset)
        X_test_eval_padded = X_te_padded
        y_test_eval_pred = y_pred_classes
        logger.info(f"Using entire test set for CF evaluation: {len(X_test_eval_padded)} samples")

        # 4) Autoencoder: 1D CNN and CF search
        reset_seeds()
        autoencoder = Autoencoder(n_timesteps_padded, D)
        autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4), loss="mse")
        early_stopping_loss = keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=5, restore_best_weights=True
        )
        logger.info("Training 1D-CNN autoencoder...")
        ae_hist = autoencoder.fit(
            X_tr_padded,
            X_tr_padded,
            epochs=50,
            batch_size=32,
            shuffle=True,
            verbose=True,
            validation_data=(X_val_padded, X_val_padded),
            callbacks=[early_stopping_loss],
        )
        ae_val_loss = float(np.min(ae_hist.history["val_loss"]))
        logger.info(f"Autoencoder best val loss: {ae_val_loss:0.6f}")

        # Step weights selection
        step_weights = np.ones((1, n_timesteps_padded, D))
        # if args.w_type == "global":
        #     step_weights = get_global_weights(
        #         X_tr_padded, y_tr, classifier, random_state=39
        #     )
        # elif args.w_type == "uniform":
        #     step_weights = np.ones((1, n_timesteps_padded, D))
        # elif args.w_type.lower() == "local":
        #     step_weights = "local"
        # elif args.w_type == "unconstrained":
        #     step_weights = np.zeros((1, n_timesteps_padded, D))
        # else:
        #     raise NotImplementedError(
        #         "w_type not implemented, choose 'local', 'global', 'uniform', or 'unconstrained'."
        #     )

        # Counterfactual search
        lr_list = args.lr_list
        logger.info(f"Trying learning rates: {lr_list}")
        best_lr, best_cf_model, best_cf_samples, _ = find_best_lr(
            classifier,
            X_samples=X_test_eval_padded,
            pred_labels=y_test_eval_pred,
            autoencoder=autoencoder,
            lr_list=lr_list,
            pred_margin_weight=pred_margin_weight,
            step_weights=step_weights,
            random_state=39,
            padding_size=padding_size,
            target_prob=tau_value,
        )
        logger.info(f"Best CF optimizer LR: {best_lr}")

        # Predicted labels of CFs
        cf_pred_probs = classifier.predict(best_cf_samples)
        cf_pred_labels = np.argmax(cf_pred_probs, axis=1)

        # Evaluation metrics (unpad for metric computation)
        X_test_eval_unpadded = remove_paddings(X_test_eval_padded, padding_size)
        best_cf_unpadded = remove_paddings(best_cf_samples, padding_size)
        
        # For multi-class, evaluate against the target class models
        # Target class is different from original prediction
        target_classes = 1 - y_test_eval_pred if K == 2 else cf_pred_labels
        
        # Get LOF and NN models for target classes
        lof_estimator_pos = lof_estimators.get(1, None)
        nn_model_pos = nn_models.get(1, None)
        lof_estimator_neg = lof_estimators.get(0, None)
        nn_model_neg = nn_models.get(0, None)

        # Evaluate results
        evaluate_res = evaluate(
            np.squeeze(X_test_eval_unpadded),
            np.squeeze(best_cf_unpadded),
            y_test_eval_pred,
            cf_pred_labels,
            lof_estimator_pos,
            lof_estimator_neg,
            nn_model_pos,
            nn_model_neg,
        )

        # Additional solver-based metrics (proximity, sparsity, plausibility)
        logger.info("Computing solver-based metrics (proximity, sparsity, plausibility)...")
        
        # Convert to torch tensors for solver metrics
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Transpose from (N, T, D) to (N, D, T) for solver
        X_test_eval_torch = torch.from_numpy(X_test_eval_padded.transpose(0, 2, 1)).float().to(device)
        best_cf_torch = torch.from_numpy(best_cf_samples.transpose(0, 2, 1)).float().to(device)
        
        # Create a temporary classifier wrapper for PyTorch
        # We need to convert the Keras classifier predictions to torch format
        # For metrics computation, we'll use the training data to get neighbors
        X_tr_torch = torch.from_numpy(X_tr_padded.transpose(0, 2, 1)).float().to(device)
        y_tr_torch = torch.from_numpy(y_tr).long().to(device)
        
        # Fit Isolation Forest on training samples for outlier scoring
        iforest_model = None
        X_tr_cpu = torch.nan_to_num(X_tr_torch.detach().float().cpu())
        if X_tr_cpu.size(0) > 0:
            iforest_model = IsolationForest(contamination=0.01, random_state=39)
            iforest_model.fit(X_tr_cpu.reshape(X_tr_cpu.size(0), -1).numpy())
        
        # Build class sample bank for neighbor lookup
        class_bank = build_class_sample_bank(X_tr_torch, y_tr_torch, K, device=device)
        
        # Desired labels: use the target class array (handles binary and multi-class)
        desired_labels_np = np.asarray(target_classes, dtype=np.int64)
        desired_labels_torch = torch.from_numpy(desired_labels_np).long().to(device)
        
        # Find k-NN neighbors in target class (using same config as train_cbf.py)
        cfg_metrics = SolverConfig(
            steps=250,
            lr=0.001,
            valid_objective="hinge",
            p_target_min=0.55,
            k_neighbors=10,
            dtw_gamma=0.1,
            dtw_normalize=True,
        )
        neighbors = select_knn_dtw_batch(
            X_test_eval_torch,
            class_bank,
            desired_labels_torch,
            k=cfg_metrics.k_neighbors,
            gamma=cfg_metrics.dtw_gamma,
            normalize=cfg_metrics.dtw_normalize,
        )
        
        # Compute metrics
        solver_metrics = compute_solver_metrics(
            X_test_eval_torch, 
            best_cf_torch, 
            neighbors,
            cf_pred_probs,
            target_classes,
            tau_value,
            dtw_gamma=cfg_metrics.dtw_gamma,
            dtw_normalize=cfg_metrics.dtw_normalize,
            iforest_model=iforest_model,
        )
        
        # Calculate metrics from train_cbf_optuna.py and add to solver_metrics
        targets_for_optuna_metrics = (y_test_eval_pred + 1) % K
        solver_metrics['validity_optuna'] = (cf_pred_labels == targets_for_optuna_metrics).mean()
        solver_metrics['proximity_l2_optuna'] = torch.norm(best_cf_torch - X_test_eval_torch, dim=(1, 2)).mean().item()
        solver_metrics['sparsity_l1_optuna'] = torch.norm(best_cf_torch - X_test_eval_torch, p=1, dim=(1, 2)).mean().item()

        outlier_all_log = (
            f"{solver_metrics['outlier_factor_all']:.6f}"
            if solver_metrics['outlier_factor_all'] is not None
            else "n/a"
        )
        outlier_valid_log = (
            f"{solver_metrics['outlier_factor_valid']:.6f}"
            if solver_metrics['outlier_factor_valid'] is not None
            else "n/a"
        )

        logger.info(f"Solver metrics - Proximity: {solver_metrics['proximity']:.6f}, "
                   f"Sparsity: {solver_metrics['sparsity']:.6f}, "
                   f"Plausibility: {solver_metrics['plausibility']:.6f}, "
                   f"Validity: {solver_metrics['validity']:.6f}, "
                   f"Outlier(all): {outlier_all_log}, "
                   f"Outlier(valid): {outlier_valid_log}\n"
                   f"Optuna Metrics - Validity: {solver_metrics['validity_optuna']:.4f}, "
                   f"Proximity L2: {solver_metrics['proximity_l2_optuna']:.6f}, "
                   f"Sparsity L1: {solver_metrics['sparsity_l1_optuna']:.6f}")

        # Write results (including original evaluate_res metrics and solver metrics)
        result_writer.write_result(
            fold_id=1, # Since we are not doing CV here
            method_name="1dCNN autoencoder",
            acc=acc,
            ae_loss=ae_val_loss,
            best_lr=best_lr,
            evaluate_res=evaluate_res,
            pred_margin_weight=pred_margin_weight,
            step_weight_type=args.w_type.lower(),
            threshold_tau=tau_value,
            solver_metrics=solver_metrics,
        )
        
        # Log solver metrics separately for now (could extend CSV format to include these)
        logger.info(f"[Summary] Dataset: {dataset} | Acc: {acc:.4f} | "
                   f"Validity: {evaluate_res[1]:.4f} | "
                   f"Proximity (L2): {evaluate_res[0]:.6f} | "
                   f"Solver Proximity: {solver_metrics['proximity']:.6f} | "
                   f"Solver Sparsity: {solver_metrics['sparsity']:.6f} | "
                   f"Solver Plausibility: {solver_metrics['plausibility']:.6f} | "
                   f"Solver Validity: {solver_metrics['validity']:.6f} | "
                   f"Optuna Validity: {solver_metrics['validity_optuna']:.4f} | "
                   f"Optuna Proximity L2: {solver_metrics['proximity_l2_optuna']:.6f} | "
                   f"Optuna Sparsity L1: {solver_metrics['sparsity_l1_optuna']:.6f}")

        # Save counterfactuals for downstream plotting
        def _to_ndt(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 2:
                return arr[:, None, :]
            if arr.ndim == 3:
                return arr.transpose(0, 2, 1)
            raise ValueError(f"Unexpected array shape {arr.shape}")

        x_orig_save = _to_ndt(X_test_eval_unpadded)
        x_cf_save = _to_ndt(best_cf_unpadded)
        out_cf_dir = output_dir / dataset
        out_cf_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_cf_dir / "counterfactuals.npz",
            x_orig=x_orig_save.astype(np.float32),
            x_cf=x_cf_save.astype(np.float32),
            y_true=np.asarray(y_test_eval_pred, dtype=np.int64),
            target_classes=np.asarray(target_classes, dtype=np.int64),
            cf_probs=np.asarray(cf_pred_probs, dtype=np.float32),
            norm_min=np.array(norm_min) if norm_min is not None else np.array([]),
            norm_range=np.array(norm_range) if norm_range is not None else np.array([]),
        )
        logger.info(f"Saved counterfactuals to {out_cf_dir / 'counterfactuals.npz'}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
