from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def _eval_classifier(
    model: nn.Module, loader: DataLoader, device: str
) -> dict[str, float]:
    model.eval()
    tot_loss = 0.0
    tot_correct = 0
    tot_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, reduction="sum")
            tot_loss += float(loss.item())
            preds = logits.argmax(dim=1)
            tot_correct += (preds == yb).sum().item()
            tot_count += yb.numel()
    avg_loss = tot_loss / max(1, tot_count)
    acc = tot_correct / max(1, tot_count)
    return {"loss": avg_loss, "acc": acc}


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cpu",
    patience: int = 20,
    min_delta: float = 1e-4,
    weight_decay: float = 0.0,
) -> dict[str, float]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    best_acc = 0.0
    patience_counter = 0

    pbar = tqdm(range(epochs), desc="Classifier epochs", leave=False)
    for epoch in pbar:
        model.train()
        ep_loss_sum = 0.0
        ep_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            ep_loss_sum += float(loss.item()) * yb.numel()
            ep_count += yb.numel()

        train_loss = ep_loss_sum / max(1, ep_count)

        val_metrics = _eval_classifier(model, val_loader, device)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]

        pbar.set_postfix(
            {
                "train": f"{train_loss:.4f}",
                "val": f"{val_loss:.4f}",
                "acc": f"{val_acc:.3f}",
                "pat": f"{patience_counter}",
            }
        )

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_acc = val_acc
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_loss": float(best_val if best_state is not None else val_loss),
        "best_val_acc": float(best_acc if best_state is not None else val_acc),
        "best_test_acc": float(best_acc if best_state is not None else val_acc),  # Alias for compatibility
        "epochs_trained": int(epoch + 1),
    }
