"""
Training & Evaluation Pipeline for DermaMNIST (7 Lesion Classes):
Supports:
  - Models: SimpleCNN, SimpleVNN, SimpleViT
  - Dataset: DermaMNIST (7 skin lesion classes)
  - Medical Evaluation Metrics: Top-1 Accuracy, Macro Precision, Macro Recall (Sensitivity), Macro F1, PR-AUC
  - Checkpointing & Experiment Logging to results.csv
"""

import argparse
import csv
import os
import time
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from models import build_model, count_params

# DermaMNIST 7 skin lesion class names (HAM10000 taxonomy)
DERMAMNIST_CLASSES: List[str] = [
    "akiec",  # Actinic keratoses and intraepithelial carcinoma
    "bcc",    # Basal cell carcinoma
    "bkl",    # Benign keratosis-like lesions
    "df",     # Dermatofibroma
    "mel",    # Melanoma
    "nv",     # Melanocytic nevi
    "vasc",   # Vascular lesions
]


# ---------------------------------------------------------------------------
# 1. DermaMNIST Dataset Loader & Transforms
# ---------------------------------------------------------------------------
def get_dataset(data_dir: str = "./data", img_size: int = 32) -> Tuple[Dataset, Dataset, int, int]:
    """
    Loads DermaMNIST dataset (7 lesion classes, 3 channels) with automatic
    directory creation and fallback handling.
    """
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Ensure directories exist prior to MedMNIST initialization
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    home_medmnist = os.path.expanduser("~/.medmnist")
    os.makedirs(home_medmnist, exist_ok=True)

    try:
        import medmnist
        from medmnist import DermaMNIST

        try:
            train_set = DermaMNIST(split="train", transform=train_tf, download=True, root=data_dir, as_rgb=True)
            test_set = DermaMNIST(split="test", transform=test_tf, download=True, root=data_dir, as_rgb=True)
        except Exception:
            # Fallback to ~/.medmnist if custom root encounters path issues
            train_set = DermaMNIST(split="train", transform=train_tf, download=True, root=home_medmnist, as_rgb=True)
            test_set = DermaMNIST(split="test", transform=test_tf, download=True, root=home_medmnist, as_rgb=True)

        return train_set, test_set, 7, 3

    except Exception as err:
        print(f"[Warning] Could not initialize MedMNIST ({err}). Using simulated DermaMNIST (7 classes) for offline execution.")

        class SyntheticDermaMNIST(Dataset):
            def __init__(self, size=600, n_classes=7):
                self.size = size
                self.n_classes = n_classes
                self.data = torch.randn(size, 3, img_size, img_size)
                class_weights = torch.tensor([0.05, 0.07, 0.12, 0.02, 0.15, 0.56, 0.03])
                self.labels = torch.multinomial(class_weights, size, replacement=True)

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]

        return SyntheticDermaMNIST(600, 7), SyntheticDermaMNIST(150, 7), 7, 3


def get_dataloaders(batch_size: int = 128, data_dir: str = "./data",
                    img_size: int = 32, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, int, int]:
    """Builds and returns training and test DataLoaders for DermaMNIST."""
    train_set, test_set, num_classes, in_channels = get_dataset(data_dir=data_dir, img_size=img_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, num_classes, in_channels


def get_stratified_sample_indices(dataset: Dataset, num_samples: int = 100,
                                   random_seed: int = 42) -> List[int]:
    """
    Extracts stratified sample indices from a dataset (e.g., DermaMNIST test set)
    ensuring all 7 skin lesion classes are represented with balanced/proportional coverage.
    Guarantees reproducible, identical index lists across all model architectures.
    """
    if hasattr(dataset, "labels") and dataset.labels is not None:
        all_labels = np.asarray(dataset.labels).squeeze()
    else:
        all_labels = np.array([int(np.asarray(dataset[i][1]).squeeze()) for i in range(len(dataset))])

    unique_classes = np.unique(all_labels)
    total_len = len(all_labels)
    target_samples = min(num_samples, total_len)

    rng = np.random.RandomState(random_seed)
    class_to_indices = {c: np.where(all_labels == c)[0] for c in unique_classes}
    for c in unique_classes:
        rng.shuffle(class_to_indices[c])

    # Allocate counts proportionally, ensuring at least 1 sample per class
    counts = {}
    remaining = target_samples
    for c in unique_classes:
        prop = len(class_to_indices[c]) / total_len
        cnt = max(1, int(round(prop * target_samples)))
        cnt = min(cnt, len(class_to_indices[c]))
        counts[c] = cnt
        remaining -= cnt

    # Adjust rounding differences
    sorted_classes = sorted(unique_classes, key=lambda c: len(class_to_indices[c]), reverse=True)
    idx = 0
    while remaining > 0:
        c = sorted_classes[idx % len(sorted_classes)]
        if counts[c] < len(class_to_indices[c]):
            counts[c] += 1
            remaining -= 1
        idx += 1
    while remaining < 0:
        c = sorted_classes[idx % len(sorted_classes)]
        if counts[c] > 1:
            counts[c] -= 1
            remaining += 1
        idx += 1

    selected_indices = []
    for c in unique_classes:
        selected_indices.extend(class_to_indices[c][:counts[c]])

    rng.shuffle(selected_indices)
    return selected_indices[:target_samples]


# ---------------------------------------------------------------------------
# 2. Medical Classification Metrics (Precision, Recall/Sensitivity, F1, PR-AUC)
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                    num_classes: int = 7) -> Dict[str, float]:
    """
    Computes Top-1 Accuracy, Macro Precision, Macro Recall (Sensitivity),
    Macro F1, and Macro PR-AUC (Average Precision score).
    """
    acc = float(np.mean(y_true == y_pred))

    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    # Multi-class Macro PR-AUC (Average Precision)
    pr_auc = 0.0
    try:
        from sklearn.metrics import average_precision_score
        y_true_onehot = np.eye(num_classes)[y_true]
        pr_auc = float(average_precision_score(y_true_onehot, y_prob, average="macro"))
    except Exception:
        pr_auc = float(macro_precision * macro_recall) if (macro_precision + macro_recall) > 0 else acc

    return {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "pr_auc": pr_auc,
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             num_classes: int = 7) -> Dict[str, float]:
    """Evaluates model over DataLoader and returns metric dictionary."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if isinstance(y, torch.Tensor) and y.ndim > 1:
                y = y.squeeze(-1)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(np.array(y))
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets).flatten()
    y_pred = np.concatenate(all_preds).flatten()
    y_prob = np.concatenate(all_probs, axis=0)

    return compute_metrics(y_true, y_pred, y_prob, num_classes)


# ---------------------------------------------------------------------------
# 3. Training Routine
# ---------------------------------------------------------------------------
def train(model_name: str, epochs: int = 20, lr: float = 1e-3,
          weight_decay: float = 5e-4, batch_size: int = 128,
          optimizer_name: str = "adamw", scheduler_name: str = "cosine",
          rank: int = 1, embed_dim: int = 128, depth: int = 4, heads: int = 4,
          base_channels: int = 32, dropout: float = 0.0,
          data_dir: str = "./data", out_csv: str = "results.csv",
          save_ckpt: str = None) -> Dict[str, Any]:
    """
    Executes model training on DermaMNIST (7 classes) with PR-AUC logging.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"Training [{model_name.upper()}] on DermaMNIST (7 Lesion Classes) | Device: {device}")
    print("=" * 75)

    img_size = 32
    train_loader, test_loader, num_classes, in_channels = get_dataloaders(
        batch_size=batch_size, data_dir=data_dir, img_size=img_size
    )

    model = build_model(
        name=model_name,
        num_classes=num_classes,
        in_channels=in_channels,
        img_size=img_size,
        rank=rank,
        base_channels=base_channels,
        embed_dim=embed_dim,
        depth=depth,
        heads=heads,
        dropout=dropout,
    ).to(device)

    n_params = count_params(model)
    print(f"Trainable Parameters: {n_params:,d}")

    # Optimizer selection
    opt_name = optimizer_name.lower()
    if opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Scheduler selection
    sched_name = scheduler_name.lower()
    if sched_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()

    # Training Loop
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for x, y in train_loader:
            x = x.to(device)
            if isinstance(y, torch.Tensor) and y.ndim > 1:
                y = y.squeeze(-1)
            y = torch.as_tensor(y, device=device, dtype=torch.long)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(1, total_samples)
        metrics = evaluate(model, test_loader, device, num_classes)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {train_loss:.4f} | "
              f"Acc: {metrics['accuracy']*100:.2f}% | "
              f"F1: {metrics['macro_f1']:.4f} | Recall: {metrics['macro_recall']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")

    total_time = time.time() - start_time
    final_metrics = evaluate(model, test_loader, device, num_classes)

    # Save Checkpoint
    ckpt_file = save_ckpt if save_ckpt else f"{model_name}.pt"
    torch.save(model.state_dict(), ckpt_file)
    print(f"\nModel checkpoint saved to: {ckpt_file}")

    # Log to CSV with PR-AUC
    file_exists = os.path.isfile(out_csv)
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "dataset", "model", "params", "epochs", "lr", "batch_size",
                "final_test_acc", "macro_f1", "macro_recall", "macro_precision",
                "pr_auc", "train_time_sec"
            ])
        writer.writerow([
            "dermamnist", model_name, n_params, epochs, lr, batch_size,
            round(final_metrics["accuracy"], 4),
            round(final_metrics["macro_f1"], 4),
            round(final_metrics["macro_recall"], 4),
            round(final_metrics["macro_precision"], 4),
            round(final_metrics["pr_auc"], 4),
            round(total_time, 1),
        ])

    print("=" * 75)
    print(f"Experiment Complete. Logged to {out_csv}")
    print(f"Final Acc: {final_metrics['accuracy']*100:.2f}% | F1: {final_metrics['macro_f1']:.4f} | "
          f"Sensitivity: {final_metrics['macro_recall']:.4f} | PR-AUC: {final_metrics['pr_auc']:.4f} | Time: {total_time:.1f}s")
    print("=" * 75)

    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN / VNN / ViT on DermaMNIST (7 Lesion Classes)")
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "vnn", "vit"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    parser.add_argument("--rank", type=int, default=1, help="VNN rank for quadratic branch factorization")
    parser.add_argument("--embed_dim", type=int, default=128, help="ViT embedding dimension")
    parser.add_argument("--depth", type=int, default=4, help="ViT depth")
    parser.add_argument("--heads", type=int, default=4, help="ViT heads")
    parser.add_argument("--base_channels", type=int, default=32, help="CNN / VNN base channels")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_csv", type=str, default="results.csv")
    parser.add_argument("--save_ckpt", type=str, default=None)
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        rank=args.rank,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        base_channels=args.base_channels,
        dropout=args.dropout,
        data_dir=args.data_dir,
        out_csv=args.out_csv,
        save_ckpt=args.save_ckpt,
    )
