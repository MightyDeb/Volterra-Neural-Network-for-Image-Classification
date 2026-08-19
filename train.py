"""
Train and evaluate CNN / VNN / ViT on CIFAR-10 and log accuracy, params, time.

Usage (run in Colab/Kaggle with GPU):
    python train.py --model vnn --epochs 20
    python train.py --model cnn --epochs 20
    python train.py --model vit --epochs 20

Results are appended to results.csv so you can plot accuracy vs. params
vs. training time across all three models after running each once.
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import build_model, count_params


def get_dataloaders(batch_size=128, data_dir="./data"):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def train(model_name, epochs=20, lr=1e-3, batch_size=128, data_dir="./data", out_csv="results.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(model_name).to(device)
    n_params = count_params(model)
    print(f"Model: {model_name} | Trainable params: {n_params:,}")

    train_loader, test_loader = get_dataloaders(batch_size, data_dir)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader, device)
        print(f"[{model_name}] epoch {epoch+1}/{epochs} | loss {train_loss:.4f} | test_acc {test_acc:.4f}")

    total_time = time.time() - start_time
    final_acc = evaluate(model, test_loader, device)

    # Save checkpoint for use in plot_comparison.py
    ckpt_path = f"{model_name}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # Log results
    file_exists = os.path.isfile(out_csv)
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "params", "epochs", "final_test_acc", "train_time_sec"])
        writer.writerow([model_name, n_params, epochs, round(final_acc, 4), round(total_time, 1)])

    print(f"\nDone. Final test accuracy: {final_acc:.4f} | Time: {total_time:.1f}s")
    print(f"Result appended to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "vnn", "vit"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    train(args.model, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
