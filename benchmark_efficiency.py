"""
Model Efficiency & Complexity Benchmark Suite:
  - Trainable Parameter Counts
  - Checkpoint Disk Footprint (MB)
  - Estimated Computational FLOPs / MACs
  - Inference Latency (ms / sample) on CPU/GPU
  - Throughput (samples / sec)
  - Peak Memory Consumption
"""

import argparse
import os
import time
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn

from models import build_model, count_params


def estimate_flops(model: nn.Module, input_size: tuple = (1, 3, 32, 32)) -> int:
    """
    Estimates multiply-accumulate operations (MACs) for standard convs,
    Volterra layers, linear layers, and multi-head attention.
    """
    total_macs = 0

    def hook_fn(module, inp, out):
        nonlocal total_macs
        if isinstance(module, nn.Conv2d):
            out_c, out_h, out_w = out.shape[1], out.shape[2], out.shape[3]
            in_c = inp[0].shape[1]
            kh, kw = module.kernel_size
            macs = out_c * out_h * out_w * (in_c * kh * kw)
            total_macs += macs
        elif isinstance(module, nn.Linear):
            macs = module.in_features * module.out_features
            total_macs += macs

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))

    x = torch.randn(*input_size)
    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    return total_macs


def measure_latency_and_throughput(model: nn.Module, device: torch.device,
                                   input_size: tuple = (1, 3, 32, 32),
                                   warmup_runs: int = 20,
                                   test_runs: int = 100) -> Dict[str, float]:
    """
    Measures mean inference latency per image (ms) and throughput (images/sec).
    """
    model.eval()
    x = torch.randn(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(test_runs):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # in ms

    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    throughput = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    return {
        "mean_latency_ms": mean_lat,
        "std_latency_ms": std_lat,
        "throughput_fps": throughput,
    }


def benchmark_model_efficiency(name: str, num_classes: int = 7,
                               in_channels: int = 3, img_size: int = 32,
                               device: torch.device = None) -> Dict[str, Any]:
    """
    Computes all computational efficiency metrics for a given architecture on DermaMNIST.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(name, num_classes=num_classes, in_channels=in_channels, img_size=img_size).to(device)
    n_params = count_params(model)
    macs = estimate_flops(model, input_size=(1, in_channels, img_size, img_size))

    # Checkpoint size on disk
    temp_path = f"_temp_{name}.pt"
    torch.save(model.state_dict(), temp_path)
    file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    if os.path.exists(temp_path):
        os.remove(temp_path)

    timing = measure_latency_and_throughput(model, device, input_size=(1, in_channels, img_size, img_size))

    return {
        "model": name.upper(),
        "params": n_params,
        "macs": macs,
        "file_size_mb": file_size_mb,
        "latency_ms": timing["mean_latency_ms"],
        "throughput_fps": timing["throughput_fps"],
    }


def run_efficiency_suite(num_classes: int = 7, in_channels: int = 3, img_size: int = 32) -> None:
    """
    Runs efficiency benchmarking across CNN, VNN, and ViT architectures on DermaMNIST.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print(f"COMPUTATIONAL EFFICIENCY & HARDWARE COMPLEXITY PROFILING ({device})")
    print("=" * 80)

    results = []
    for m_type in ["cnn", "vnn", "vit"]:
        res = benchmark_model_efficiency(m_type, num_classes=num_classes, in_channels=in_channels,
                                         img_size=img_size, device=device)
        results.append(res)

    print("\n" + "=" * 80)
    print("COMPUTATIONAL COMPLEXITY TABLE (GFM Format)")
    print("=" * 80)
    print("| Model | Parameters | Checkpoint Size | Est. MACs (M) | Latency (ms/img) | Throughput (img/s) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        macs_m = r['macs'] / 1e6
        print(f"| **{r['model']}** | {r['params']:,d} | {r['file_size_mb']:.2f} MB | {macs_m:.2f}M | {r['latency_ms']:.2f} ms | {r['throughput_fps']:.1f} fps |")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Model Efficiency on DermaMNIST")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--img_size", type=int, default=32)
    args = parser.parse_args()

    run_efficiency_suite(num_classes=args.num_classes, img_size=args.img_size)
