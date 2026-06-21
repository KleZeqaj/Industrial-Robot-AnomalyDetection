import torch
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np


def split_kuka(dataset, train_ratio=0.7, val_ratio=0.15):
    labels = dataset.y

    normal_idx = torch.where(labels == 0)[0]
    anomaly_idx = torch.where(labels == 1)[0]

    # shuffle normal data
    perm = torch.randperm(len(normal_idx))
    normal_idx = normal_idx[perm]

    # split normal into train / val / test
    n_total = len(normal_idx)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    train_normal_idx = normal_idx[:n_train]
    val_normal_idx = normal_idx[n_train : n_train + n_val]
    test_normal_idx = normal_idx[n_train + n_val :]

    # build anomaly splits: split anomalies separately
    perm_anom = torch.randperm(len(anomaly_idx))
    anomaly_idx = anomaly_idx[perm_anom]

    n_val_anom = int(0.5 * len(anomaly_idx))
    val_anomaly_idx = anomaly_idx[:n_val_anom]
    test_anomaly_idx = anomaly_idx[n_val_anom:]

    train_dataset = Subset(dataset, train_normal_idx)
    val_dataset = Subset(dataset, torch.cat([val_normal_idx, val_anomaly_idx]))
    test_dataset = Subset(dataset, torch.cat([test_normal_idx, test_anomaly_idx]))

    return train_dataset, val_dataset, test_dataset


def compute_threshold(model, dataloader, normal_label=0, quantile=0.95):
    model.eval()
    device = next(model.parameters()).device

    all_errors = []

    with torch.no_grad():
        for x, y in dataloader:
            mask = y == normal_label
            if mask.sum().item() == 0:
                continue

            x = x[mask].to(device)
            recon = model(x)
            # err = (recon - x).pow(2).mean(dim=1)
            err = (recon - x).pow(2).topk(6, dim=1).values.mean(dim=1)
            all_errors.append(err.cpu())  # Move to CPU to keep memory footprint low

    all_errors = torch.cat(all_errors)

    t = torch.quantile(all_errors, quantile)
    return t


def plot_reconstruction_errors(errors, labels, threshold, quantile):
    errors_np = errors.cpu().numpy()
    labels_np = labels.cpu().numpy()
    threshold = threshold.item()

    if len(errors_np) != len(labels_np):
        raise ValueError(f"Errors and labels size mismatch: {len(errors_np)} vs {len(labels_np)}")

    normal_errors = errors_np[labels_np == 0]
    slow_errors = errors_np[labels_np == 1]

    # A few extreme outliers can stretch the axis and hide the useful part of the distribution.
    plot_max = float(np.percentile(errors_np, 99.9))
    if plot_max <= 0:
        plot_max = float(errors_np.max())

    normal_in_range = normal_errors[normal_errors <= plot_max]
    slow_in_range = slow_errors[slow_errors <= plot_max]
    normal_clipped = len(normal_errors) - len(normal_in_range)
    slow_clipped = len(slow_errors) - len(slow_in_range)

    plt.figure(figsize=(12, 6))
    plt.hist(
        normal_in_range,
        bins=120,
        range=(0, plot_max),
        alpha=0.6,
        color="tab:blue",
        label=f"Normal (clipped: {normal_clipped})",
    )
    plt.hist(
        slow_in_range,
        bins=120,
        range=(0, plot_max),
        alpha=0.6,
        color="tab:orange",
        label=f"Slow (clipped: {slow_clipped})",
    )

    if threshold <= plot_max:
        plt.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.6f}")
    else:
        plt.axvline(
            plot_max, color="red", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.6f} (out of range)"
        )

    plt.xlim(0, plot_max)
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Count")
    plt.title(f"Reconstruction Error Distribution (Test Set, {int(quantile*100)}th Percentile Zoom)")
    plt.legend()
    plt.tight_layout()
    plt.show()
