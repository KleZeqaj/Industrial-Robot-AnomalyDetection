import torch
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np


def split_kuka(dataset, train_ratio=0.7, val_ratio=0.15):
    """Split the Kuka dataset into train, validation, and test sets.

    Normal samples are split into train/val/test according to `train_ratio` and
    `val_ratio` (the remainder goes to test), so the training set contains only
    normal data. Anomalous samples are split 50/50 and added only to the
    validation and test sets.

    Args:
        dataset: Dataset exposing a `y` tensor of labels (0 for normal, 1 for anomaly).
        train_ratio: Fraction of normal samples used for training.
        val_ratio: Fraction of normal samples used for validation. The
            remaining normal samples go to the test set.

    Returns:
        tuple[Subset, Subset, Subset]: train, validation, and test datasets.
    """
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


def compute_threshold(model, dataloader, normal_label=0, quantile=0.95, err_fn=None):
    """Compute the threshold as the given quantile of the reconstruction
    error distribution of only normal samples.

    Args:
        model: Model to use for the threshold computation.
        dataloader: Iterable of (x, y) batches, where `x` are the input samples
            and `y` are their labels.
        normal_label: Label value identifying normal (non-anomalous) samples.
            Only samples with this label are used to compute the threshold.
        quantile: Quantile (between 0 and 1) of the normal-sample error
            distribution used as the threshold.
        err_fn: Optional callable `err_fn(recon, x) -> Tensor` returning a
            per-sample reconstruction error. If None, defaults to the mean
            squared error per sample.

    Returns:
        float: The computed threshold.
    """
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
            err = (recon - x).pow(2).mean(dim=1) if err_fn is None else err_fn(recon, x)
            all_errors.append(err.cpu())  # move to CPU to keep memory footprint low

    all_errors = torch.cat(all_errors)

    t = torch.quantile(all_errors, quantile)
    return t.item()


def plot_reconstruction_errors(errors, labels, threshold, quantile):
    errors_np = errors.cpu().numpy()
    labels_np = labels.cpu().numpy()

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
        density=True,
        label=f"Normal (clipped: {normal_clipped})",
    )
    plt.hist(
        slow_in_range,
        bins=120,
        range=(0, plot_max),
        alpha=0.6,
        color="tab:orange",
        density=True,
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
    plt.ylabel("Density")
    plt.title(f"Reconstruction Error Distribution (Test Set, {int(quantile*100)}th Percentile Zoom)")
    plt.legend()
    plt.tight_layout()
    plt.show()
