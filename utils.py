import torch

import matplotlib.pyplot as plt
import numpy as np


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

            if isinstance(recon, tuple):
                recon = recon[0]

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


def plot_scores(scores, labels, threshold, title="Score Distribution"):
    scores_np = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()

    if len(scores_np) != len(labels_np):
        raise ValueError(f"Scores and labels size mismatch: {len(scores_np)} vs {len(labels_np)}")

    normal_scores = scores_np[labels_np == 0]
    anomaly_scores = scores_np[labels_np == 1]

    # Ignore only the most extreme 0.1% on each side
    plot_min = np.percentile(scores_np, 0.1)
    plot_max = np.percentile(scores_np, 99.9)

    # Fallback in degenerate cases
    if plot_min == plot_max:
        plot_min = scores_np.min()
        plot_max = scores_np.max()

    normal_in_range = normal_scores[(normal_scores >= plot_min) & (normal_scores <= plot_max)]
    anomaly_in_range = anomaly_scores[(anomaly_scores >= plot_min) & (anomaly_scores <= plot_max)]

    normal_clipped = len(normal_scores) - len(normal_in_range)
    anomaly_clipped = len(anomaly_scores) - len(anomaly_in_range)

    plt.figure(figsize=(12, 6))

    plt.hist(
        normal_in_range,
        bins=500,
        range=(plot_min, plot_max),
        density=True,
        alpha=0.6,
        color="tab:blue",
        label=f"Normal (clipped: {normal_clipped})",
    )

    plt.hist(
        anomaly_in_range,
        bins=500,
        range=(plot_min, plot_max),
        density=True,
        alpha=0.6,
        color="tab:orange",
        label=f"Anomaly (clipped: {anomaly_clipped})",
    )

    if plot_min <= threshold <= plot_max:
        plt.axvline(
            threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold = {threshold:.4f}",
        )

    plt.xlim(plot_min, plot_max)
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
