import torch
import itertools
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

    plot_min = np.min(scores_np)

    norm_upper = np.percentile(normal_scores, 99.5) if len(normal_scores) > 0 else 0
    anom_upper = np.percentile(anomaly_scores, 75) if len(anomaly_scores) > 0 else 0 # cap anomalies distr to 75%, everything after is discarded
    
    # Set the upper bound to view the decision boundary and the bulk of both distributions
    plot_max = max(norm_upper * 1.5, anom_upper)

    # Safeguard: Ensure the threshold is always comfortably visible in the plot
    if threshold > plot_max and threshold < scores_np.max():
        plot_max = threshold * 1.25

    # Fallback in degenerate cases
    if plot_min >= plot_max:
        plot_min = scores_np.min()
        plot_max = scores_np.max()

    normal_in_range = normal_scores[(normal_scores >= plot_min) & (normal_scores <= plot_max)]
    anomaly_in_range = anomaly_scores[(anomaly_scores >= plot_min) & (anomaly_scores <= plot_max)]

    normal_clipped = len(normal_scores) - len(normal_in_range)
    anomaly_clipped = len(anomaly_scores) - len(anomaly_in_range)

    plt.figure(figsize=(12, 6))

    plt.hist(
        normal_in_range,
        bins=150, 
        range=(plot_min, plot_max),
        density=True,
        alpha=0.6,
        color="tab:blue",
        label=f"Normal (clipped: {normal_clipped})",
    )

    plt.hist(
        anomaly_in_range,
        bins=150,
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

def plot_latent(model, dataloader, latent_dim, title=""):

    model.eval()
    device = next(model.parameters()).device

    all_z = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)

            z = model.encoder(x)

            all_z.append(z.cpu())
            all_labels.append(y)

    all_z = torch.cat(all_z).numpy()
    all_labels = torch.cat(all_labels).numpy()

    pairs = list(itertools.combinations(range(latent_dim), 2))

    ncols = 4
    nrows = (len(pairs) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    axes = axes.flatten()

    for ax, (i, j) in zip(axes, pairs):

        normal = all_labels == 0
        anomaly = all_labels == 1

        ax.scatter(all_z[normal, i], all_z[normal, j], s=3, alpha=0.3, label="Normal")
        ax.scatter(all_z[anomaly, i], all_z[anomaly, j], s=3, alpha=0.3, label="Anomaly")
        ax.set_title(f"z{i} vs z{j}")

        x_low, x_high = np.percentile(all_z[:, i], [1, 99])
        y_low, y_high = np.percentile(all_z[:, j], [1, 99])

        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)

    for ax in axes[len(pairs) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.title(title)
    plt.show()