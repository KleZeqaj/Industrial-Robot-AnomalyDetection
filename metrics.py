from typing import Dict, Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    auc,
)

def compute_pointwise_metrics(
    labels,
    preds,
    scores,
    beta: float = 2.0,
    zero_division: int = 0,
) -> Dict[str, Any]:
    """
    Compute point-wise anomaly-detection metrics.

    Parameters
    ----------
    labels:
        Ground-truth binary labels.
        Expected convention:
            0 = normal
            1 = anomaly

    preds:
        Thresholded binary predictions.
        Expected convention:
            0 = predicted normal
            1 = predicted anomaly

    scores:
        Continuous anomaly scores, where larger values indicate
        stronger evidence of anomalous behaviour.

        For the current AE implementation, these are reconstruction-based
        anomaly scores computed before thresholding.

    beta:
        Beta parameter for F-beta.
        beta > 1 gives more weight to recall.
        beta < 1 gives more weight to precision.

    zero_division:
        Value returned by precision/recall/F-score metrics when the
        denominator is zero. Usually 0 is appropriate.

    Returns
    -------
    dict
        Dictionary containing point-wise metrics.
    """

    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")

    # Convert torch tensors or array-like inputs to flat NumPy arrays.
    labels_np = _to_1d_numpy(labels, name="labels")
    preds_np = _to_1d_numpy(preds, name="preds")
    scores_np = _to_1d_numpy(scores, name="scores")

    # Basic shape validation.
    n = len(labels_np)

    if len(preds_np) != n:
        raise ValueError(
            f"labels and preds must have the same length: "
            f"{n} != {len(preds_np)}"
        )

    if len(scores_np) != n:
        raise ValueError(
            f"labels and scores must have the same length: "
            f"{n} != {len(scores_np)}"
        )

    # Validate binary ground truth and predictions.
    _validate_binary(labels_np, name="labels")
    _validate_binary(preds_np, name="preds")

    # Scores must be finite because PR-curve computation assumes
    # meaningful real-valued ranking scores.
    if not np.isfinite(scores_np).all():
        raise ValueError("scores contains NaN or infinite values")

    # Threshold-dependent metrics.
    precision = precision_score(
        labels_np,
        preds_np,
        pos_label=1,
        zero_division=zero_division,
    )

    recall = recall_score(
        labels_np,
        preds_np,
        pos_label=1,
        zero_division=zero_division,
    )

    f1 = f1_score(
        labels_np,
        preds_np,
        pos_label=1,
        zero_division=zero_division,
    )

    f_beta = fbeta_score(
        labels_np,
        preds_np,
        beta=beta,
        pos_label=1,
        zero_division=zero_division,
    )

    accuracy = accuracy_score(
        labels_np,
        preds_np,
    )

    balanced_accuracy = balanced_accuracy_score(
        labels_np,
        preds_np,
    )

    # Threshold-free PR metrics.
    #
    # Average Precision (AP) and trapezoidal PR-AUC are related but
    # not identical quantities, so compute both explicitly.
    average_precision = average_precision_score(
        labels_np,
        scores_np,
    )

    pr_precision, pr_recall, _ = precision_recall_curve(
        labels_np,
        scores_np,
        pos_label=1,
    )

    # sklearn's precision_recall_curve may return recall in decreasing
    # order. auc() handles monotonic increasing or decreasing x-values.
    pr_auc = auc(
        pr_recall,
        pr_precision,
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        f"f_{beta:g}": float(f_beta),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "pr_auc": float(pr_auc),
        "average_precision": float(average_precision),
        "beta": float(beta),
        "n_samples": int(n),
        "n_normal": int(np.sum(labels_np == 0)),
        "n_anomaly": int(np.sum(labels_np == 1)),
    }

def _to_1d_numpy(values, name: str) -> np.ndarray:
    """
    Convert torch tensor or array-like input to a 1D NumPy array.
    """
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    else:
        values = np.asarray(values)

    values = np.asarray(values).reshape(-1)

    if len(values) == 0:
        raise ValueError(f"{name} cannot be empty")

    return values


def _validate_binary(values: np.ndarray, name: str) -> None:
    """
    Ensure an array contains only binary values {0, 1}.
    """
    unique_values = np.unique(values)

    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(
            f"{name} must contain only binary values {{0, 1}}, "
            f"found {unique_values.tolist()}"
        )