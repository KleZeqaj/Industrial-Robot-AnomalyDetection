from pathlib import Path
from dataclasses import dataclass
import numpy as np


@dataclass
class KukaSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def split_kuka_numpy(
    data_dir,
    train_ratio=0.70,
    val_ratio=0.15,
    anomaly_val_ratio=0.50,
    seed=None,
    binary_labels=True,
):
    data_dir = Path(data_dir)

    x_normal = np.load(data_dir / "KukaNormal.npy")
    slow_raw = np.load(data_dir / "KukaSlow.npy")

    x_slow = slow_raw[:, :-1]
    y_slow = slow_raw[:, -1].astype(np.int64)

    if binary_labels:
        y_slow = (y_slow != 0).astype(np.int64)

    y_normal = np.zeros(len(x_normal), dtype=np.int64)

    rng = np.random.default_rng(seed)

    # split dei normali
    normal_idx = rng.permutation(len(x_normal))

    n_total = len(normal_idx)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_normal_idx = normal_idx[:n_train]
    val_normal_idx = normal_idx[n_train:n_train + n_val]
    test_normal_idx = normal_idx[n_train + n_val:]

    # split degli slow/anomali
    slow_idx = rng.permutation(len(x_slow))
    n_val_slow = int(anomaly_val_ratio * len(slow_idx))

    val_slow_idx = slow_idx[:n_val_slow]
    test_slow_idx = slow_idx[n_val_slow:]

    x_train = x_normal[train_normal_idx]
    y_train = y_normal[train_normal_idx]

    x_val = np.concatenate([
        x_normal[val_normal_idx],
        x_slow[val_slow_idx],
    ], axis=0)

    y_val = np.concatenate([
        y_normal[val_normal_idx],
        y_slow[val_slow_idx],
    ], axis=0)

    x_test = np.concatenate([
        x_normal[test_normal_idx],
        x_slow[test_slow_idx],
    ], axis=0)

    y_test = np.concatenate([
        y_normal[test_normal_idx],
        y_slow[test_slow_idx],
    ], axis=0)

    # shuffle finale di validation/test, per evitare blocchi ordinati normal-then-slow
    val_perm = rng.permutation(len(x_val))
    test_perm = rng.permutation(len(x_test))

    return KukaSplits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val[val_perm],
        y_val=y_val[val_perm],
        x_test=x_test[test_perm],
        y_test=y_test[test_perm],
    )