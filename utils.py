import torch
from torch.utils.data import Subset


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


def compute_threshold(model, dataloader):
    model.eval()

    all_errors = []

    with torch.no_grad():
        for x, _ in dataloader:
            recon = model(x)
            err = (recon - x).pow(2).mean(dim=1)
            all_errors.append(err)

    all_errors = torch.cat(all_errors)

    t = all_errors.mean() + 4 * all_errors.std()
    return t
