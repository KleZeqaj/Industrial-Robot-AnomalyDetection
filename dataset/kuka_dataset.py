from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class KukaDataset(Dataset):

    def __init__(self, root_dir, normal_only=False, transform=None, dtype=torch.float32):
        self.root_dir = Path(root_dir)
        self.transform = transform

        normal = np.load(self.root_dir / "KukaNormal.npy")
        slow = np.load(self.root_dir / "KukaSlow.npy")

        x_normal = normal
        y_normal = np.zeros(len(normal), dtype=np.int64)

        x_slow = slow[:, :-1]
        y_slow = slow[:, -1].astype(np.int64)

        self.x = np.concatenate([x_normal, x_slow], axis=0)
        self.y = np.concatenate([y_normal, y_slow], axis=0)

        self.x = torch.tensor(self.x, dtype=dtype)
        self.y = torch.tensor(self.y, dtype=torch.long)

        if normal_only:
            mask = self.y == 0
            self.x = self.x[mask]
            self.y = self.y[mask]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y
