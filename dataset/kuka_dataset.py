# dataset/kuka_dataset.py

import torch
from torch.utils.data import Dataset


class KukaDataset(Dataset):
    def __init__(self, x, y, transform=None, dtype=torch.float32):
        if len(x) != len(y):
            raise ValueError(f"x and y have different lengths: {len(x)} vs {len(y)}")

        self.x = torch.as_tensor(x, dtype=dtype)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y