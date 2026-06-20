from dataset.kuka_dataset import KukaDataset
from torch.utils.data import DataLoader
from models.AE import AE
from utils import split_kuka, compute_threshold, plot_reconstruction_errors
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

def train(model, training_loader, lr=1e-3, epochs=500):
    reconstruct_loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, _ in training_loader:
            optimizer.zero_grad()
            loss = reconstruct_loss_fn(model(inputs), inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}: loss {running_loss / len(training_loader):.4f}")


DATA_DIR = "./data"

dataset = KukaDataset(DATA_DIR)


train_dataset, val_dataset, test_dataset = split_kuka(dataset)

BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


x, y = train_dataset[0]
input_dim = x.shape[0]
latent_dim = 32

ae = AE(input_dim, latent_dim)
train(ae, train_loader, lr=0.001, epochs=10)


t = compute_threshold(ae, val_loader, quantile=0.95)
print(f"Threshold: {t.item():.6f}")


all_preds = []
all_labels = []
all_errors = []

with torch.no_grad():
    for x, y in test_loader:
        recon = ae(x)
        err = (recon - x).pow(2).mean(dim=1)

        preds = (err > t).int()

        all_preds.append(preds)
        all_labels.append(y)
        all_errors.append(err)

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
all_errors = torch.cat(all_errors)

print(f"Anomalies detected: {(all_preds[all_labels == 1]).sum().item()} / {(all_labels == 1).sum().item()}")



cm = confusion_matrix(all_labels.cpu().numpy(), all_preds.cpu().numpy(), labels=[0, 1])

print("Confusion Matrix:")
print(cm)

print(classification_report(all_labels.cpu().numpy(), all_preds.cpu().numpy(), target_names=["Normal", "Anomaly"]))

plot_reconstruction_errors(all_errors, all_labels, t)