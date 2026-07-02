from dataset.kuka_dataset import KukaDataset
from torch.utils.data import DataLoader
from models.AE import AE
from utils import split_kuka, compute_threshold, plot_reconstruction_errors
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train(model, training_loader, device, lr=1e-3, epochs=500):
    reconstruct_loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, _ in training_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            loss = reconstruct_loss_fn(model(inputs), inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}: loss {running_loss / len(training_loader):.4f}")


def err_fn(pred, x):
    return (pred - x).pow(2).topk(5, dim=1).values.mean(dim=1)
    # return (pred - x).pow(2).mean(dim=1).cpu()


DATA_DIR = "./data"

dataset = KukaDataset(DATA_DIR, normalize=True)


train_dataset, val_dataset, test_dataset = split_kuka(dataset)

BATCH_SIZE = 1024
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

x, y = train_dataset[0]
input_dim = x.shape[0]
latent_dim = 4

# best: lr=0.001, epochs=1500, latent_dim=4
ae = AE(input_dim, latent_dim).to(device)
# train(ae, train_loader, device, lr=0.001, epochs=1500)
# torch.save(ae.state_dict(), "model_weights.pth")
ae.load_state_dict(torch.load("ae_best.pth"))

QUANTILE = 0.85
t = compute_threshold(ae, val_loader, quantile=QUANTILE, err_fn=err_fn)
print(f"Threshold: {t:.6f}")


all_preds = []
all_labels = []
all_errors = []

ae.eval()
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        recon = ae(x)

        err = err_fn(recon, x)
        preds = (err > t).int()

        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
        all_errors.append(err.cpu())

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
all_errors = torch.cat(all_errors)

print(f"Anomalies detected: {(all_preds[all_labels == 1]).sum().item()} / {(all_labels == 1).sum().item()}")

cm = confusion_matrix(all_labels.cpu().numpy(), all_preds.cpu().numpy(), labels=[0, 1])

print("Confusion Matrix:")
print(cm)

print(classification_report(all_labels.cpu().numpy(), all_preds.cpu().numpy(), target_names=["Normal", "Anomaly"]))

plot_reconstruction_errors(all_errors, all_labels, t, quantile=QUANTILE)


############## FEATURE ERRORS #############
feature_names = np.load(f"{DATA_DIR}/KukaColumnNames.npy", allow_pickle=True)

normal_errors = []
anomaly_errors = []

ae.eval()

with torch.no_grad():
    for x, y in test_loader:

        x = x.to(device)
        recon = ae(x)

        # per-feature squared error
        feature_err = (recon - x).pow(2)

        normal_mask = y == 0
        anomaly_mask = y == 1

        if normal_mask.any():
            normal_errors.append(feature_err[normal_mask].cpu())

        if anomaly_mask.any():
            anomaly_errors.append(feature_err[anomaly_mask].cpu())

normal_errors = torch.cat(normal_errors, dim=0)
anomaly_errors = torch.cat(anomaly_errors, dim=0)

mean_normal_error = normal_errors.mean(dim=0)
mean_anomaly_error = anomaly_errors.mean(dim=0)

# separation_ratio = (mean_anomaly_error / (mean_normal_error + 1e-8)).numpy()
normal_mean = normal_errors.mean(dim=0)
normal_std = normal_errors.std(dim=0)
separation_ratio = ((anomaly_errors.mean(dim=0) - normal_mean) / (normal_std + 1e-8)).numpy()

order = np.argsort(separation_ratio)

sorted_ratios = separation_ratio[order]
sorted_names = np.array(feature_names)[order]
plt.figure(figsize=(12, 20))
plt.barh(range(len(sorted_ratios)), sorted_ratios)
plt.yticks(range(len(sorted_names)), sorted_names, fontsize=8)
plt.xlabel("Mean Anomaly Reconstruction Error / Mean Normal Reconstruction Error")
plt.ylabel("Feature")
plt.title("Feature-wise Reconstruction Error Separation")

plt.tight_layout()


############# LATENT SPACE PLOT ###########

ae.eval()

all_z = []
all_labels = []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)

        z = ae.encoder(x)

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
plt.show()
