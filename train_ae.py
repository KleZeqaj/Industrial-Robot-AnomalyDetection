import argparse
from datetime import datetime
import os

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset.kuka_dataset import KukaDataset
from models.AE import AE
from utils import compute_threshold, split_kuka, plot_reconstruction_errors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate autoencoder model for anomaly detection on the Kuka dataset."
    )

    parser.add_argument("--data-dir", type=str, default="./data", help="Path to the dataset directory.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for the data loaders.")
    parser.add_argument("--latent-dim", type=int, default=4, help="Dimensionality of the autoencoder latent space.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--quantile", type=float, default=0.85, help="Quantile used to compute the anomaly threshold.")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top errors averaged in the reconstruction error function."
    )
    parser.add_argument(
        "--weights-out",
        type=str,
        default=f"weights/ae_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth",
        help="Path to save the trained model weights.",
    )
    parser.add_argument(
        "--weights-in",
        type=str,
        default=None,
        help="Path to the weights loaded for evaluation. If provided, training is skipped.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")

    return parser.parse_args()


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def build_dataloaders(data_dir, batch_size):
    dataset = KukaDataset(data_dir, normalize=True)
    train_dataset, val_dataset, test_dataset = split_kuka(dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, train_loader, val_loader, test_loader


def err_fn(pred, x, top_k):
    return (pred - x).pow(2).topk(top_k, dim=1).values.mean(dim=1)


def train(model, training_loader, device, lr, epochs):
    reconstruct_loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    print(f"Starting training for {epochs} epochs...")
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


def evaluate(model, test_loader, device, threshold, top_k):
    all_preds = []
    all_labels = []
    all_errors = []

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            recon = model(x)

            err = err_fn(recon, x, top_k)
            preds = (err > threshold).int()

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_errors.append(err.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_errors = torch.cat(all_errors)

    return all_preds, all_labels, all_errors


def report_results(all_preds, all_labels):
    print(f"Anomalies detected: {(all_preds[all_labels == 1]).sum().item()} / {(all_labels == 1).sum().item()}")

    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy(), labels=[0, 1])
    print("Confusion Matrix:")
    print(cm)

    print(classification_report(all_labels.numpy(), all_preds.numpy(), target_names=["Normal", "Anomaly"]))


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = get_device()

    train_dataset, train_loader, val_loader, test_loader = build_dataloaders(args.data_dir, args.batch_size)

    x, _ = train_dataset[0]
    input_dim = x.shape[0]

    ae = AE(input_dim, args.latent_dim).to(device)

    if args.weights_in is not None:
        print(f"Loading pre-trained weights from {args.weights_in}, skipping training.")
        ae.load_state_dict(torch.load(args.weights_in, map_location=device))
    else:
        train(ae, train_loader, device, lr=args.lr, epochs=args.epochs)
        output_dir = os.path.dirname(args.weights_out)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(ae.state_dict(), args.weights_out)

    t = compute_threshold(
        ae,
        val_loader,
        quantile=args.quantile,
        err_fn=lambda pred, x: err_fn(pred, x, args.top_k),
    )
    print(f"Threshold: {t:.6f}")

    all_preds, all_labels, all_errors = evaluate(ae, test_loader, device, t, args.top_k)
    report_results(all_preds, all_labels)
    plot_reconstruction_errors(all_errors, all_labels, t, quantile=args.quantile)


if __name__ == "__main__":
    main()
