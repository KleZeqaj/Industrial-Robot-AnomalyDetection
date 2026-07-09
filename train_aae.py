import argparse
from datetime import datetime
import os

import torch
import torch.nn as nn

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from torch.utils.data import DataLoader

from dataset.kuka_dataset import KukaDataset
from dataset.kuka_splits import split_kuka_numpy
from models.AAE import AAE
from processing.kuka_processing import KukaPreprocessor
from utils import compute_threshold, plot_reconstruction_errors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate autoencoder model for anomaly detection on the Kuka dataset."
    )

    parser.add_argument("--data-dir", type=str, default="./data", help="Path to the dataset directory.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for the data loaders.")
    parser.add_argument("--latent-dim", type=int, default=4, help="Dimensionality of the autoencoder latent space.")
    parser.add_argument("--lr_ae", type=float, default=1e-3, help="Learning rate for the autoencoder optimizer.")
    parser.add_argument("--lr_d", type=float, default=1e-3, help="Learning rate for the discriminator optimizer.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=30,
        help="Number of epochs for the reconstruction-only pretraining phase (encoder + decoder), run before adversarial training.",
    )
    parser.add_argument("--quantile", type=float, default=0.85, help="Quantile used to compute the anomaly threshold.")
    parser.add_argument("--lambda-adv", type=float, default=0.001, help="Learning rate for the discriminator optimizer.")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top errors averaged in the reconstruction error function."
    )
    parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to apply preprocessing (normalization excluded).",
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
    parser.add_argument("--seed", type=int, default=123, help="Optional random seed for reproducibility.")

    return parser.parse_args()


def error_distribution_report(all_errors, all_labels, name="model"):
    y = all_labels.numpy()
    s = all_errors.numpy()

    normal_errors = all_errors[all_labels == 0]
    anomaly_errors = all_errors[all_labels == 1]

    roc_auc = roc_auc_score(y, s)
    pr_auc = average_precision_score(y, s)

    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    for label_name, e in [("Normal", normal_errors), ("Anomaly", anomaly_errors)]:
        qs = torch.quantile(e, torch.tensor([0.50, 0.75, 0.90, 0.95, 0.99]))

        print(f"\n{label_name} errors")
        print(f"mean: {e.mean().item():.6f}")
        print(f"q50 : {qs[0].item():.6f}")
        print(f"q75 : {qs[1].item():.6f}")
        print(f"q90 : {qs[2].item():.6f}")
        print(f"q95 : {qs[3].item():.6f}")
        print(f"q99 : {qs[4].item():.6f}")


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def build_dataloaders(data_dir, batch_size, seed=None, preprocess=True):
    splits = split_kuka_numpy(data_dir, seed=seed)

    if preprocess:
        preprocessor = KukaPreprocessor(
            normalization=True,
            remove_constant=False,
            corr_threshold=0.98,
            clip_percentiles=None,
        )
    else:
        preprocessor = KukaPreprocessor(
            normalization=True,
            remove_constant=False,
            corr_threshold=None,
            clip_percentiles=None,
        )

    x_train = preprocessor.fit_transform(splits.x_train)
    x_val = preprocessor.transform(splits.x_val)
    x_test = preprocessor.transform(splits.x_test)

    train_dataset = KukaDataset(x_train, splits.y_train)
    val_dataset = KukaDataset(x_val, splits.y_val)
    test_dataset = KukaDataset(x_test, splits.y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, train_loader, val_loader, test_loader, preprocessor



def err_fn(pred, x, top_k):
    return (pred - x).pow(2).topk(top_k, dim=1).values.mean(dim=1)


def pretrain_ae(
    model,
    train_loader,
    epochs,
    device,
    lr_ae=1e-3,
):
    """Reconstruction-only pretraining of the encoder + decoder.

    The discriminator is left untouched at this stage; this phase simply
    gives the autoencoder a good starting point before adversarial
    (AAE) training begins.
    """
    model.to(device)

    reconstruction_loss_fn = nn.MSELoss()

    optimizer_ae = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=lr_ae
    )

    history = {"reconstruction_loss": []}

    for epoch in range(epochs):
        model.train()

        epoch_recon_loss = 0.0

        for x, _ in train_loader:
            x = x.float().to(device)

            optimizer_ae.zero_grad()

            x_hat, _ = model(x)
            recon_loss = reconstruction_loss_fn(x_hat, x)

            recon_loss.backward()
            optimizer_ae.step()

            epoch_recon_loss += recon_loss.item()

        epoch_recon_loss /= len(train_loader)
        history["reconstruction_loss"].append(epoch_recon_loss)

        print(f"[Pretrain] Epoch [{epoch + 1}/{epochs}] Recon Loss: {epoch_recon_loss:.6f}")

    return history


def train_aae(
    model,
    train_loader,
    epochs,
    device,
    lr_ae=1e-3,
    lr_d=1e-3,
    lambda_adv=1.0
):
    model.to(device)

    reconstruction_loss_fn = nn.MSELoss()
    adversarial_loss_fn = nn.BCEWithLogitsLoss()

    optimizer_ae = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=lr_ae
    )

    optimizer_d = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=lr_d
    )

    optimizer_g = torch.optim.Adam(model.encoder.parameters(), lr=lr_ae)

    history = {
        "reconstruction_loss": [],
        "discriminator_loss": [],
        "generator_loss": []
    }

    for epoch in range(epochs):
        model.train()

        epoch_recon_loss = 0.0
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_z_mean = 0.0
        epoch_z_std = 0.0

        for x, _ in train_loader:

            x = x.float().to(device)

            batch_size = x.size(0)
            latent_dim = model.encoder(x).size(1)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)


            # 1. Reconstruction phase: train encoder + decoder
            optimizer_ae.zero_grad()

            x_hat, z_fake = model(x)
            recon_loss = reconstruction_loss_fn(x_hat, x)

            recon_loss.backward()
            optimizer_ae.step()

            # 2. Discriminator phase: train discriminator
            optimizer_d.zero_grad()

            z_real = torch.randn(batch_size, latent_dim).to(device)

            with torch.no_grad():
                z_fake = model.encoder(x)

            d_real_logits = model.discriminator(z_real)
            d_fake_logits = model.discriminator(z_fake)

            d_real_loss = adversarial_loss_fn(d_real_logits, real_labels)
            d_fake_loss = adversarial_loss_fn(d_fake_logits, fake_labels)

            d_loss = d_real_loss + d_fake_loss

            d_loss.backward()
            optimizer_d.step()


            # 3. Generator phase: train encoder to fool discriminator
            optimizer_g.zero_grad()

            _, z_fake = model(x)
            d_fake_logits = model.discriminator(z_fake)

            # Recompute reconstruction loss fresh here: the original recon_loss's
            # graph was already freed by the .backward() call in phase 1, so it
            # can't be reused in a second backward() call without retain_graph.
            #recon_loss_g = reconstruction_loss_fn(x_hat_g, x)

            # Encoder wants the discriminator to classify encoded z as real
            g_loss = adversarial_loss_fn(d_fake_logits, real_labels)

            adversarial_encoder_loss = lambda_adv * g_loss
            adversarial_encoder_loss.backward()
            optimizer_g.step()

            epoch_recon_loss += recon_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_z_mean += z_fake.detach().mean().item()
            epoch_z_std += z_fake.detach().std().item()

        epoch_z_mean /= len(train_loader)
        epoch_z_std /= len(train_loader)
        epoch_recon_loss /= len(train_loader)
        epoch_d_loss /= len(train_loader)
        epoch_g_loss /= len(train_loader)

        history["reconstruction_loss"].append(epoch_recon_loss)
        history["discriminator_loss"].append(epoch_d_loss)
        history["generator_loss"].append(epoch_g_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Recon Loss: {epoch_recon_loss:.6f} | "
            f"D Loss: {epoch_d_loss:.6f} | "
            f"G Loss: {epoch_g_loss:.6f} |"
            f"z mean: {epoch_z_mean} |"
            f"z std: {epoch_z_std}"
        )

    return history




def evaluate(model, test_loader, device, threshold, top_k):
    all_preds = []
    all_labels = []
    all_errors = []

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            recon = model(x)[0]

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

    train_dataset, train_loader, val_loader, test_loader, _ = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        preprocess=args.preprocess,
    )

    x, _ = train_dataset[0]
    input_dim = x.shape[0]

    aae = AAE(input_dim, args.latent_dim).to(device)

    if args.weights_in is not None:
        print(f"Loading pre-trained weights from {args.weights_in}, skipping training.")
        aae.load_state_dict(torch.load(args.weights_in, map_location=device))
    else:
        if args.pretrain_epochs > 0:
            print(f"Pretraining autoencoder (reconstruction only) for {args.pretrain_epochs} epochs...")
            pretrain_ae(aae, train_loader, epochs=args.pretrain_epochs, device=device, lr_ae=args.lr_ae)

        train_aae(aae, train_loader, epochs=args.epochs, device=device, lr_ae=args.lr_ae, lr_d=args.lr_d, lambda_adv=args.lambda_adv)
        output_dir = os.path.dirname(args.weights_out)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(aae.state_dict(), args.weights_out)

    t = compute_threshold(
        aae,
        val_loader,
        quantile=args.quantile,
        err_fn=lambda pred, x: err_fn(pred, x, args.top_k),
    )
    print(f"Threshold: {t:.6f}")

    all_preds, all_labels, all_errors = evaluate(aae, test_loader, device, t, args.top_k)
    error_distribution_report(all_errors, all_labels, name="current run")
    report_results(all_preds, all_labels)
    plot_reconstruction_errors(all_errors, all_labels, t, quantile=args.quantile)





if __name__ == "__main__":
    main()