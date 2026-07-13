import argparse
from datetime import datetime
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.kuka_dataset import KukaDataset
from dataset.kuka_splits import split_kuka_numpy
from models.AAE import AAE
from processing.kuka_processing import KukaPreprocessor
from utils import plot_scores

from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity


from metrics import compute_pointwise_metrics
from train_ae import report_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate autoencoder model for anomaly detection on the Kuka dataset."
    )

    parser.add_argument("--data-dir", type=str, default="./data", help="Path to the dataset directory.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for the data loaders.")
    parser.add_argument("--latent-dim", type=int, default=4, help="Dimensionality of the autoencoder latent space.")
    parser.add_argument("--lr_ae", type=float, default=1e-3, help="Learning rate for the autoencoder optimizer.")
    parser.add_argument("--lr_d", type=float, default=5e-4, help="Learning rate for the discriminator optimizer.")
    parser.add_argument(
        "--lr_g", type=float, default=1e-4, help="Learning rate for the generator (ae encoder) optimizer."
    )
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--quantile", type=float, default=0.85, help="Quantile used to compute the anomaly threshold.")
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=0,
        help="Number of epochs for the reconstruction-only pretraining phase (encoder + decoder), run before adversarial training.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top errors averaged in the reconstruction error function."
    )
    parser.add_argument(
        "--weights-out",
        type=str,
        default=f"weights/aae/aae_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth",
        help="Path to save the trained AAE weights together with the fitted outlier estimator and fusion model.",
    )
    parser.add_argument(
        "--weights-in",
        type=str,
        default=None,
        help="Path to a checkpoint bundle (AAE weights + outlier estimator + fusion model). If provided, training is skipped.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Optional random seed for reproducibility.")

    return parser.parse_args()


def save_checkpoint(path, aae, input_dim, latent_dim, outlier_estimator, fusion_model):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    checkpoint = {
        "aae_state_dict": aae.state_dict(),
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "outlier_estimator": outlier_estimator,
        "fusion_model": fusion_model,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint (AAE + outlier estimator + fusion model) to {path}")


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    aae = AAE(checkpoint["input_dim"], checkpoint["latent_dim"]).to(device)
    aae.load_state_dict(checkpoint["aae_state_dict"])
    aae.eval()

    outlier_estimator = checkpoint["outlier_estimator"]
    fusion_model = checkpoint["fusion_model"]

    return aae, outlier_estimator, fusion_model


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def build_dataloaders(data_dir, batch_size, seed=None):
    splits = split_kuka_numpy(data_dir, seed=seed)

    preprocessor = KukaPreprocessor(
        normalization=True,
        remove_constant=False,
        corr_threshold=0.98,
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


def pretrain(
    model,
    train_loader,
    epochs,
    device,
    lr=1e-3,
):
    model.to(device)

    reconstruction_loss_fn = nn.MSELoss()
    optimizer_ae = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=lr)

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


def hinge(x, maximize=True):
    if maximize:
        return torch.relu(1 - x)
    else:
        return torch.relu(1 + x)


def train(model, train_loader, epochs, device, lr_ae=1e-3, lr_g=1e-3, lr_d=1e-3):
    model.to(device)

    reconstruction_loss_fn = nn.MSELoss()

    optimizer_AE = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=lr_ae)
    optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=lr_d)
    optimizer_G = torch.optim.Adam(model.encoder.parameters(), lr=lr_g)

    history = {"reconstruction_loss": [], "discriminator_loss": [], "generator_loss": []}

    for epoch in range(epochs):
        model.train()

        epoch_recon_loss = 0.0
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        for x, _ in train_loader:

            x = x.float().to(device)

            batch_size = x.size(0)
            latent_dim = model.encoder(x).size(1)

            # reconstruction phase: train encoder + decoder
            optimizer_AE.zero_grad()

            x_hat, z_fake = model(x)
            Lrec = reconstruction_loss_fn(x_hat, x)

            Lrec.backward()
            optimizer_AE.step()

            # discriminator phase: train discriminator
            optimizer_D.zero_grad()

            z_real = torch.randn(batch_size, latent_dim).to(device)
            z_real.requires_grad_(True)

            with torch.no_grad():
                z_fake = model.encoder(x)

            d_real_logits = model.discriminator(z_real)
            d_fake_logits = model.discriminator(z_fake)
            Ld = (hinge(d_real_logits, maximize=True) + hinge(d_fake_logits, maximize=False)).mean()

            # R1 penalty
            grad_real = torch.autograd.grad(
                outputs=d_real_logits.sum(),
                inputs=z_real,
                create_graph=True,
            )[0]

            r1_penalty = grad_real.pow(2).sum(dim=1).mean()
            r1_gamma = 10.0
            Ld_total = Ld + 0.5 * r1_gamma * r1_penalty

            Ld_total.backward()
            optimizer_D.step()

            # Generator phase: train encoder to fool discriminator
            optimizer_G.zero_grad()
            z_fake = model.encoder(x)
            d_fake_logits = model.discriminator(z_fake)

            Lg = hinge(d_fake_logits, maximize=True).mean()
            Lg.backward()
            optimizer_G.step()

            epoch_recon_loss += Lrec.item()
            epoch_d_loss += Ld_total.item()
            epoch_g_loss += Lg.item()

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
            f"G Loss: {epoch_g_loss:.6f}"
        )

    return history


@torch.no_grad()
def compute_scores(model, dataloader, recon_top_k, outlier_estimator):
    recon_scores = []
    outlier_scores = []
    latent_scores = []
    labels = []

    model.eval()
    for x, y in dataloader:
        recon, z = model(x)

        # reconstruction error score
        recon_score = (recon - x).pow(2).topk(recon_top_k, dim=1).values.mean(dim=1)

        # outlier score
        outlier_score = torch.from_numpy(-outlier_estimator.score_samples(z.cpu().numpy()))

        # latent score, derived from negative log-likelihood of the latent vectors
        latent_score = z.pow(2).sum(dim=1)

        recon_scores.append(recon_score.cpu())
        outlier_scores.append(outlier_score.cpu())
        latent_scores.append(latent_score.cpu())
        labels.append(y.cpu())

    recon_scores = torch.cat(recon_scores)
    outlier_scores = torch.cat(outlier_scores)
    latent_scores = torch.cat(latent_scores)
    labels = torch.cat(labels)

    return recon_scores, outlier_scores, latent_scores, labels


@torch.no_grad()
def extract_latent(model, dataloader):
    model.eval()

    z = []
    for x, _ in dataloader:
        x = x.float()
        latent = model.encoder(x)
        z.append(latent.cpu())

    return torch.cat(z).cpu().numpy()


@torch.no_grad()
def train_outlier_estimator(model, dataloader):
    estimator = KernelDensity(bandwidth=0.01, rtol=0.01, kernel="gaussian")
    # estimator = LocalOutlierFactor(novelty=True, n_neighbors=10)
    # estimator = GaussianMixture(n_components=1, covariance_type="full")
    # estimator = BayesianGaussianMixture(n_components=10, covariance_type="full", weight_concentration_prior=0.01)
    # estimator = EllipticEnvelope(contamination=0.001, support_fraction: None)    # to be used with `decision_function`

    z = extract_latent(model, dataloader)
    estimator.fit(z)
    return estimator


def train_fusion_model(labels, *scores):
    """
    Trains a Logistic Regression fusion model on an arbitrary number of score tensors.

    Args:
        labels (array-like or Tensor): The ground truth labels.
        *scores (Tensor): Variable number of 1D PyTorch tensors representing model scores.

    Returns:
        LogisticRegression: The fitted scikit-learn classifier.
    """
    if not scores:
        raise ValueError("You must provide at least one score tensor.")

    # validation checks
    base_length = len(scores[0])

    for i, score_tensor in enumerate(scores):
        if score_tensor.dim() != 1:
            raise ValueError(f"Score tensor at index {i} is not 1-dimensional (dim={score_tensor.dim()}).")

        if len(score_tensor) != base_length:
            raise ValueError(
                f"Length mismatch: Score tensor at index {i} has length {len(score_tensor)}, "
                f"but expected {base_length}."
            )

    if len(labels) != base_length:
        raise ValueError(f"Labels length ({len(labels)}) does not match the scores length ({base_length}).")

    x = torch.column_stack(scores)

    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # logistic regression as default fusion model
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    clf.fit(x, labels)

    return clf


def fuse_scores(clf, *scores):
    """
    Combines an arbitrary number of score tensors using a trained fusion classifier.

    Args:
        clf: The trained scikit-learn classifier (e.g., LogisticRegression).
        *scores (Tensor): Variable number of 1D PyTorch tensors representing model scores.

    Returns:
        Tensor: A 1D PyTorch tensor of the fused scores.
    """
    if not scores:
        raise ValueError("You must provide at least one score tensor.")

    base_length = len(scores[0])

    for i, score_tensor in enumerate(scores):
        if score_tensor.dim() != 1:
            raise ValueError(f"Score tensor at index {i} is not 1-dimensional (dim={score_tensor.dim()}).")

        if len(score_tensor) != base_length:
            raise ValueError(
                f"Length mismatch: Score tensor at index {i} has length {len(score_tensor)}, "
                f"but expected {base_length}."
            )

    x_test = torch.column_stack(scores)

    if isinstance(x_test, torch.Tensor):
        x_test_np = x_test.cpu().numpy()
    else:
        x_test_np = x_test

    fused_scores_np = clf.decision_function(x_test_np)
    return torch.from_numpy(fused_scores_np.copy())


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = get_device()

    train_dataset, train_loader, val_loader, test_loader, _ = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    x, _ = train_dataset[0]
    input_dim = x.shape[0]

    if args.weights_in is not None:
        print(f"Loading checkpoint from {args.weights_in}, skipping training.")
        aae, estimator, fusion_model = load_checkpoint(args.weights_in, device)
    else:
        aae = AAE(input_dim, args.latent_dim).to(device)

        if args.pretrain_epochs > 0:
            print(f"Pretraining autoencoder (reconstruction only) for {args.pretrain_epochs} epochs...")
            pretrain(aae, train_loader, epochs=args.pretrain_epochs, device=device, lr=args.lr_ae)
        train(aae, train_loader, epochs=args.epochs, device=device, lr_ae=args.lr_ae, lr_g=args.lr_g, lr_d=args.lr_d)

        print("Training outlier estimator...")
        estimator = train_outlier_estimator(aae, train_loader)

        # compute and fuse scores on validation set
        print("Training fusion model...")
        recon_scores, outlier_scores, latent_scores, labels = compute_scores(aae, val_loader, args.top_k, estimator)
        fusion_model = train_fusion_model(labels, recon_scores, outlier_scores, latent_scores)

        save_checkpoint(args.weights_out, aae, input_dim, args.latent_dim, estimator, fusion_model)

    # compute and evaluate scores on test set
    print("Testing the model")
    recon_scores, outlier_scores, latent_scores, labels = compute_scores(aae, test_loader, args.top_k, estimator)
    scores = fuse_scores(fusion_model, recon_scores, outlier_scores, latent_scores)

    # predict positive score as anomaly (threshold fixed to 0)
    t = 0.0
    preds = (scores > t).int()

    report_results(preds, labels, compute_pointwise_metrics(labels, preds, scores))
    plot_scores(scores, labels, threshold=t, title="Fusion Score")


if __name__ == "__main__":
    main()
