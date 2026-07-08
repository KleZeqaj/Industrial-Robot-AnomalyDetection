import torch
import torch.nn as nn
from models.AE import Encoder, Decoder


class Discriminator(nn.Module):
    """
    Discriminator in latent space.
    It tries to distinguish:
      - real z sampled from N(0, I)
      - fake z produced by Encoder(x)
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # logits, no sigmoid here
        )

    def forward(self, z):
        return self.layers(z)


class AAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.discriminator = Discriminator(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def reconstruct(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

