import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            self._block(input_dim, 72),
            self._block(72, 64),
            self._block(64, 56),
            self._block(56, 48),
            self._block(48, 32),
            self._block(32, 24),
            self._block(24, 16),
            nn.Linear(16, output_dim),
        )

    def _block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            self._block(input_dim, 16),
            self._block(16, 24),
            self._block(24, 32),
            self._block(32, 48),
            self._block(48, 56),
            self._block(56, 64),
            self._block(64, 72),
            nn.Linear(72, output_dim),
        )

    def _block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class AE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
