import torch
import torch.nn as nn

class LABE(nn.Module):
    """
    LABE: Lightweight Autoencoder module for inference and encoding/decoding.
    This module wraps a Generator (G) and Encoder (E) and provides forward methods.
    No training, logging, or saving logic is included.
    """
    def __init__(self, G, E, zsize=None, kl_weight=1.0, binary_reco_loss=False):
        super().__init__()
        self.G = G
        self.E = E
        self.zsize = zsize
        self.kl_weight = kl_weight
        self.binary_reco_loss = binary_reco_loss
        self.mse = nn.MSELoss(reduction='sum')

    def reparam_log(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def encode(self, x):
        """Encode input x using the encoder E. Returns (mu, logvar, z, quant_error)."""
        mu, logvar, ze, _, err_quant = self.E(x)
        z = mu
        return mu, logvar, z, err_quant

    def decode(self, z, ws=None):
        """Decode latent z using the generator G. Optionally pass shared weights ws."""
        return self.G(z, ws)

    def forward(self, x, ws=None):
        """Encode and decode input x. Returns reconstruction and optionally losses."""
        mu, logvar, z, err_quant = self.encode(x)
        xr = self.decode(z, ws)
        return xr, mu, logvar, z, err_quant

    def reconstruction_loss(self, x, xr, mu=None, logvar=None):
        """Compute reconstruction loss (MSE or BCE)."""
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        xr = xr.view(batch_size, -1)
        if self.binary_reco_loss:
            loss_reco = nn.functional.binary_cross_entropy(xr, x, reduction='sum') / batch_size
        else:
            loss_reco = self.mse(xr, x) / batch_size
        loss = loss_reco
        return loss, loss_reco, None
