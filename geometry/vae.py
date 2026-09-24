"""Per-cell VAEs for edge curves and face surfaces -- lends CLR-Wire's
curve-VAE idea (src/vae/vae_curve.py: encode a curve to a small Gaussian
latent, decode back, KL-regularize toward N(0, I)) to this codebase's
already-fixed-size per-cell geometry. CLR-Wire's encoder/decoder are 1D-conv
+ cross-attention stacks because a curve there is a variable-length sampled
polyline needing a learned downsampling schedule; here an edge/face is
already a fixed-size vector (`geometry_model.DIMS`), so a plain MLP
encoder/decoder is the direct equivalent, not a simplification of substance.

Vertices are left un-latent (raw xyz, flowed on directly): they're already
3 numbers, so a VAE would only add reconstruction error, not compression.

    python -m cvpr_imperial.scripts.train_geometry_vae --steps 4000
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .. import config as cfg


class CellVAE(nn.Module):
    """Diagonal-Gaussian VAE over one rank's fixed-size raw geometry vector."""

    def __init__(self, dim_in: int, latent_dim: int, hidden: int = cfg.VAE_HIDDEN):
        super().__init__()
        self.dim_in, self.latent_dim = dim_in, latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim_in),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x).chunk(2, dim=-1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, exp(logvar)) || N(0, I)), averaged over batch and latent dim
    -- the same closed-form term CLR-Wire's DiagonalGaussianDistribution.kl()
    computes."""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).mean()


VAE_LATENT = {"e": cfg.VAE_LATENT_E, "f": cfg.VAE_LATENT_F}


def build_vaes(dims: dict) -> dict[str, CellVAE]:
    """One CellVAE per rank in `VAE_LATENT` (edges, faces -- not vertices)."""
    return {r: CellVAE(dims[r], VAE_LATENT[r]) for r in VAE_LATENT}


def load_vaes(run_dir: Path, dims: dict, device: str) -> dict[str, CellVAE] | None:
    """Loads {run_dir}/{r}_vae.pt for every latent rank, frozen; None if any
    is missing (the flow then falls back to raw-geometry flow matching, see
    `GeometryFlow`) so this is a strict opt-in, not a silent partial one."""
    vaes = build_vaes(dims)
    for r, vae in vaes.items():
        path = run_dir / f"{r}_vae.pt"
        if not path.exists():
            return None
        vae.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        vae.to(device).eval()
        vae.requires_grad_(False)
    return vaes
