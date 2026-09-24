"""Pretrain the per-edge / per-face geometry VAEs that `train_geometry.py`'s
GeometryFlow then flow-matches on in latent space instead of raw geometry
(see `geometry_vae.py`). Cell-level, not map-level: every edge/face across
every cached shape is one training example, independent of which map it
came from.

    python -m cvpr_imperial.train_geometry_vae --steps 4000
    python -m cvpr_imperial.train_geometry --steps 8000   # then picks these up
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import config as cfg
from .geometry_model import DIMS, RANKS
from .geometry_vae import VAE_LATENT, CellVAE, kl_loss


class CellDataset(Dataset):
    """Every edge/face vector in the geometry cache, flattened across shapes."""

    def __init__(self, files: list, rank: str):
        key = {"e": "edge_curve", "f": "face_surface"}[rank]
        delta_key = "edge_delta" if rank == "e" else None
        rows = []
        for f in files:
            d = np.load(f)
            arr = d[key].reshape(len(d[key]), -1).astype(np.float32)
            if delta_key is not None:
                arr = np.concatenate([arr, d[delta_key][:, None].astype(np.float32)], axis=1)
            rows.append(arr)
        self.data = torch.from_numpy(np.concatenate(rows, axis=0))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int):
        return self.data[i]


def train_one(rank: str, files: list, args) -> None:
    ds = CellDataset(files, rank)
    split = int(0.95 * len(ds))
    train_ds = torch.utils.data.Subset(ds, range(split))
    val_ds = torch.utils.data.Subset(ds, range(split, len(ds)))
    print(f"{rank}: {len(train_ds)} train / {len(val_ds)} val cells", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=True)

    vae = CellVAE(DIMS[rank], VAE_LATENT[rank]).to(args.device)
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=0.01)
    step, t0, run, best = 0, time.time(), 0.0, float("inf")
    while step < args.steps:
        for x in train_loader:
            x = x.to(args.device)
            recon, mu, logvar = vae(x)
            loss = ((recon - x) ** 2).mean() + cfg.VAE_KL_WEIGHT * kl_loss(mu, logvar)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            run += float(loss)
            if step % 200 == 0:
                print(f"{rank} {step:6d} loss={run/200:.5f} ({(time.time()-t0)/step:.3f}s/step)",
                      flush=True)
                run = 0.0
            if step % 500 == 0 or step == args.steps:
                vae.eval()
                with torch.no_grad():
                    v = np.mean([float(((vae(x.to(args.device))[0] - x.to(args.device)) ** 2).mean())
                                 for x in val_loader])
                vae.train()
                print(f"{rank} {step:6d} val_recon={v:.5f}", flush=True)
                if v < best:
                    best = v
                    torch.save(vae.state_dict(), cfg.RUN_DIR / f"{rank}_vae.pt")
            if step >= args.steps:
                break
    print(f"{rank}: best val_recon {best:.5f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    files = sorted(cfg.GEOM_CACHE.glob("*.npz"))
    cfg.RUN_DIR.mkdir(parents=True, exist_ok=True)
    for rank in VAE_LATENT:
        train_one(rank, files, args)


if __name__ == "__main__":
    main()
