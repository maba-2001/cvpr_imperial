"""Train the map-conditioned geometry flow.

    python -m cvpr_imperial.extract --n 20000        # build the geometry cache
    python -m cvpr_imperial.train_geometry --steps 8000 --batch 16

Batched across maps of different sizes via `geometry_model.map_tensors_batch`
(the PyG disjoint-union trick) -- message passing needs no padding at all,
only the flow's attention does (see `geometry_model._pad`).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import config as cfg
from .cmap import from_arrays
from .geometry_model import DIMS, GeometryFlow, RANKS, compute_scales
from .geometry_vae import load_vaes


class GeomDataset(Dataset):
    def __init__(self, files: list):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        d = np.load(self.files[i])
        m = from_arrays(d["alpha"], d["phi"], d["face_of_dart"])
        geom = {r: torch.from_numpy(d[k].reshape(len(d[k]), -1).astype(np.float32))
                for r, k in zip(RANKS, ("vertex_xyz", "edge_curve", "face_surface"))}
        # delta (how far the curve reaches beyond its chord) is the one scalar
        # not recoverable from the two endpoint vertices -- 0 for lines and
        # arcs up to a semicircle -- appended as an extra feature; see
        # geometry_model.DIMS and frame.py.
        geom["e"] = torch.cat([geom["e"], torch.from_numpy(d["edge_delta"][:, None])], dim=1)
        # edge_type/face_type are already one-hot in `m`'s own numbering
        # (extract.py reindexes them); MapEncoder wants the class index.
        # loop_is_outer is already a plain 0/1 class index, not one-hot.
        edge_type = d["edge_type"].argmax(1).astype(np.int64)
        face_type = d["face_type"].argmax(1).astype(np.int64)
        loop_is_outer = d["loop_is_outer"].astype(np.int64)
        return m, geom, edge_type, face_type, loop_is_outer


def collate(batch: list) -> tuple[list, list, list, list, list]:
    """No tensor stacking here -- GeometryFlow.loss batches variable-sized
    maps itself, via map_tensors_batch."""
    maps, geoms, edge_types, face_types, loop_is_outers = zip(*batch)
    return (list(maps), list(geoms), list(edge_types), list(face_types),
            list(loop_is_outers))


def make_scheduler(opt, kind: str, lr: float, steps: int, warmup_frac: float = 0.03):
    if kind == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=warmup_frac)
    warmup_steps = max(1, int(steps * warmup_frac))
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup_steps))


def evaluate(net: GeometryFlow, loader: DataLoader, device: str, n_batches: int = 10) -> float:
    net.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i, (maps, geoms, edge_types, face_types, loop_is_outers) in enumerate(loader):
            if i >= n_batches:
                break
            g = [{r: v.to(device) for r, v in geom.items()} for geom in geoms]
            total += float(net.loss(g, maps, edge_types, face_types, loop_is_outers))
            n += 1
    net.train()
    return total / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-faces", type=int, default=None,
                    help="restrict training/val to shapes with at most this many faces")
    ap.add_argument("--sched", choices=["onecycle", "constant"], default="onecycle",
                    help="constant = linear warmup then flat lr, no decay-to-zero")
    ap.add_argument("--ranks", default=None,
                    help="cells to generate, e.g. 'ef' or 'e' (wireframe); default: 'ef' "
                         "when stage 1 places vertices (cfg.VERTEX_MODE), else 'vef'")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    files = sorted(cfg.GEOM_CACHE.glob("*.npz"))
    if args.max_faces is not None:
        # face_type is one row per face (extract.py sizes it to n_faces), so
        # its length is the face count without building the full CMap.
        files = [f for f in files if len(np.load(f)["face_type"]) <= args.max_faces]
    split = int(0.95 * len(files))
    train_ds, val_ds = GeomDataset(files[:split]), GeomDataset(files[split:])
    print(f"geometry cache: {len(train_ds)} train / {len(val_ds)} val"
          + (f" (<= {args.max_faces} faces)" if args.max_faces else ""), flush=True)
    kw = dict(batch_size=args.batch, collate_fn=collate, num_workers=4, drop_last=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    ranks = tuple(args.ranks) if args.ranks else None
    net = GeometryFlow(vaes=load_vaes(cfg.RUN_DIR, DIMS, args.device),
                       ranks=ranks).to(args.device)
    print(f"generating ranks {net.ranks}"
          + (" (vertices given by stage 1)" if net.vertex_cond else ""), flush=True)
    scales = {r: s for r, s in compute_scales(files[:split], net.vaes, args.device).items()
              if r in net.ranks}
    print(f"per-rank scales: {scales}", flush=True)
    for r, s in scales.items():
        getattr(net, f"scale_{r}").fill_(s)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = make_scheduler(opt, args.sched, args.lr, args.steps)
    cfg.RUN_DIR.mkdir(parents=True, exist_ok=True)
    step, t0, run, best = 0, time.time(), 0.0, float("inf")
    while step < args.steps:
        for maps, geoms, edge_types, face_types, loop_is_outers in train_loader:
            g = [{r: v.to(args.device) for r, v in geom.items()} for geom in geoms]
            loss = net.loss(g, maps, edge_types, face_types, loop_is_outers)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run += float(loss)
            if step % 100 == 0:
                print(f"{step:6d} loss={run/100:.5f} ({(time.time()-t0)/step:.3f}s/step)",
                      flush=True)
                run = 0.0
            if step % 1000 == 0 or step == args.steps:
                v = evaluate(net, val_loader, args.device)
                print(f"{step:6d} val={v:.5f}", flush=True)
                if v < best:
                    best = v
                    torch.save(net.state_dict(), cfg.RUN_DIR / "geometry.pt")
            if step >= args.steps:
                break
    print(f"best val {best:.5f}", flush=True)


if __name__ == "__main__":
    main()
