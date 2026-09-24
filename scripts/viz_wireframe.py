"""Quick-look wireframe PNGs of sampled shapes (straight edges, no curvature
fitting -- just vertex positions from the geometry flow joined per CMap edge).

    python -m cvpr_imperial.scripts.viz_wireframe --n 6
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .. import config as cfg
from ..geometry.frame import from_deviation
from ..geometry.flow import DIMS, GeometryFlow
from ..geometry.vae import load_vaes
from ..geometry.realize import realize
from ..topology.model import CodeTransformer, sample


def edge_curves(m, geom: dict) -> list[np.ndarray]:
    """Per-edge (N_CURVE_SAMPLES, 3) world-space polyline, reconstructed the
    same way realize.py does -- endpoints plus the flow's own chord
    deviations, at the generated bulge-beyond-chord scale -- so curved edges
    and multi-edges (same two vertices, different curves) render distinctly."""
    vpos = geom["v"].reshape(-1, 3).astype(np.float64)
    e_flat = geom["e"].astype(np.float64)
    interior = e_flat[:, :-1].reshape(-1, cfg.N_CURVE_SAMPLES - 2, 3)
    delta = np.clip(e_flat[:, -1], 0.0, None)
    lower = np.minimum(np.arange(m.n_darts), m.alpha)
    curves = []
    for e in range(interior.shape[0]):
        d = int(np.nonzero((m.edge_of_dart == e) & (lower == np.arange(m.n_darts)))[0][0])
        v0, v1 = int(m.vertex_of_dart[d]), int(m.vertex_of_dart[m.alpha[d]])
        p0, p1 = vpos[v0], vpos[v1]
        chord = float(np.linalg.norm(p1 - p0))
        D = max(chord + delta[e], 1e-6)
        dev = np.zeros((cfg.N_CURVE_SAMPLES, 3))
        dev[1:-1] = interior[e]
        curves.append(from_deviation(dev, p0, p1, D))
    return curves


def plot_wireframe(ax, m, geom: dict, title: str):
    verts = geom["v"].reshape(-1, 3)
    for pts in edge_curves(m, geom):
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="tab:blue", linewidth=1.2)
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], color="tab:red", s=12)
    ax.set_title(title, fontsize=9)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--faces", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(cfg.RUN_DIR / "wireframes.png"))
    ap.add_argument("--any", action="store_true",
                     help="skip the kernel_valid filter and plot whatever comes out")
    ap.add_argument("--pool", type=int, default=64,
                     help="how many samples to draw from before filtering to kernel_valid")
    args = ap.parse_args()

    topo = CodeTransformer().to(args.device)
    topo.load_state_dict(torch.load(cfg.RUN_DIR / "topology.pt", weights_only=True))
    topo.eval()
    flow = GeometryFlow.load(cfg.RUN_DIR / "geometry.pt",
                             load_vaes(cfg.RUN_DIR, DIMS, args.device), args.device)

    cond = (None, args.faces) if args.faces is not None else None
    pool = sample(topo, args.pool if not args.any else args.n, cond=cond, device=args.device)

    maps = [s.m for s in pool]
    geoms = flow.sample_batch(maps, [s.edge_type for s in pool], [s.face_type for s in pool],
                              [s.loop_is_outer for s in pool],
                              [s.vertices for s in pool] if flow.vertex_cond else None,
                              device=args.device)

    picked = []
    for m, g in zip(maps, geoms):
        if args.any:
            picked.append((m, g, True))
        else:
            r = realize(m, g)
            if r["kernel_valid"]:
                picked.append((m, g, True))
        if len(picked) >= args.n:
            break
    if not picked:
        print(f"no kernel_valid samples found in a pool of {args.pool}; "
              f"re-run with --any to plot invalid ones anyway")
        return
    if len(picked) < args.n:
        print(f"only found {len(picked)}/{args.n} kernel_valid samples in a pool of {args.pool}")

    ncols = min(3, len(picked))
    nrows = (len(picked) + ncols - 1) // ncols
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    for i, (m, g, valid) in enumerate(picked):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        inv = m.invariants()
        tag = "valid" if valid else "?"
        plot_wireframe(ax, m, g,
                        f"V={m.counts()[0]} F={inv['n_faces']} genus={inv['genus']} ({tag})")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
