"""Quick-look plots of sampled CMaps' *combinatorial* graph structure (1-
skeleton: vertices + edges via a force layout), as opposed to `viz_wireframe.py`
which plots the *embedded* (3D) geometry. Useful for seeing multi-edges,
self-loops (full circles) and bigon/short faces directly, independent of how
good the geometry model's placement is.

    python -m cvpr_imperial.scripts.viz_graph --n 6
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from scipy.spatial import ConvexHull

from .. import config as cfg
from ..topology.model import CodeTransformer, sample


def to_multigraph(m) -> nx.MultiGraph:
    """One edge per map edge (including parallel edges and self-loops)."""
    g = nx.MultiGraph()
    g.add_nodes_from(range(m.counts()[0]))
    lower = np.minimum(np.arange(m.n_darts), m.alpha)
    for d in range(m.n_darts):
        if lower[d] != d:
            continue
        v0, v1 = int(m.vertex_of_dart[d]), int(m.vertex_of_dart[m.alpha[d]])
        g.add_edge(v0, v1)
    return g


def face_blob(points: np.ndarray, color, alpha: float = 0.28):
    """A translucent patch enveloping `points`, used to show face membership
    without drawing more edge-like lines (a filled hull, not a graph edge)."""
    pts = np.unique(points, axis=0)
    centroid = pts.mean(axis=0)
    if len(pts) == 1:
        return plt.Circle(pts[0], radius=0.06, color=color, alpha=alpha, lw=0)
    if len(pts) == 2:
        d = pts[1] - pts[0]
        length = np.linalg.norm(d) + 1e-9
        perp = np.array([-d[1], d[0]]) / length
        r = 0.05
        corners = np.array([pts[0] + perp * r, pts[1] + perp * r,
                             pts[1] - perp * r, pts[0] - perp * r])
        return plt.Polygon(corners, closed=True, color=color, alpha=alpha, lw=0)
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]
    padded = centroid + (hull_pts - centroid) * 1.35
    return plt.Polygon(padded, closed=True, color=color, alpha=alpha, lw=0)


def plot_graph(ax, m, title: str):
    g = to_multigraph(m)
    pos = nx.spring_layout(g, seed=0)

    n_faces = m.counts()[2]
    cmap = plt.get_cmap("tab20" if n_faces > 10 else "tab10")

    # one blob per face, covering all vertices touched by any of its loops
    for face in range(n_faces):
        loops = np.nonzero(m.face_of_loop == face)[0]
        verts = set()
        for loop in loops:
            for d in m.loop_cycle(loop):
                verts.add(int(m.vertex_of_dart[d]))
        pts = np.array([pos[v] for v in verts])
        ax.add_patch(face_blob(pts, cmap(face % cmap.N)))

    # thin gray curved arcs for the underlying multigraph, drawn on top of blobs
    drawn = {}
    for u, v, k in g.edges(keys=True):
        n_parallel = g.number_of_edges(u, v)
        rad = 0.0 if n_parallel == 1 else 0.25 * (k - (n_parallel - 1) / 2)
        if u == v:
            rad = 0.3 + 0.2 * k
        ax.annotate("", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle="-", color="0.4", lw=1.0,
                                    connectionstyle=f"arc3,rad={rad}"))

    nx.draw_networkx_nodes(g, pos, ax=ax, node_color="0.9", node_size=160,
                            edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=7)

    handles = [plt.Polygon([[0, 0]], closed=True, color=cmap(f % cmap.N),
                            alpha=0.5, label=f"face {f}")
               for f in range(n_faces)]
    ax.legend(handles=handles, fontsize=6, loc="upper right", framealpha=0.6)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--faces", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(cfg.RUN_DIR / "graphs.png"))
    args = ap.parse_args()

    topo = CodeTransformer().to(args.device)
    topo.load_state_dict(torch.load(cfg.RUN_DIR / "topology.pt", weights_only=True))
    topo.eval()

    cond = (None, args.faces) if args.faces is not None else None
    samples = sample(topo, args.n, cond=cond, device=args.device)

    ncols = min(3, args.n)
    nrows = (args.n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, m in enumerate(s.m for s in samples):
        inv = m.invariants()
        V, E, F, L = m.counts()
        plot_graph(axes[i], m, f"V={V} E={E} F={F} genus={inv['genus']}")
    for ax in axes[len(samples):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
