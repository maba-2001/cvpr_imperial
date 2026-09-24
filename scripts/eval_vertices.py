"""Vertex-block evaluation with the topology held fixed.

Loss values can't compare the two vertex heads (cross-entropy vs a flow
MSE), so this measures what matters for CAD instead, on held-out shapes whose
ground-truth topology is fed as the prefix and only the vertex block is
sampled:

  err        mean |g x - x_gt| per vertex, minimised over the 48 signed axis
             permutations g -- the symmetries of the normalised unit-box
             frame. p(V | T) is multimodal (a box's topology fits its mirror
             images equally well), so plain distance to the one ground-truth
             instance would penalise perfectly valid samples.
  coinc@tol  of ground-truth vertex pairs sharing a coordinate exactly
             (same x, or y, or z), the fraction that still agree within tol
  axis@tol   of axis-aligned ground-truth edges, the fraction still
             axis-aligned within tol
  planar@tol of planar ground-truth faces with >= 4 corners, the fraction
             whose sampled corners are coplanar within tol

tol = 1e-5 asks for exact agreement; tol = one 10-bit bin (~1e-3) for
approximate. "gt_quantized" scores the ground truth snapped to the
quantisation grid: the ceiling for the quant head.

    CVPR_VERTEX_MODE=quant python -m cvpr_imperial.scripts.eval_vertices --n 256
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import torch

from ..topology import grammar as C
from .. import config as cfg
from ..data.dataset import CodeDataset, collate
from ..topology.model import CodeTransformer
from ..geometry.vertex_head import dequantize, quantize

TOLS = (1e-5, 1.0 / 2 ** cfg.VERTEX_BITS)
GT_EQ = 1e-7
BOX_SYM = [np.diag(s)[:, p] for p in itertools.permutations(range(3))
           for s in itertools.product((1, -1), repeat=3)]


@torch.no_grad()
def sample_vertex_block(model, batch: dict, temperature: float = 1.0) -> list[np.ndarray]:
    kinds, device = batch["kinds"], batch["kinds"].device
    coords = torch.zeros_like(batch["coords"])
    vpos = [torch.nonzero(k == C.VERTEX).squeeze(1) for k in kinds]
    for step in range(max(len(p) for p in vpos)):
        live = [b for b, p in enumerate(vpos) if step < len(p)]
        h = model.forward(kinds, batch["values"], batch["cond"], force_null=True,
                          coords=coords, slot_idx=batch["slot_idx"], slot_w=batch["slot_w"])
        t = torch.stack([vpos[b][step] for b in live])
        b = torch.tensor(live, device=device)
        coords[b, t] = model.vhead.sample(h[b, t].float(), temperature)
    return [coords[b, p].cpu().numpy() for b, p in enumerate(vpos)]


def _structure(kinds, values):
    tok = [(int(k), int(v)) for k, v in zip(kinds, values) if k != C.VERTEX]
    return C.decode(tok)


def metrics(x: np.ndarray, gt: np.ndarray, m, face_type) -> dict:
    out = {"err": min(float(np.linalg.norm(x @ g - gt, axis=1).mean()) for g in BOX_SYM)}
    pairs = [(i, j, a) for i, j in itertools.combinations(range(len(gt)), 2)
             for a in range(3) if abs(gt[i, a] - gt[j, a]) < GT_EQ]
    lower = np.minimum(np.arange(m.n_darts), m.alpha)
    ends = [(m.vertex_of_dart[d], m.vertex_of_dart[m.alpha[d]])
            for d in range(m.n_darts) if lower[d] == d]
    axis_edges = [(u, v) for u, v in ends if (np.abs(gt[u] - gt[v]) < GT_EQ).sum() == 2]
    planar = []
    for f in np.nonzero(face_type == 0)[0]:
        vs = np.unique(m.vertex_of_dart[m.face_of_dart == f])
        if len(vs) >= 4:
            planar.append(vs)
    for tol in TOLS:
        if pairs:
            out[f"coinc@{tol:.0e}"] = np.mean([abs(x[i, a] - x[j, a]) < tol for i, j, a in pairs])
        if axis_edges:
            out[f"axis@{tol:.0e}"] = np.mean(
                [(np.abs(x[u] - x[v]) < tol).sum() >= 2 for u, v in axis_edges])
        if planar:
            res = []
            for vs in planar:
                p = x[vs] - x[vs].mean(0)
                n = np.linalg.svd(p)[2][-1]
                res.append(np.abs(p @ n).max() < tol)
            out[f"planar@{tol:.0e}"] = np.mean(res)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--max-faces", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = CodeTransformer().to(args.device)
    model.load_state_dict(torch.load(cfg.RUN_DIR / "topology.pt", weights_only=True))
    model.eval()
    ds = CodeDataset("val", limit=args.n, max_faces=args.max_faces)
    rows = {"model": [], "gt_quantized": []}
    for i in range(0, len(ds), args.batch):
        items = [ds[j] for j in range(i, min(i + args.batch, len(ds)))]
        batch = {k: v.to(args.device) for k, v in collate(items).items()}
        for item, x in zip(items, sample_vertex_block(model, batch, args.temperature)):
            gt = item["verts"].numpy()
            m, _, face_type, _ = _structure(item["kinds"].numpy(), item["values"].numpy())
            rows["model"].append(metrics(x, gt, m, face_type))
            gq = dequantize(quantize(torch.from_numpy(gt), cfg.VERTEX_BITS), cfg.VERTEX_BITS)
            rows["gt_quantized"].append(metrics(gq.numpy(), gt, m, face_type))
    print(f"{cfg.VERTEX_MODE} head, {len(ds)} val shapes (<= {args.max_faces} faces)")
    for name, rs in rows.items():
        keys = sorted({k for r in rs for k in r})
        print(f"  {name:13s} " + "  ".join(
            f"{k}={np.mean([r[k] for r in rs if k in r]):.4f}" for k in keys))


if __name__ == "__main__":
    main()
