"""Evaluation: the validity ladder, exact novelty, and conditioning accuracy.

Three things distinguish this from how the field usually reports.

The validity ladder is explicit and tiered, so "valid" cannot quietly mean
"OCCT did not crash". Tier 0 (a closed orientable manifold map) and tier 1
(connected) hold by construction here, and are still measured, because a
number that is 100% by construction is a claim the reader can check.

Novelty and uniqueness are exact. Two generated shapes have the same topology
iff their canonical codes are equal (`code.code_key`), so memorisation is a set
lookup, not a quantised geometry hash.

Conditioning accuracy is exact too: the requested genus and face count are
compared against the invariants computed from the sample itself.

    python -m cvpr_imperial.evaluate --n 512 --geometry
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np
import torch

from . import code as C
from . import config as cfg
from .topology_model import CodeTransformer, sample


def train_code_set() -> set:
    """Canonical (isomorphism-invariant) codes of the training set -- built by
    `build_train_signatures.py`, separately from `codes_train.npz` (the
    `quick_encode`-based training *data*, which is not comparable to the
    `code_key` values generated maps are checked against here)."""
    path = cfg.RUN_DIR / "train_signatures.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python -m cvpr_imperial.build_train_signatures`")
    blob = np.load(path, allow_pickle=True)
    return {tuple(int(x) for x in k) for k in blob["keys"]}


def topology_report(samples: list, train_codes: set, target=None) -> dict:
    """Validity, novelty and conditioning for a batch of sampled
    `topology_model.TopoSample`s."""
    rep = Counter()
    codes = []
    for s in samples:
        chk = s.m.check()
        rep["tier0"] += chk["tier0_pass"]
        rep["tier1"] += chk["tier1_pass"]
        inv = chk["invariants"]
        codes.append(C.code_key(s.m, s.edge_type, s.face_type, s.loop_is_outer))
        if target is not None:
            rep["genus_match"] += inv["genus"] == target[0]
            rep["faces_match"] += inv["n_faces"] == target[1]
    n = max(len(samples), 1)
    uniq = set(codes)
    return {
        "n": len(samples),
        "tier0_valid": rep["tier0"] / n,
        "tier1_valid": rep["tier1"] / n,
        "unique": len(uniq) / n,
        "novel": sum(c not in train_codes for c in uniq) / max(len(uniq), 1),
        "genus_match": rep["genus_match"] / n if target else None,
        "faces_match": rep["faces_match"] / n if target else None,
    }


def geometry_report(samples: list, flow, device: str, chunk: int = 64) -> dict:
    """Tier 2 (embedding) and tier 3 (kernel) on realized samples. The flow
    is sampled in batches (see `GeometryFlow.sample_batch`), not one map at a
    time -- `realize()` still runs per-sample since it's a CPU routine with
    no batched form."""
    from .realize import realize

    rep = Counter()
    by_faces = defaultdict(list)
    for i in range(0, len(samples), chunk):
        part = samples[i:i + chunk]
        geoms = flow.sample_batch([s.m for s in part], [s.edge_type for s in part],
                                   [s.face_type for s in part],
                                   [s.loop_is_outer for s in part],
                                   [s.vertices for s in part] if flow.vertex_cond else None,
                                   device=device)
        for m, g in zip((s.m for s in part), geoms):
            r = realize(m, g)
            rep["sewn"] += r["sewn"]
            rep["kernel_valid"] += r["kernel_valid"]
            rep["nested"] += r.get("tier2", {}).get("loops_nested", False)
            rep["n"] += 1
            by_faces[m.counts()[2]].append(int(r["kernel_valid"]))
    n = max(rep["n"], 1)
    out = {"sewn": rep["sewn"] / n, "kernel_valid": rep["kernel_valid"] / n,
           "loops_nested": rep["nested"] / n, "min_k": {}}
    for k in (5, 10, 20, 30):
        vals = [v for kk, vs in by_faces.items() if kk >= k for v in vs]
        if vals:
            out["min_k"][k] = (float(np.mean(vals)), len(vals))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--genus", type=int)
    ap.add_argument("--faces", type=int)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--geometry", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = CodeTransformer().to(args.device)
    model.load_state_dict(torch.load(cfg.RUN_DIR / "topology.pt", weights_only=True))
    target = (args.genus, args.faces) if args.genus is not None else None
    samples = sample(model, args.n, cond=target, guidance=args.guidance,
                     temperature=args.temperature, device=args.device)

    rep = topology_report(samples, train_code_set(), target)
    print("topology")
    for k, v in rep.items():
        if v is not None:
            print(f"  {k:14s} {v:.3f}" if isinstance(v, float) else f"  {k:14s} {v}")

    if args.geometry:
        from .geometry_model import DIMS, GeometryFlow
        from .geometry_vae import load_vaes
        flow = GeometryFlow.load(cfg.RUN_DIR / "geometry.pt",
                                 load_vaes(cfg.RUN_DIR, DIMS, args.device), args.device)
        g = geometry_report(samples, flow, args.device)
        print("geometry")
        for k, v in g.items():
            print(f"  {k:14s} {v}" if k == "min_k" else f"  {k:14s} {v:.3f}")


if __name__ == "__main__":
    main()
