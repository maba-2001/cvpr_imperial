"""Datasets.

Stage 1 (topology) reads the existing dart cache produced by
`src/imperial/dartbrep/extract_batch.py` -- 59,665 MFCAD++ models are already
extracted -- keeps the closed, connected 2-manifold shells, and turns each into
its canonical token sequence. The canonical codes are cached, since
canonicalization costs ~10 ms per model.

Stage 2 (geometry) additionally needs sampled curves and surfaces, which the
dart cache does not carry; `extract.py` builds that.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..topology import grammar as C
from .. import config as cfg
from .cmap import from_arrays, reindex_cells


def load_map(path: Path):
    """Closed connected manifold CMap from a dart-cache file, else None."""
    d = np.load(path)
    a = d["alpha"]
    idx = np.arange(len(a))
    if not (np.array_equal(a[a], idx) and np.all(a != idx)):
        return None                      # open or non-manifold edge: out of scope
    m = from_arrays(a, d["phi"], d["face_of_dart"])
    return m if m.invariants()["b0"] == 1 else None


def split_ids(split: str) -> list[str]:
    return (cfg.DATA_DIR / f"{split}.txt").read_text().split()


def _types_of(path: Path, m):
    """(edge_type, face_type, loop_is_outer) reindexed from the dart cache's
    own (dartbrep) numbering onto `m`'s numbering -- see
    `cmap.reindex_cells`. edge_type/face_type are one-hot in the cache;
    stored/consumed everywhere else as the argmax class id. loop_is_outer is
    already a bool per loop."""
    d = np.load(path)
    edge_type = reindex_cells(d["edge_type"], m.edge_of_dart, d["edge_of_dart"]).argmax(1)
    face_type = reindex_cells(d["face_type"], m.face_of_dart, d["face_of_dart"]).argmax(1)
    loop_is_outer = reindex_cells(d["loop_is_outer"].astype(np.int64),
                                  m.loop_of_dart, d["loop_of_dart"])
    return edge_type.astype(np.int64), face_type.astype(np.int64), loop_is_outer


def _load(fid: str):
    """(CMap, edge_type, face_type, loop_is_outer, vertex_xyz | None), or None.

    Without a vertex block this reads the dart cache, which covers every
    extractable shape. With one it reads the geometry cache instead: that is
    where vertex positions live in the unit-box frame the geometry flow also
    uses (extract.py), with every per-cell array already reindexed to the
    map's own numbering, and its shapes are already scope-filtered.
    """
    if cfg.VERTEX_MODE is None:
        path = cfg.DART_CACHE / f"{fid}.npz"
        if not path.exists():
            return None
        m = load_map(path)
        return None if m is None else (m, *_types_of(path, m), None)
    path = cfg.GEOM_CACHE / f"{fid}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    m = from_arrays(d["alpha"], d["phi"], d["face_of_dart"])
    return (m, d["edge_type"].argmax(1).astype(np.int64),
            d["face_type"].argmax(1).astype(np.int64),
            d["loop_is_outer"].astype(np.int64), d["vertex_xyz"].astype(np.float32))


def _code_of(fid: str):
    """(kinds, values, cond, vertices, vertex_of_dart) for one model, or None
    if it is out of scope. vertices/vertex_of_dart are None without a vertex
    block; otherwise both are in the *decoded* map's numbering -- the one
    stage 1 reproduces at sampling time (σ-orbits numbered by first dart in
    walk order), so the vertex block order is a function of the prefix."""
    got = _load(fid)
    if got is None:
        return None
    m, edge_type, face_type, loop_is_outer, xyz = got
    if m.n_darts > cfg.MAX_DARTS:
        return None
    # Training data only needs *a* deterministic order per shape, not a
    # canonical (isomorphism-invariant) one -- quick_encode walks once, never
    # branches on ties, and is what makes this safe to run over the whole
    # dataset. `canonical` is reserved for evaluation (see evaluate.py).
    tok, order = C.quick_encode(m, edge_type, face_type, loop_is_outer)
    if len(tok) > cfg.MAX_TOKENS:
        return None
    if max(v for k, v in tok if k == C.LEN) > cfg.MAX_LOOP_LEN:
        return None
    if max(v for k, v in tok if k == C.NLOOPS) > cfg.MAX_LOOPS_PER_FACE:
        return None
    inv = m.invariants()
    kinds = [k for k, _ in tok]
    values = [v for _, v in tok]
    verts = vod = None
    if xyz is not None:
        dec = C.decode(tok)[0]
        vod = dec.vertex_of_dart
        verts = np.zeros((int(vod.max()) + 1, 3), dtype=np.float32)
        verts[vod] = xyz[m.vertex_of_dart[order]]     # decoded dart i == original dart order[i]
        kinds += [C.VERTEX] * len(verts)
        values += [0] * len(verts)
    return (np.array(kinds, dtype=np.int16), np.array(values, dtype=np.int16),
            np.array([inv["genus"], inv["n_faces"]], dtype=np.int64), verts,
            None if vod is None else vod.astype(np.int32))


def build_codes(split: str, workers: int = 0) -> Path:
    """Encode a split once and cache it."""
    import multiprocessing as mp

    out = cfg.RUN_DIR / f"codes_{split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    ids = split_ids(split)
    with mp.Pool(workers or mp.cpu_count()) as pool:
        res = pool.map(_code_of, ids, chunksize=16)
    keep = [r for r in res if r is not None]
    cols = {name: np.empty(len(keep), dtype=object)
            for name in ("kinds", "values", "verts", "vod")}
    for i, (k, v, _, verts, vod) in enumerate(keep):
        cols["kinds"][i], cols["values"][i] = k, v
        cols["verts"][i], cols["vod"][i] = verts, vod
    # quant and flow heads read the same tokens -- only whether a vertex
    # block exists at all changes the cache
    np.savez_compressed(out, cond=np.stack([r[2] for r in keep]),
                        vertex_block=np.array(cfg.VERTEX_MODE is not None), **cols)
    print(f"{split}: kept {len(keep)}/{len(ids)} -> {out}")
    return out


class CodeDataset(Dataset):
    """Token sequences + their conditioning invariants (+ vertex block)."""

    def __init__(self, split: str, limit: int | None = None, max_faces: int | None = None):
        cache = cfg.RUN_DIR / f"codes_{split}.npz"
        if not cache.exists():
            raise FileNotFoundError(
                f"{cache} missing -- run `python -m cvpr_imperial.data.dataset --build {split}`")
        blob = np.load(cache, allow_pickle=True)
        built = bool(blob["vertex_block"]) if "vertex_block" in blob else False
        if built != (cfg.VERTEX_MODE is not None):
            raise ValueError(f"{cache} was built {'with' if built else 'without'} a vertex "
                             f"block but cfg.VERTEX_MODE={cfg.VERTEX_MODE} -- rebuild it")
        cols = {k: blob[k] for k in ("kinds", "values", "cond", "verts", "vod")}
        keep = np.ones(len(cols["cond"]), dtype=bool)
        if max_faces is not None:
            keep &= cols["cond"][:, 1] <= max_faces
        idx = np.nonzero(keep)[0][:limit or None]
        for k, v in cols.items():
            setattr(self, k, v[idx])

    def __len__(self) -> int:
        return len(self.kinds)

    def __getitem__(self, i: int) -> dict:
        item = {
            "kinds": torch.from_numpy(self.kinds[i].astype(np.int64)),
            "values": torch.from_numpy(self.values[i].astype(np.int64)),
            "cond": torch.from_numpy(self.cond[i]),
        }
        if self.verts[i] is not None:
            item["verts"] = torch.from_numpy(self.verts[i])
            item["vod"] = torch.from_numpy(self.vod[i].astype(np.int64))
        return item


def collate(batch: list[dict]) -> dict:
    """Pad, and build the per-step supervision, pointer masks, and (with a
    vertex block) each vertex token's coordinates and structural slot."""
    from ..topology.model import targets_from_tokens, vertex_slots

    B = len(batch)
    n = max(len(b["kinds"]) for b in batch)
    out = {
        "kinds": torch.zeros(B, n, dtype=torch.long),
        "values": torch.zeros(B, n, dtype=torch.long),
        "mask": torch.zeros(B, n, dtype=torch.bool),
        "tgt": torch.full((B, n), -100, dtype=torch.long),
        "open_mask": torch.zeros(B, n, n, dtype=torch.bool),
        "cond": torch.stack([b["cond"] for b in batch]),
    }
    has_v = "verts" in batch[0]
    if has_v:
        out["coords"] = torch.zeros(B, n, 3)
        slot = []
    for i, b in enumerate(batch):
        k = len(b["kinds"])
        kinds, values = b["kinds"].numpy(), b["values"].numpy()
        tgt, om = targets_from_tokens(kinds, values)
        out["kinds"][i, :k] = b["kinds"]
        out["values"][i, :k] = b["values"]
        out["mask"][i, :k] = True
        # ALPHA targets index token positions; the OPEN class sits at index n
        tgt = np.where((kinds == C.ALPHA) & (tgt == k), n, tgt)
        out["tgt"][i, :k] = torch.from_numpy(tgt)
        out["open_mask"][i, :k, :k] = torch.from_numpy(om)
        if has_v:
            vpos = np.nonzero(kinds == C.VERTEX)[0]
            out["coords"][i, vpos] = b["verts"]
            for t, j, w in vertex_slots(kinds, b["vod"].numpy()):
                slot.append((i, t, j, w))
    if has_v:
        s = np.array(slot, dtype=np.float64).reshape(-1, 4)
        out["slot_idx"] = torch.from_numpy(s[:, :3].astype(np.int64))
        out["slot_w"] = torch.from_numpy(s[:, 3].astype(np.float32))
    return out


def _stats(n: int) -> None:
    from collections import Counter
    kept, reasons = 0, Counter()
    lens, faces, genus = [], [], []
    for fid in split_ids("train")[:n]:
        p = cfg.DART_CACHE / f"{fid}.npz"
        if not p.exists():
            reasons["no cache"] += 1
            continue
        m = load_map(p)
        if m is None:
            reasons["open / non-manifold / disconnected"] += 1
            continue
        edge_type, face_type, loop_is_outer = _types_of(p, m)
        tok, _ = C.quick_encode(m, edge_type, face_type, loop_is_outer)
        inv = m.invariants()
        lens.append(len(tok))
        faces.append(inv["n_faces"])
        genus.append(inv["genus"])
        kept += 1
    print(f"kept {kept}/{n}", dict(reasons))
    for name, arr in (("tokens", lens), ("faces", faces), ("genus", genus)):
        a = np.array(arr)
        print(f"  {name:7s} p50={np.percentile(a,50):6.0f} p99={np.percentile(a,99):6.0f} "
              f"max={a.max():6.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=int, default=0)
    ap.add_argument("--build", choices=["train", "val", "test"])
    a = ap.parse_args()
    if a.stats:
        _stats(a.stats)
    if a.build:
        build_codes(a.build)
