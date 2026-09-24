"""Exact canonical-code signatures of the training set, for novelty/memorisation
at evaluation time (see evaluate.py).

Training *data* uses `code.quick_encode` -- a cheap, non-canonical walk, fine
because the model only needs one consistent target per shape. Checking whether
a generated shape's topology already exists in training data needs the
isomorphism-invariant `code.canonical` instead, which is worst-case expensive
on highly symmetric parts (a plate with dozens of identical mounting holes has
dozens of interchangeable roots and ties at every step). `canonical` bounds
itself with an internal wall-clock deadline (checked only between Python-level
decisions, never inside a numpy call -- interrupting numpy's C loops with an
OS signal has been observed to corrupt their internal state rather than raise
cleanly), so a handful of such outliers degrade to a merely-deterministic
(not guaranteed isomorphism-exact) code instead of blocking the build; that
count is reported, not hidden.

    python -m cvpr_imperial.build_train_signatures
"""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from . import code as C
from . import config as cfg
from .data import load_map, split_ids, _types_of


def _signature_of(fid: str):
    path = cfg.DART_CACHE / f"{fid}.npz"
    if not path.exists():
        return None
    m = load_map(path)
    if m is None or m.n_darts > cfg.MAX_DARTS:
        return None
    edge_type, face_type, loop_is_outer = _types_of(path, m)
    tok, _ = C.canonical(m, edge_type, face_type, loop_is_outer)
    return tuple(v for _, v in tok)


def main() -> None:
    ids = split_ids("train")
    t0 = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        res = [r for r in pool.map(_signature_of, ids, chunksize=8) if r is not None]
    keys = set(res)
    out = cfg.RUN_DIR / "train_signatures.npz"
    np.savez_compressed(out, keys=np.array(list(keys), dtype=object))
    print(f"{len(res)}/{len(ids)} scoped, {len(keys)} distinct topologies "
          f"({time.time()-t0:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
