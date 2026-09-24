"""Build the dart cache (phi, alpha, cell geometry) for any configured dataset.

MFCAD++ already has one, built by `src/imperial/dartbrep/extract_batch.py`
(which additionally attaches machining-feature labels this project does not
use, and does not seam-split -- see below). This is the same core extraction
-- `extract_darts.extract_map`, unchanged -- for datasets without that label
file, starting with Fusion360.

The shape is seam-split first (`normalize.normalize_shape`): a full-circle
hole boundary is otherwise one edge whose two darts share a single vertex
(chord length exactly 0), and a cylindrical or toroidal face is periodic in
its own parametrization. Splitting removes both cases at the source -- every
loop becomes simple -- rather than requiring every downstream piece (the
per-edge local frame, the loop-nesting check) to handle a chord of 0 or a
periodic domain. `extract.py` (geometry) must see the same split shape as
this (topology), so both call through this same normalization.

    python -m cvpr_imperial.build_dart_cache --dataset fusion360
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time

import numpy as np

from . import config as cfg

sys.path.insert(0, str(cfg.ROOT / "src/imperial/dartbrep"))


def _one(args: tuple[str, str]) -> tuple[str, str, int]:
    fid, dataset = args
    ds = cfg.DATASETS[dataset]
    out_path = ds["dart_cache"] / f"{fid}.npz"
    if out_path.exists():
        return fid, "skip", 0
    from extract_darts import DartExtractionError, extract_map

    from .normalize import read_step_normalized

    try:
        m = extract_map(read_step_normalized(ds["step_dir"] / f"{fid}{ds['step_ext']}"))
    except DartExtractionError as e:
        return fid, f"invalid: {e}", 0
    except Exception as e:                                        # noqa: BLE001
        return fid, f"error: {type(e).__name__}: {e}", 0
    tmp = out_path.with_name(f".{fid}.tmp.npz")
    np.savez_compressed(
        tmp,
        phi=m["phi"], alpha=m["alpha"], face_of_dart=m["face_of_dart"],
        vertex_of_dart=m["vertex_of_dart"], edge_of_dart=m["edge_of_dart"],
        loop_of_dart=m["loop_of_dart"], loop_is_outer=m["loop_is_outer"],
        points=m["points"], face_normal_of_dart=m["face_normal_of_dart"],
        vertex_xyz=m["vertex_xyz"], edge_length=m["edge_length"],
        edge_type=m["edge_type"], face_type=m["face_type"],
        face_area=m["face_area"], face_center=m["face_center"],
        counts=np.array([m["n_darts"], m["n_vertices"], m["n_edges"], m["n_faces"],
                         m["n_loops"], m["n_inner_loops"], m["euler_char"],
                         m["two_genus"]]),
    )
    tmp.rename(out_path)
    return fid, "ok", m["n_darts"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=cfg.DATASET, choices=list(cfg.DATASETS))
    ap.add_argument("--n", type=int, default=None, help="cap per split, for a quick run")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    args = ap.parse_args()

    ds = cfg.DATASETS[args.dataset]
    ds["dart_cache"].mkdir(parents=True, exist_ok=True)
    ids = []
    for split in ("train", "val", "test"):
        split_ids = (ds["root"] / f"{split}.txt").read_text().split()
        ids += split_ids[: args.n] if args.n else split_ids

    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(_one, [(fid, args.dataset) for fid in ids], chunksize=16)
    ok = [r for r in results if r[1] in ("ok", "skip")]
    bad = [r for r in results if r[1] not in ("ok", "skip")]
    dart_counts = [r[2] for r in results if r[1] == "ok"]
    print(f"{len(ok)}/{len(ids)} extracted in {time.time()-t0:.0f}s; {len(bad)} rejected")
    if dart_counts:
        print(f"darts: min={min(dart_counts)} max={max(dart_counts)} "
              f"mean={np.mean(dart_counts):.0f} median={np.median(dart_counts):.0f}")
    for fid, msg, _ in bad[:20]:
        print(f"  {fid}: {msg}")


if __name__ == "__main__":
    main()
