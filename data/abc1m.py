"""Loading Fusion360-style ABC-1M (github.com/datasets/ADSKAILab/ABC-1M) into
this project's pipeline.

The dataset ships each solid as a native OCCT ASCII BRep blob (a `bytes`
column in Hugging Face parquet shards) rather than STEP -- see `extract()`.
The blobs are missing one mandatory line: per the format's own BNF spec
(github.com/Open-Cascade-SAS/OCCT/wiki/brep_format), the `<shapes>` section
ends with a required `<shape final record>` giving the root shape's
orientation and location. Without it, `BRepTools::Read` doesn't fail
cleanly -- it throws deep inside subshape-reference resolution
(`NCollection_IndexedMap::FindKey`), which reads like a version/reader
incompatibility rather than one missing line. It isn't: appending it
(`+1 0`, i.e. "the most recently declared shape, forward, no location")
loads cleanly, verified against 100+ sampled rows and the dataset's own
worked example in the OCCT spec.

    python -m cvpr_imperial.data.abc1m --split train --download 4

Storage policy: a parquet shard is only ever a transient download. Once its
rows are unpacked to `step/{split}/{stem}.brep` and extracted into
`dart_cache/{stem}.npz`, the shard is deleted and its name recorded in
`{split}_shards.txt` -- the parquet is never the only copy of anything for
longer than one `build_dart_cache` call, so the abc/ folder never carries
both the raw shard and its unpacked contents at once.
"""

from __future__ import annotations

import argparse

from .. import config as cfg


def fix_brep(raw: bytes) -> bytes:
    """Append the missing mandatory root-shape reference line."""
    return raw.rstrip(b"\n") + b"\n\n+1 0\n"


def load_shape(raw: bytes):
    """ABC-1M's raw `bytes` blob -> a TopoDS_Shape (the solid, not the raw
    compound -- see below).

    Goes through a temp file rather than `BRepTools_ShapeSet.ReadFromString`
    -- the latter's own shape count bookkeeping doesn't reflect what a read
    populated (it stayed 0 even on a verified-successful read), where the
    plain `BRepTools.Read` function is proven reliable.

    The blob's top-level shape is a compound whose direct children are the
    real solid *plus* dozens-to-thousands of orphaned loose vertices/edges/
    wires/faces -- leftover intermediate construction pieces from the
    exporter, sitting as siblings rather than nested inside the solid
    (verified across many rows: always exactly 1 solid + 1 shell, sharing
    the solid's own TShape, + N unrelated loose sub-shapes). Passed through
    unfiltered, those loose edges show up in STEP previews and even survive
    seam-splitting as literal dangling edges -- one per face, since each
    face's construction left its own leftover pieces behind. Real geometry
    lives only in the solid, so that's what this returns.
    """
    import os
    import tempfile

    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import TopoDS_Shape

    fd, path = tempfile.mkstemp(suffix=".brep")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(fix_brep(raw))
        shape = TopoDS_Shape()
        ok = breptools.Read(shape, path, BRep_Builder())
    finally:
        os.remove(path)
    if not ok:
        raise ValueError("BRepTools::Read reported failure")

    solids = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        solids.append(exp.Current())
        exp.Next()
    if len(solids) != 1:
        raise ValueError(f"expected exactly 1 solid in compound, found {len(solids)}")
    return solids[0]


def iter_shard(path) -> "list[tuple[str, bytes]]":
    """(stem, fixed .brep bytes) for every valid row of one parquet shard."""
    import pyarrow.parquet as pq

    t = pq.read_table(str(path), columns=["stem", "bytes", "err"])
    return [(row["stem"], fix_brep(row["bytes"]))
            for row in t.to_pylist() if not row["err"]]


def _shard_list(split: str) -> list[str]:
    """All shard paths of a split in the HF repo, sorted, e.g.
    'train/brepgen-...-000003-000000-0.parquet'."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files("ADSKAILab/ABC-1M", repo_type="dataset")
    return sorted(f for f in files if f.startswith(f"{split}/"))


def _done_shards_path(split: str):
    return cfg.DATASETS["abc1m"]["root"] / f"{split}_shards.txt"


def done_shards(split: str) -> set:
    """Basenames of shards already fully processed (unpacked + extracted +
    deleted) for a split -- the dedup source of truth for `download_shards`,
    since a processed shard's parquet no longer exists locally to check."""
    p = _done_shards_path(split)
    return set(p.read_text().split()) if p.exists() else set()


def _mark_shard_done(split: str, name: str) -> None:
    with open(_done_shards_path(split), "a") as f:
        f.write(name + "\n")


def downloaded_shards(split: str) -> list:
    """Shard paths of a split currently present locally (not yet processed),
    sorted."""
    d = cfg.DATASETS["abc1m"]["parquet_dir"] / split
    return sorted(d.glob("*.parquet")) if d.exists() else []


def download_shards(split: str, n: int) -> list:
    """Download the first `n` not-yet-processed shards of a split via the
    `hf` CLI, return their local paths. Skips shards already downloaded
    *or* already processed (and thus deleted) -- `done_shards` covers the
    latter, since a processed shard's parquet is gone by the time this
    runs again."""
    import subprocess

    out_dir = cfg.DATASETS["abc1m"]["parquet_dir"]
    have = {p.name for p in downloaded_shards(split)} | done_shards(split)
    todo = [f for f in _shard_list(split) if f.split("/")[-1] not in have][:n]
    if todo:
        subprocess.run(
            ["hf", "download", "ADSKAILab/ABC-1M", "--repo-type", "dataset",
             "--include", *todo, "--local-dir", str(out_dir)],
            check=True)
    return [out_dir / f for f in todo]


def _extract_npz(m: dict) -> dict:
    import numpy as np

    return dict(
        phi=m["phi"], alpha=m["alpha"], face_of_dart=m["face_of_dart"],
        vertex_of_dart=m["vertex_of_dart"], edge_of_dart=m["edge_of_dart"],
        loop_of_dart=m["loop_of_dart"], loop_is_outer=m["loop_is_outer"],
        points=m["points"], face_normal_of_dart=m["face_normal_of_dart"],
        vertex_xyz=m["vertex_xyz"], edge_length=m["edge_length"],
        edge_type=m["edge_type"], face_type=m["face_type"],
        face_area=m["face_area"], face_center=m["face_center"],
        counts=np.array([m["n_darts"], m["n_vertices"], m["n_edges"],
                         m["n_faces"], m["n_loops"], m["n_inner_loops"],
                         m["euler_char"], m["two_genus"]]),
    )


def build_dart_cache(split: str) -> None:
    """Process every locally-downloaded shard of a split: unpack each row's
    fixed `.brep` bytes to `step/{split}/{stem}.brep`, extract (phi, alpha,
    cell geometry) into `dart_cache/{stem}.npz` (the same cache format
    `build_dart_cache.py` uses for the other datasets, so `data/dataset.py`/
    everything downstream treats 'abc1m' as just another `config.DATASETS`
    entry), then delete the shard and record it in `{split}_shards.txt`.
    Finally rewrites `{split}.txt` from the dart_cache's actual contents
    (not just this run's), since a full listing must survive across
    incremental calls that each only download a few shards.
    """
    import time

    import numpy as np

    from .extract_darts import DartExtractionError, extract_map
    from .normalize import normalize_shape

    ds = cfg.DATASETS["abc1m"]
    ds["dart_cache"].mkdir(parents=True, exist_ok=True)
    step_split_dir = ds["step_dir"] / split
    step_split_dir.mkdir(parents=True, exist_ok=True)
    shards = downloaded_shards(split)
    print(f"{split}: {len(shards)} shards to process", flush=True)

    n_ok, n_bad = 0, 0
    t0 = time.time()
    for shard in shards:
        for stem, raw in iter_shard(shard):
            step_path = step_split_dir / f"{stem}.brep"
            if not step_path.exists():
                step_path.write_bytes(raw)
            out_path = ds["dart_cache"] / f"{stem}.npz"
            if out_path.exists():
                continue
            try:
                shape = load_shape(raw)
                m = extract_map(normalize_shape(shape))
            except (DartExtractionError, ValueError):
                n_bad += 1
                continue
            except Exception:                                     # noqa: BLE001
                n_bad += 1
                continue
            tmp = out_path.with_name(f".{stem}.tmp.npz")
            np.savez_compressed(tmp, **_extract_npz(m))
            tmp.rename(out_path)
            n_ok += 1
        _mark_shard_done(split, shard.name)
        shard.unlink()

    # step_split_dir has this split's own stems (dart_cache is one flat
    # directory shared by all splits, so globbing it would fold other
    # splits' stems in); intersect with dart_cache so a row whose .brep was
    # unpacked but failed extraction doesn't end up in {split}.txt with no
    # matching .npz for data/dataset.py to load.
    split_stems = {p.stem for p in step_split_dir.glob("*.brep")}
    cached_stems = {p.stem for p in ds["dart_cache"].glob("*.npz")}
    stems = sorted(split_stems & cached_stems)
    (ds["root"] / f"{split}.txt").write_text("\n".join(stems) + "\n")
    print(f"{split}: {n_ok} newly extracted, {n_bad} rejected this run, "
          f"{len(stems)} total in {ds['root']}/{split}.txt "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--download", type=int, default=0,
                    help="number of additional shards to download before building")
    args = ap.parse_args()
    if args.download:
        for p in download_shards(args.split, args.download):
            print("downloaded", p)
    build_dart_cache(args.split)
