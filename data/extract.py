"""STEP -> combinatorial map + per-cell geometry.

The map comes from `extract_darts.py` (vendored, unchanged). This module adds
what generation needs: per-edge curve samples, sampled face surfaces, and
primitive/outer-loop labels.

Edges are stored as `N_CURVE_SAMPLES - 2` interior samples (endpoints dropped
-- they're exactly the vertex positions stage 2 already generates) plus one
scalar `delta`, in the chord-deviation representation `geometry/frame.py`
documents: resample at uniform arc length, store each sample's world-axis
deviation from the chord at the same arc fraction, divide by `D = c + delta`.
See `frame.py` for why (bounded even for closed edges, where chord-relative
scaling blows up) and the README's "Edge and face geometry" section for the
measured numbers.

    python -m cvpr_imperial.data.extract --n 20000 --workers 32
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time

import numpy as np

from .. import config as cfg
from .cmap import from_arrays, reindex_cells
from ..geometry.frame import to_deviation

def _sample_curve_arclength(edge, n: int, n_fine: int = 400) -> np.ndarray:
    """(n, 3) points along the edge, uniformly spaced in *arc length* (not
    the curve's own parametrization, which can bunch samples arbitrarily --
    e.g. near a spline knot -- and would make the same shape yield different
    targets depending on how the CAD kernel happened to parameterize it).
    A fine parameter-uniform pass gives a piecewise-linear arc-length proxy
    accurate enough to resample at `n`; `n_fine` is generous relative to the
    16 samples actually stored.
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.TopAbs import TopAbs_FORWARD

    c = BRepAdaptor_Curve(edge.Oriented(TopAbs_FORWARD))
    ts = np.linspace(c.FirstParameter(), c.LastParameter(), n_fine)
    fine = np.array([[c.Value(float(t)).X(), c.Value(float(t)).Y(), c.Value(float(t)).Z()]
                     for t in ts], dtype=np.float64)
    seg = np.linalg.norm(np.diff(fine, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    cum /= max(cum[-1], 1e-12)
    su = np.linspace(0.0, 1.0, n)
    out = np.stack([np.interp(su, cum, fine[:, a]) for a in range(3)], axis=1)
    return out.astype(np.float32)


def _sample_surface(face, n: int) -> np.ndarray:
    """(n, n, 3) grid over the face's UV bounds."""
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepTools import breptools

    s = BRepAdaptor_Surface(face)
    umin, umax, vmin, vmax = breptools.UVBounds(face)
    grid = np.empty((n, n, 3), dtype=np.float32)
    for i, u in enumerate(np.linspace(umin, umax, n)):
        for j, v in enumerate(np.linspace(vmin, vmax, n)):
            p = s.Value(float(u), float(v))
            grid[i, j] = (p.X(), p.Y(), p.Z())
    return grid


def extract(step_path: str) -> dict:
    """Map + geometry, with all coordinates in a shared unit-box frame."""
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    from .extract_darts import extract_map

    from .normalize import read_step_normalized

    shape = read_step_normalized(step_path)
    m = extract_map(shape)

    # Same scope as data.load_map (closed, connected 2-manifold): geometry for
    # anything outside it is never generated, so extracting it here would just
    # train the flow on geometry for shapes it will never be asked to fill in
    # -- and, concretely, `alpha` having a fixed point (an open/boundary edge,
    # one face-use only) makes that edge's "two endpoints" both sigma-orbit 0,
    # producing a bogus near-zero-length edge indistinguishable, without this
    # check, from a genuine closed loop.
    a, idx = m["alpha"], np.arange(len(m["alpha"]))
    if not (np.array_equal(a[a], idx) and np.all(a != idx)):
        raise ValueError(f"open or non-manifold edge: {m['n_boundary_edges']} boundary, "
                         f"{m['n_nonmanifold_edges']} non-manifold")
    if from_arrays(a, m["phi"], m["face_of_dart"]).invariants()["b0"] != 1:
        raise ValueError("disconnected (multiple shells)")

    # OCCT walks faces and edges in the same order extract_map did, so the
    # dense ids it returned index straight into these lists.
    faces, edges = [], {}
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        faces.append(topods.Face(exp.Current()))
        exp.Next()
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        e = topods.Edge(exp.Current())
        edges[hash(e.TShape())] = e
        exp.Next()

    # extract_map numbers edges by np.unique over the TShape hashes of the
    # darts' edges, so the same ranking recovers which OCCT edge each dense
    # id refers to -- no geometric matching needed.
    ekeys = np.array([np.int64(k) for k in edges], dtype=np.int64)
    if len(ekeys) != m["n_edges"]:
        raise ValueError(f"{len(ekeys)} edges in shape vs {m['n_edges']} in map")
    rank = np.argsort(ekeys)
    ordered = [list(edges.values())[i] for i in rank]

    curves = np.stack([_sample_curve_arclength(e, cfg.N_CURVE_SAMPLES) for e in ordered])
    # Convention shared with realize.py: an edge's samples run in the direction
    # of the *lower-indexed* of its two darts, so orientation is recoverable
    # from the map alone and needs no extra field.
    lo = np.minimum(np.arange(m["n_darts"]), m["alpha"])
    lower_dart_of_edge = np.zeros(m["n_edges"], dtype=np.int64)
    for eid in range(m["n_edges"]):
        d = int(np.nonzero((m["edge_of_dart"] == eid) & (lo == np.arange(m["n_darts"])))[0][0])
        lower_dart_of_edge[eid] = d
        if np.linalg.norm(curves[eid][0] - m["points"][d][0]) > \
           np.linalg.norm(curves[eid][-1] - m["points"][d][0]):
            curves[eid] = curves[eid][::-1]
    surfaces = np.stack([_sample_surface(f, cfg.N_SURF_GRID) for f in faces])

    # Reindex geometry from OCCT cell ids onto the map's own orbit ids
    # (vertices = sigma-orbits, edges = alpha-orbits), so geometry and topology
    # are addressed by the same object downstream.
    cm = from_arrays(m["alpha"], m["phi"], m["face_of_dart"])
    vertex_xyz = reindex_cells(m["vertex_xyz"], cm.vertex_of_dart, m["vertex_of_dart"])
    edge_curve = reindex_cells(curves, cm.edge_of_dart, m["edge_of_dart"])
    # edge_type/face_type are per-cell too (dartbrep's own edge/face
    # numbering), so they need the same reindex -- previously copied through
    # raw (see `out = {k: m[k] for k in (...)}` below), silently misaligned
    # with `cm`'s own numbering wherever `from_arrays` renumbers faces or
    # `orbits(alpha)` orders edges differently than dartbrep's extraction did.
    edge_type = reindex_cells(m["edge_type"], cm.edge_of_dart, m["edge_of_dart"])
    face_type = reindex_cells(m["face_type"], cm.face_of_dart, m["face_of_dart"])
    loop_is_outer = reindex_cells(m["loop_is_outer"].astype(np.int64),
                                  cm.loop_of_dart, m["loop_of_dart"])
    # each edge's two endpoint vertex ids, in the map's own (post-reindex)
    # vertex numbering, ordered to match the (already-fixed) curve direction
    # and scattered into the map's own (post-reindex) edge numbering.
    d0 = lower_dart_of_edge
    edge_ends = np.zeros((m["n_edges"], 2), dtype=np.int64)
    edge_ends[cm.edge_of_dart[d0]] = np.stack(
        [cm.vertex_of_dart[d0], cm.vertex_of_dart[m["alpha"][d0]]], axis=1)

    # Normalize by the extent of *all* sampled geometry, not just vertices:
    # a curved edge or a bulging surface patch can sit well outside the
    # vertex bounding box, and normalizing by vertices alone leaves those
    # points unnormalized -- rare (~0.2% of Fusion360), but the resulting
    # 100+ magnitude outliers are enough to destabilize training on them.
    all_pts = np.concatenate([vertex_xyz.reshape(-1, 3), edge_curve.reshape(-1, 3),
                              surfaces.reshape(-1, 3)])
    lo, hi = all_pts.min(0), all_pts.max(0)
    center, scale = (lo + hi) / 2, max(float((hi - lo).max()), 1e-6)
    vertex_xyz = (vertex_xyz - center) / scale
    edge_curve = (edge_curve - center) / scale
    face_surface = (surfaces - center) / scale

    # Each edge's interior samples as their world-axis deviation from the
    # chord, divided by D = chord length + delta (`frame.to_deviation`);
    # delta is the one scalar that needs predicting (0 for lines and arcs up
    # to a semicircle) -- see frame.py and the module docstring above.
    edge_delta = np.zeros(m["n_edges"], dtype=np.float32)
    edge_interior = np.zeros((m["n_edges"], cfg.N_CURVE_SAMPLES - 2, 3), dtype=np.float32)
    for eid in range(m["n_edges"]):
        p0, p1 = vertex_xyz[edge_ends[eid, 0]], vertex_xyz[edge_ends[eid, 1]]
        dev, D = to_deviation(edge_curve[eid].astype(np.float64), p0, p1)
        edge_interior[eid] = dev[1:-1]
        edge_delta[eid] = max(D - float(np.linalg.norm(p1 - p0)), 0.0)

    out = {k: m[k] for k in ("alpha", "phi", "face_of_dart")}
    out["vertex_xyz"] = vertex_xyz
    out["edge_curve"] = edge_interior
    out["edge_delta"] = edge_delta
    out["edge_type"] = edge_type
    out["face_type"] = face_type
    out["loop_is_outer"] = loop_is_outer
    out["face_surface"] = face_surface
    # The shared frame both stages generate in (vertices in the AR model,
    # edges/faces in the flow) -- kept so any cached shape maps back to
    # its original CAD coordinates.
    out["center"] = center.astype(np.float32)
    out["scale"] = np.float32(scale)
    return out


def _one(fid: str) -> tuple[str, str]:
    dst = cfg.GEOM_CACHE / f"{fid}.npz"
    if dst.exists():
        return fid, "skip"
    try:
        d = extract(str(cfg.STEP_DIR / f"{fid}{cfg.STEP_EXT}"))
    except Exception as e:                                # noqa: BLE001
        return fid, f"{type(e).__name__}: {e}"
    tmp = dst.with_name(f".{fid}.tmp.npz")
    np.savez_compressed(tmp, **d)
    tmp.rename(dst)
    return fid, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    args = ap.parse_args()
    cfg.GEOM_CACHE.mkdir(parents=True, exist_ok=True)
    ids = []
    for split in ("train", "val", "test"):
        ids += (cfg.DATA_DIR / f"{split}.txt").read_text().split()[: args.n]
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        res = pool.map(_one, ids, chunksize=8)
    bad = [r for r in res if r[1] not in ("ok", "skip")]
    print(f"{len(res) - len(bad)}/{len(res)} in {time.time()-t0:.0f}s; {len(bad)} failed")
    for fid, msg in bad[:10]:
        print(f"  {fid}: {msg}")


if __name__ == "__main__":
    main()
