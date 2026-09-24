"""
Extract the oriented combinatorial map (dart set + permutations) of a B-rep
shell from a STEP file via OCCT.

A dart is a coedge: one traversal of an edge by one face loop, i.e. the pair
(edge, traversal orientation). We extract two permutations over the dart set:

  phi   ("next"): successor dart along its face loop (from ordered wire
        traversal). Orbits of phi = loops.
  alpha ("radial next"): successor face-use around the shared edge, in
        Weiler's radial-edge sense (cyclic order by dihedral angle around
        the edge's tangent). Orbits of alpha = edges.

On a closed 2-manifold edge (exactly 2 face-uses) this reduces to the plain
mate-swap involution used previously -- a 2-cycle *is* an involution, so
nothing changes for the common case. It generalizes cleanly beyond that:

  - a free/open-boundary edge (1 face-use) gives alpha[d] = d, a fixed
    point, instead of being rejected as invalid;
  - a non-manifold edge (3+ face-uses, e.g. sheet-metal/lattice/assembly
    junctions) gives a single k-cycle ordering the face-uses angularly
    around the edge, instead of raising "duplicate dart key".

This is why dart identity no longer needs a uniqueness key: every wire
traversal produces a new dart unconditionally (occurrence-based identity),
and edges are grouped by TShape identity afterward regardless of how many
face-uses they have.

sigma = phi o alpha (derived, not stored) rotates darts around their origin
vertex on a 2-manifold edge, by the same argument as before. For radial
(3+) groups, whether sigma's rotation-around-a-vertex property still holds
depends on the radial ordering's rotational sense matching phi's -- this is
NOT independently re-derived/verified here, so sigma-vertex consistency is
recorded as a diagnostic (sigma_vertex_violations), not enforced as a hard
validity requirement. It was relaxed for the 2-manifold case too, since a
single OCC vertex can legitimately be a "pinch point" touched by two
topologically disjoint local neighborhoods (this alone was silently
rejecting a chunk of otherwise-valid MFCAD++ models under the previous
strict check) -- vertex_of_dart is built directly from OCC vertex identity,
not from sigma-orbit counts, so relaxing this check does not affect the
correctness of any downstream embedding.

Per dart we also record which face and which loop it belongs to, whether its
loop is the outer wire of the face, its origin vertex id, and simple oriented
geometry (start point, midpoint, end point of the oriented edge).

Validity checks performed per model (model rejected on any failure):
  - phi is a bijection with no fixed points
  - alpha is a bijection (true by construction: edge-groups partition the
    dart set and each group's local cyclic assignment is itself a
    bijection on that group -- kept as a cheap regression guard)
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepTools import BRepTools_WireExplorer, breptools
from OCC.Core.GeomAbs import (
    GeomAbs_Circle,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Line,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.GProp import GProp_GProps
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_REVERSED,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopoDS import topods

# surface/curve type one-hots: fixed order, "other" bucket last so unrecognized
# GeomAbs types (e.g. B-spline/Bezier surfaces, offset curves) degrade to a
# single shared bucket rather than raising.
_SURFACE_TYPES = [GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus]
_CURVE_TYPES = [GeomAbs_Line, GeomAbs_Circle]


def _surface_type_onehot(face) -> np.ndarray:
    t = BRepAdaptor_Surface(face).GetType()
    vec = np.zeros(len(_SURFACE_TYPES) + 1, dtype=np.float32)
    for i, st in enumerate(_SURFACE_TYPES):
        if t == st:
            vec[i] = 1.0
            return vec
    vec[-1] = 1.0  # other
    return vec


def _face_area_and_center(face):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    c = props.CentreOfMass()
    return props.Mass(), (c.X(), c.Y(), c.Z())


def _curve_type_onehot(edge) -> np.ndarray:
    t = BRepAdaptor_Curve(edge).GetType()
    vec = np.zeros(len(_CURVE_TYPES) + 1, dtype=np.float32)
    for i, ct in enumerate(_CURVE_TYPES):
        if t == ct:
            vec[i] = 1.0
            return vec
    vec[-1] = 1.0  # other
    return vec


def _edge_length(edge) -> float:
    props = GProp_GProps()
    brepgprop.LinearProperties(edge, props)
    return props.Mass()


def _edge_tangent_canonical(edge):
    """Tangent at the edge's midpoint, in the edge's own canonical
    (forced-FORWARD) orientation -- one shared reference direction for
    every face-use of this edge, independent of which occurrence's
    orientation flag we're currently processing. None on any degeneracy."""
    canonical = edge.Oriented(TopAbs_FORWARD)
    curve = BRepAdaptor_Curve(canonical)
    um = 0.5 * (curve.FirstParameter() + curve.LastParameter())
    pnt, d1 = gp_Pnt(), gp_Vec()
    curve.D1(um, pnt, d1)
    mag = d1.Magnitude()
    if mag < 1e-9:
        return None
    return (d1.X() / mag, d1.Y() / mag, d1.Z() / mag)


def _face_normal_at_edge(face, edge):
    """Outward face normal where this occurrence's edge meets the face,
    sampled at the edge's midpoint via the face's own UV parametrization.
    None on any degeneracy (missing pcurve, undefined normal)."""
    res = BRep_Tool.CurveOnSurface(edge, face)
    if res is None or res[0] is None:
        return None
    curve2d, u0, u1 = res
    uv = curve2d.Value(0.5 * (u0 + u1))
    surf = BRep_Tool.Surface(face)
    props = GeomLProp_SLProps(surf, uv.X(), uv.Y(), 1, 1e-6)
    if not props.IsNormalDefined():
        return None
    n = props.Normal()
    nx, ny, nz = n.X(), n.Y(), n.Z()
    if face.Orientation() == TopAbs_REVERSED:
        nx, ny, nz = -nx, -ny, -nz
    return (nx, ny, nz)


def _radial_order(group: list[int], darts: list[dict], tangent) -> list[int]:
    """Cyclic order of a shared edge's face-uses by dihedral angle around
    the edge's tangent (Weiler radial-edge structure). Falls back to plain
    encounter order on any degeneracy (missing tangent/normal, a normal
    parallel to the tangent) rather than raising -- a radial ordering is
    a refinement for message-passing locality, not a validity requirement,
    so a degenerate fallback is always structurally safe (still a valid
    permutation, just not angle-sorted)."""
    if tangent is None:
        return list(group)
    t = np.array(tangent, dtype=np.float64)
    normals = [darts[idx]["face_normal"] for idx in group]
    if any(fn is None for fn in normals):
        return list(group)
    proj = []
    for fn in normals:
        nvec = np.array(fn, dtype=np.float64)
        p = nvec - np.dot(nvec, t) * t
        pn = np.linalg.norm(p)
        if pn < 1e-9:
            return list(group)  # normal ~parallel to tangent: degenerate
        proj.append(p / pn)
    ref = proj[0]
    angles = [0.0]
    for p in proj[1:]:
        cos_a = float(np.clip(np.dot(ref, p), -1.0, 1.0))
        sin_a = float(np.dot(np.cross(ref, p), t))
        angles.append(math.atan2(sin_a, cos_a))
    return [group[i] for i in np.argsort(angles)]


class DartExtractionError(Exception):
    pass


def read_step(path: str):
    reader = STEPControl_Reader()
    if reader.ReadFile(path) != IFSelect_RetDone:
        raise DartExtractionError(f"STEP read failed: {path}")
    reader.TransferRoots()
    return reader.OneShape()


def _edge_key(edge) -> int:
    # TShape identity ignoring orientation/location: stable id for the edge.
    return hash(edge.TShape())


def _vertex_key(vertex) -> int:
    return hash(vertex.TShape())


def _oriented_endpoints(edge):
    """(first_vertex, last_vertex) of the edge respecting its orientation."""
    v_first = topexp.FirstVertex(edge, True)  # CumOri=True: respects edge orientation
    v_last = topexp.LastVertex(edge, True)
    return v_first, v_last


def _edge_points(edge):
    """Start / mid / end 3D points of the *oriented* edge."""
    curve = BRepAdaptor_Curve(edge)
    u0, u1 = curve.FirstParameter(), curve.LastParameter()
    if edge.Orientation() == TopAbs_REVERSED:
        params = (u1, 0.5 * (u0 + u1), u0)
    else:
        params = (u0, 0.5 * (u0 + u1), u1)
    pts = []
    for u in params:
        p = curve.Value(u)
        pts.append((p.X(), p.Y(), p.Z()))
    return pts  # [start, mid, end]


def extract_map(shape) -> dict:
    """Extract dart set and (phi, alpha) permutations from a shape.

    Returns a dict of numpy arrays; raises DartExtractionError on any
    violation of the (now much smaller) set of hard invariants.
    """
    darts = []           # per-dart records
    edge_key_to_group = defaultdict(list)  # edge_key -> [dart_idx, ...] in encounter order
    edge_key_to_tangent = {}
    loops = []           # (face_idx, is_outer)
    face_type_list = []  # one surface-type one-hot per face_idx
    face_area_list = []  # one area per face_idx
    face_center_list = []  # one centroid (x,y,z) per face_idx
    vertex_key_to_point = {}   # vertex_key -> (x, y, z), populated lazily
    edge_key_to_feat = {}      # edge_key -> (length, curve_type_onehot), populated lazily

    face_idx = -1
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        face_idx += 1
        outer_wire = breptools.OuterWire(face)
        face_type_list.append(_surface_type_onehot(face))
        area, center = _face_area_and_center(face)
        face_area_list.append(area)
        face_center_list.append(center)

        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_explorer.More():
            wire = topods.Wire(wire_explorer.Current())
            loop_idx = len(loops)
            loops.append((face_idx, wire.IsSame(outer_wire)))

            loop_darts = []
            wexp = BRepTools_WireExplorer(wire, face)
            while wexp.More():
                edge = topods.Edge(wexp.Current())  # oriented as traversed in this loop
                ek = _edge_key(edge)
                fwd = edge.Orientation() == TopAbs_FORWARD
                if edge.Orientation() not in (TopAbs_FORWARD, TopAbs_REVERSED):
                    raise DartExtractionError("edge with INTERNAL/EXTERNAL orientation in wire")
                v_first, v_last = _oriented_endpoints(edge)
                origin_key = _vertex_key(v_first)
                if origin_key not in vertex_key_to_point:
                    p = BRep_Tool.Pnt(v_first)
                    vertex_key_to_point[origin_key] = (p.X(), p.Y(), p.Z())
                if ek not in edge_key_to_feat:
                    edge_key_to_feat[ek] = (_edge_length(edge), _curve_type_onehot(edge))
                if ek not in edge_key_to_tangent:
                    edge_key_to_tangent[ek] = _edge_tangent_canonical(edge)
                dart_idx = len(darts)
                darts.append(
                    {
                        "edge_key": ek,
                        "forward": fwd,
                        "loop": loop_idx,
                        "face": face_idx,
                        "origin_vertex": origin_key,
                        "dest_vertex": _vertex_key(v_last),
                        "points": _edge_points(edge),
                        "face_normal": _face_normal_at_edge(face, edge),
                    }
                )
                edge_key_to_group[ek].append(dart_idx)
                loop_darts.append(dart_idx)
                wexp.Next()

            if len(loop_darts) == 0:
                raise DartExtractionError("empty wire")
            for i, d in enumerate(loop_darts):
                darts[d]["next"] = loop_darts[(i + 1) % len(loop_darts)]
            wire_explorer.Next()
        face_explorer.Next()

    n = len(darts)
    if n == 0:
        raise DartExtractionError("no darts")

    # alpha: radial-next face-use around the shared edge (Weiler radial-edge
    # structure). Group size 1 -> fixed point (open boundary). Group size 2
    # -> plain mate swap (a 2-cycle is an involution, identical to the old
    # behavior). Group size >=3 -> angularly-sorted k-cycle.
    alpha = np.empty(n, dtype=np.int64)
    for ek, group in edge_key_to_group.items():
        k = len(group)
        if k == 1:
            alpha[group[0]] = group[0]
        elif k == 2:
            alpha[group[0]], alpha[group[1]] = group[1], group[0]
        else:
            order = _radial_order(group, darts, edge_key_to_tangent.get(ek))
            for i in range(k):
                alpha[order[i]] = order[(i + 1) % k]
    if len(np.unique(alpha)) != n:
        raise DartExtractionError("alpha (radial permutation) is not a bijection")

    phi = np.array([d["next"] for d in darts], dtype=np.int64)
    if len(np.unique(phi)) != n:
        raise DartExtractionError("phi is not a bijection")

    edge_keys = set(edge_key_to_group.keys())

    # sigma = phi o alpha rotates darts around their origin vertex on a
    # 2-manifold edge; recorded as a diagnostic rather than enforced, since
    # (a) a single OCC vertex can legitimately be a "pinch point" spanning
    # multiple sigma-orbits even with every edge manifold, and (b) this
    # property is not independently re-verified for radial (3+) groups here.
    # vertex_of_dart below is built directly from OCC vertex identity, not
    # from sigma-orbit structure, so it is unaffected either way.
    sigma = phi[alpha]
    origin = np.array([d["origin_vertex"] for d in darts], dtype=np.int64)
    sigma_vertex_violations = int(np.sum(origin[sigma] != origin))

    faces = np.array([d["face"] for d in darts], dtype=np.int64)
    loop_ids = np.array([d["loop"] for d in darts], dtype=np.int64)
    points = np.array([d["points"] for d in darts], dtype=np.float32)  # (n, 3, 3)
    # per-dart local surface normal (already computed for the radial sort
    # above, exported here for equivariant/Vector-Neurons feature use);
    # degenerate cases (missing pcurve/undefined normal) fall back to zero.
    face_normal_of_dart = np.array(
        [d["face_normal"] if d["face_normal"] is not None else (0.0, 0.0, 0.0) for d in darts],
        dtype=np.float32,
    )  # (n, 3)

    # dense 0..n_vertices-1 / 0..n_edges-1 ids per dart, for orbit-level (vertex =
    # sigma-orbit, edge = alpha-orbit) pooling -- alongside face_of_dart, these are
    # the only bookkeeping a multi-rank message-passing scheme needs; no separate
    # incidence/boundary matrices to build or keep consistent.
    unique_vertex_keys, vertex_of_dart = np.unique(origin, return_inverse=True)
    edge_key_arr = np.array([d["edge_key"] for d in darts], dtype=np.int64)
    unique_edge_keys, edge_of_dart = np.unique(edge_key_arr, return_inverse=True)
    n_vertices = len(unique_vertex_keys)

    # per-cell real geometric features, dense-indexed to match vertex_of_dart /
    # edge_of_dart / face_of_dart exactly (same order np.unique produced above).
    vertex_xyz = np.array(
        [vertex_key_to_point[k] for k in unique_vertex_keys], dtype=np.float32
    )  # (n_vertices, 3)
    edge_length = np.array(
        [edge_key_to_feat[k][0] for k in unique_edge_keys], dtype=np.float32
    )  # (n_edges,)
    edge_type = np.array(
        [edge_key_to_feat[k][1] for k in unique_edge_keys], dtype=np.float32
    )  # (n_edges, 3)
    face_type = np.array(face_type_list, dtype=np.float32)  # (n_faces, 6)
    face_area = np.array(face_area_list, dtype=np.float32)  # (n_faces,)
    face_center = np.array(face_center_list, dtype=np.float32)  # (n_faces, 3)

    n_faces = faces.max() + 1
    n_loops = len(loops)
    n_edges = len(edge_keys)
    n_nonmanifold_edges = sum(1 for g in edge_key_to_group.values() if len(g) > 2)
    n_boundary_edges = sum(1 for g in edge_key_to_group.values() if len(g) == 1)
    # Euler characteristic with multiply-connected faces:
    # chi = V - E + F - H, H = inner loops
    n_inner = sum(1 for _, outer in loops if not outer)
    chi = n_vertices - n_edges + n_faces - n_inner
    genus2 = 2 - chi  # 2g for a single closed shell

    return {
        "phi": phi,
        "alpha": alpha,
        "face_of_dart": faces,
        "vertex_of_dart": vertex_of_dart.astype(np.int64),
        "edge_of_dart": edge_of_dart.astype(np.int64),
        "loop_of_dart": loop_ids,
        "loop_is_outer": np.array([outer for _, outer in loops], dtype=bool),
        "points": points,  # oriented start/mid/end per dart
        "face_normal_of_dart": face_normal_of_dart,  # (n_darts, 3) local surface normal, raw
        "vertex_xyz": vertex_xyz,        # (n_vertices, 3) raw, un-normalized
        "edge_length": edge_length,      # (n_edges,) raw arc length
        "edge_type": edge_type,          # (n_edges, 3) [line, circle, other]
        "face_type": face_type,          # (n_faces, 6) [plane, cyl, cone, sphere, torus, other]
        "face_area": face_area,          # (n_faces,) raw
        "face_center": face_center,      # (n_faces, 3) surface centroid, raw un-normalized
        "n_darts": n,
        "n_vertices": n_vertices,
        "n_edges": n_edges,
        "n_faces": int(n_faces),
        "n_loops": n_loops,
        "n_inner_loops": n_inner,
        "n_nonmanifold_edges": n_nonmanifold_edges,
        "n_boundary_edges": n_boundary_edges,
        "sigma_vertex_violations": sigma_vertex_violations,
        "euler_char": int(chi),
        "two_genus": int(genus2),
    }


def _count_orbits(perm: np.ndarray) -> int:
    n = len(perm)
    seen = np.zeros(n, dtype=bool)
    count = 0
    for i in range(n):
        if not seen[i]:
            count += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = perm[j]
    return count


def extract_mfcad_labels(step_path: str) -> np.ndarray:
    """MFCAD++ stores one machining-feature label per face as the numeric
    tag on each ADVANCED_FACE entity, in file order (same convention as
    src/ccbrep/brep_step_reader.py's function of the same name -- kept as
    a local copy here since dartbrep's face_idx also comes from a plain
    TopExp_Explorer(shape, TopAbs_FACE) walk, in the same file order)."""
    text = Path(step_path).read_text()
    return np.array(
        [int(x) for x in re.findall(r"ADVANCED_FACE\('(\d+)'", text)], dtype=np.int64
    )


def extract_from_step(path: str) -> dict:
    m = extract_map(read_step(path))
    labels = extract_mfcad_labels(path)
    if len(labels) != m["n_faces"]:
        raise DartExtractionError(
            f"{path}: {len(labels)} labels vs {m['n_faces']} faces -- "
            "ADVANCED_FACE order assumption likely violated for this file"
        )
    m["face_labels"] = labels
    m["label_of_dart"] = labels[m["face_of_dart"]]
    return m


if __name__ == "__main__":
    import sys

    result = extract_from_step(sys.argv[1])
    print(
        f"darts={result['n_darts']} V={result['n_vertices']} E={result['n_edges']} "
        f"F={result['n_faces']} loops={result['n_loops']} inner={result['n_inner_loops']} "
        f"chi={result['euler_char']} 2g={result['two_genus']} "
        f"nonmanifold_edges={result['n_nonmanifold_edges']} "
        f"boundary_edges={result['n_boundary_edges']} "
        f"sigma_violations={result['sigma_vertex_violations']}"
    )
