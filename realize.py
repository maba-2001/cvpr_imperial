"""Tiers 2-3: a valid map plus geometry -> an OCCT solid.

Two things here answer criticisms of the penalty-based pipeline directly.

Primitives, not splines everywhere. CAD is mostly planes, cylinders, lines and
circles; fitting a degree-3 B-spline through 16 samples of a straight edge is
the main reason kernel validity trails topological validity. Each cell is fit
with the cheapest analytic primitive whose residual is under tolerance, and
falls back to a spline only when none is.

Outer loops are *derived from orientation*, not guessed from bounding boxes.
Each loop is projected into the face's own parameter domain and its signed area
is computed; the loop whose orientation agrees with the surface normal is the
outer one and the rest are holes. The same projection gives the tier-2 nesting
check -- every inner loop must lie inside the outer polygon -- which is the one
genuinely contingent embedding condition the combinatorics cannot decide.
"""

from __future__ import annotations

import numpy as np

from . import config as cfg
from .cmap import CMap
from .frame import from_deviation


# --------------------------------------------------------------- primitives

def fit_line(pts: np.ndarray) -> float:
    """Residual of the best straight line through the samples."""
    d = pts[-1] - pts[0]
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.inf
    d = d / n
    off = pts - pts[0]
    return float(np.abs(off - (off @ d)[:, None] * d).max())


def fit_circle(pts: np.ndarray):
    """(centre, normal, radius, residual) of the best circle, or None."""
    c0 = pts.mean(0)
    _, s, vt = np.linalg.svd(pts - c0)
    if s[2] > cfg.PRIMITIVE_TOL * max(s[0], 1e-9) * 10:
        return None                                   # not planar
    normal = vt[2]
    e1, e2 = vt[0], vt[1]
    x, y = (pts - c0) @ e1, (pts - c0) @ e2
    a = np.stack([x, y, np.ones_like(x)], 1)
    sol, *_ = np.linalg.lstsq(a, x ** 2 + y ** 2, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cx ** 2 + cy ** 2
    if r2 <= 0:
        return None
    r = float(np.sqrt(r2))
    res = float(np.abs(np.hypot(x - cx, y - cy) - r).max())
    return c0 + cx * e1 + cy * e2, normal, r, res


def fit_plane(grid: np.ndarray):
    """(point, normal, residual) of the best plane."""
    pts = grid.reshape(-1, 3)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c)
    n = vt[2]
    return c, n, float(np.abs((pts - c) @ n).max())


def fit_cylinder(grid: np.ndarray):
    """(point, axis, radius, residual) of the best cylinder, or None.

    The axis is the grid direction with the smallest spread of chord
    directions; once it is fixed the problem is a 2-D circle fit.
    """
    pts = grid.reshape(-1, 3)
    best = None
    for axis in (grid[1:] - grid[:-1]).reshape(-1, 3), (grid[:, 1:] - grid[:, :-1]).reshape(-1, 3):
        d = axis / np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-9)
        d = d * np.sign(d @ d[0])[:, None]
        a = d.mean(0)
        a = a / max(np.linalg.norm(a), 1e-9)
        proj = pts - (pts @ a)[:, None] * a
        c0 = proj.mean(0)
        e1, e2 = _frame(a)
        x, y = (proj - c0) @ e1, (proj - c0) @ e2
        m = np.stack([x, y, np.ones_like(x)], 1)
        sol, *_ = np.linalg.lstsq(m, x ** 2 + y ** 2, rcond=None)
        cx, cy = sol[0] / 2, sol[1] / 2
        r2 = sol[2] + cx ** 2 + cy ** 2
        if r2 <= 0:
            continue
        r = float(np.sqrt(r2))
        res = float(np.abs(np.hypot(x - cx, y - cy) - r).max())
        cand = (c0 + cx * e1 + cy * e2, a, r, res)
        if best is None or res < best[3]:
            best = cand
    return best


def _frame(a: np.ndarray):
    t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, t)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(a, e1)


# -------------------------------------------------------------- realization

def realize(m: CMap, geom: dict, tol: float = cfg.PRIMITIVE_TOL) -> dict:
    """Build the solid. Returns tier-2 and tier-3 outcomes plus the shape."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeWire, BRepBuilderAPI_Sewing,
    )
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.GeomAPI import (
        GeomAPI_PointsToBSpline, GeomAPI_PointsToBSplineSurface, GeomAPI_ProjectPointOnSurf,
    )
    from OCC.Core.Geom import Geom_Circle, Geom_CylindricalSurface, Geom_Plane
    from OCC.Core.GeomAbs import GeomAbs_C2
    from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Shape
    from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
    from OCC.Core.TopAbs import TopAbs_SHELL
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods
    from OCC.Core.gp import gp_Ax2, gp_Ax3, gp_Dir, gp_Pnt

    out = {"tier2": {}, "sewn": False, "kernel_valid": False, "solid": None, "error": None}
    try:
        vpos = geom["v"].reshape(-1, 3).astype(np.float64)
        e_flat = geom["e"].astype(np.float64)
        interior = e_flat[:, :-1].reshape(-1, cfg.N_CURVE_SAMPLES - 2, 3)
        delta = np.clip(e_flat[:, -1], 0.0, None)   # generated bulge-beyond-chord, floored at 0
        grids = geom["f"].reshape(-1, cfg.N_SURF_GRID, cfg.N_SURF_GRID, 3).astype(np.float64)
        vod, eod, fol = m.vertex_of_dart, m.edge_of_dart, m.face_of_loop
        lower = np.minimum(np.arange(m.n_darts), m.alpha)

        # One OCCT vertex per map vertex, shared by every edge that meets it:
        # wires then close exactly instead of relying on a fit tolerance.
        occ_v = [BRepBuilderAPI_MakeVertex(gp_Pnt(*p)).Vertex() for p in vpos]
        ends = np.zeros((interior.shape[0], 2), dtype=np.int64)
        curves = np.zeros((interior.shape[0], cfg.N_CURVE_SAMPLES, 3))
        for e in range(interior.shape[0]):
            d = int(np.nonzero((eod == e) & (lower == np.arange(m.n_darts)))[0][0])
            ends[e] = (vod[d], vod[m.alpha[d]])
            p0, p1 = vpos[ends[e][0]], vpos[ends[e][1]]
            chord = float(np.linalg.norm(p1 - p0))
            D = max(chord + delta[e], 1e-6)
            dev = np.zeros((cfg.N_CURVE_SAMPLES, 3))
            dev[1:-1] = interior[e]
            curves[e] = from_deviation(dev, p0, p1, D)

        kinds, occ_edges = [], []
        for e, pts in enumerate(curves):
            v0, v1 = occ_v[int(ends[e][0])], occ_v[int(ends[e][1])]
            closed = ends[e][0] == ends[e][1]
            if not closed and fit_line(pts) < tol:
                occ_edges.append(BRepBuilderAPI_MakeEdge(v0, v1).Edge())
                kinds.append("line")
                continue
            circ = fit_circle(pts)
            if circ is not None and circ[3] < tol:
                c, nrm, r_, _ = circ
                if closed:                      # a full circle bounds a loop of one dart
                    nrm = _loop_normal(pts, c)
                geom_c = Geom_Circle(gp_Ax2(gp_Pnt(*c), gp_Dir(*nrm)), r_)
                mk = BRepBuilderAPI_MakeEdge(geom_c, v0, v1)
                if mk.IsDone():
                    occ_edges.append(mk.Edge())
                    kinds.append("circle")
                    continue
            arr = TColgp_Array1OfPnt(1, len(pts))
            for i, p in enumerate(pts):
                arr.SetValue(i + 1, gp_Pnt(*p))
            # Low-degree approximation, not high-degree interpolation:
            # GeomAPI_PointsToBSpline(arr) alone fits an exact curve through
            # every sample at up to degree 8, which for only 16 generated
            # (imperfectly smooth) points is numerically unstable -- Runge-
            # phenomenon overshoot, control points landing 2-3 orders of
            # magnitude outside the data's own bounding box (measured: degree
            # 8 blew a 0.6-unit edge out to poles 160+ units away; capping at
            # degree 2 keeps every case tested within ~1.5x of the raw
            # extent). tol=0.02 rather than `tol` (1e-3, the primitive-fit
            # threshold): forcing quadratic poles within 1e-3 of noisy points
            # reintroduces the same overshoot.
            curve = GeomAPI_PointsToBSpline(arr, 1, 2, GeomAbs_C2, 0.02).Curve()
            mk = BRepBuilderAPI_MakeEdge(curve, v0, v1)
            if not mk.IsDone():
                return {**out, "error": f"edge build failed (edge {e})"}
            occ_edges.append(mk.Edge())
            kinds.append("spline")
        out["curve_kinds"] = kinds

        def build_surface(f: int, flip: bool):
            """Analytic surface for face f; `flip` reverses its normal."""
            grid = grids[f]
            pt, nrm, res = fit_plane(grid)
            if res < tol:
                n = -nrm if flip else nrm
                return Geom_Plane(gp_Ax3(gp_Pnt(*pt), gp_Dir(*n))), "plane"
            cyl = fit_cylinder(grid)
            if cyl is not None and cyl[3] < tol:
                c, a, r_, _ = cyl
                a = -a if flip else a
                return Geom_CylindricalSurface(gp_Ax3(gp_Pnt(*c), gp_Dir(*a)), r_), "cylinder"
            arr2 = TColgp_Array2OfPnt(1, cfg.N_SURF_GRID, 1, cfg.N_SURF_GRID)
            for i in range(cfg.N_SURF_GRID):
                for j in range(cfg.N_SURF_GRID):
                    arr2.SetValue(i + 1, j + 1, gp_Pnt(*grid[i, j]))
            # Same overshoot risk as the curve case above, same fix: cap
            # degree at 2 and use a tolerance loose enough not to force the
            # fit back toward interpolation (see the curve comment).
            s = GeomAPI_PointsToBSplineSurface(arr2, 1, 2, GeomAbs_C2, 0.02).Surface()
            if flip:
                s.UReverse()
            return s, "spline"

        def project(surface, pts: np.ndarray) -> np.ndarray:
            uv = []
            for q in pts:
                pr = GeomAPI_ProjectPointOnSurf(gp_Pnt(*q), surface)
                uv.append(pr.LowerDistanceParameters() if pr.NbPoints() else (0.0, 0.0))
            return np.array(uv, dtype=np.float64)

        surf_kinds, sew = [], BRepBuilderAPI_Sewing(cfg.SEW_TOL)
        n_checked, n_bad_nesting = 0, 0
        for f in range(grids.shape[0]):
            loops = np.nonzero(fol == f)[0]
            cycles = [m.loop_cycle(int(l)) for l in loops]
            pts3d = []
            for darts in cycles:
                seq = []
                for d in darts:
                    e = int(eod[d])
                    c = curves[e] if lower[d] == d else curves[e][::-1]
                    seq.append(c[:-1])
                pts3d.append(np.concatenate(seq))

            # The face's orientation comes from the map: the outer loop is the
            # one of largest area, and its traversal direction fixes the
            # surface normal -- the signed information a bbox heuristic throws
            # away.
            surface, kind = build_surface(f, False)
            polys = [project(surface, p) for p in pts3d]
            areas = np.array([_signed_area(p) for p in polys])
            outer = int(np.argmax(np.abs(areas)))
            if areas[outer] < 0:
                surface, kind = build_surface(f, True)
                polys = [project(surface, p) for p in pts3d]
                areas = np.array([_signed_area(p) for p in polys])
                outer = int(np.argmax(np.abs(areas)))
            surf_kinds.append(kind)
            # Nesting is only meaningful on a non-periodic domain: on a
            # cylinder the seam wraps the polygon and both the area and the
            # crossing test lose their meaning, so those faces are reported as
            # unchecked rather than silently passed.
            if not surface.IsUPeriodic() and not surface.IsVPeriodic():
                for i, poly in enumerate(polys):
                    if i == outer:
                        continue
                    n_checked += 1
                    if areas[i] > 0 or not _inside(poly.mean(0), polys[outer]):
                        n_bad_nesting += 1

            wires = []
            for darts in cycles:
                mw = BRepBuilderAPI_MakeWire()
                for d in darts:
                    e = int(eod[d])
                    mw.Add(occ_edges[e] if lower[d] == d
                           else topods.Edge(occ_edges[e].Reversed()))
                if not mw.IsDone():
                    return {**out, "error": f"wire build failed (face {f})"}
                wires.append(mw.Wire())
            mf = BRepBuilderAPI_MakeFace(surface, wires[outer], True)
            for i, w in enumerate(wires):
                if i != outer:
                    mf.Add(topods.Wire(w.Reversed()))
            if not mf.IsDone():
                return {**out, "error": f"face build failed (face {f})"}
            sf = ShapeFix_Face(mf.Face())
            sf.Perform()          # builds the missing pcurves BRepCheck wants
            sew.Add(sf.Face())
        out["surface_kinds"] = surf_kinds
        out["tier2"] = {"loops_nested": n_bad_nesting == 0,
                        "n_inner_loops": n_checked, "n_bad_nesting": n_bad_nesting}

        sew.Perform()
        exp = TopExp_Explorer(sew.SewedShape(), TopAbs_SHELL)
        if not exp.More():
            return {**out, "error": "sewing produced no shell"}
        shell = topods.Shell(exp.Current())
        out["sewn"] = True
        out["shell_closed"] = bool(BRep_Tool.IsClosed(shell))
        ms = BRepBuilderAPI_MakeSolid(shell)
        if not ms.IsDone():
            return {**out, "error": "MakeSolid failed"}
        fix = ShapeFix_Shape(ms.Solid())
        fix.Perform()
        out["solid"] = fix.Shape()
        out["kernel_valid"] = bool(BRepCheck_Analyzer(out["solid"]).IsValid())
        return out
    except Exception as e:                                        # noqa: BLE001
        return {**out, "error": f"{type(e).__name__}: {e}"}


def _loop_normal(pts: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Orientation of a closed sample loop, by its own vector area."""
    rel = pts - centre
    n = np.cross(rel, np.roll(rel, -1, axis=0)).sum(0)
    return n / max(np.linalg.norm(n), 1e-9)


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _inside(pt: np.ndarray, poly: np.ndarray) -> bool:
    x, y = pt
    xs, ys = poly[:, 0], poly[:, 1]
    xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)
    crosses = ((ys > y) != (ys2 > y)) & (x < (xs2 - xs) * (y - ys) / (ys2 - ys + 1e-18) + xs)
    return bool(crosses.sum() % 2 == 1)
