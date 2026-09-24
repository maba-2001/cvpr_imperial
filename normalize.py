"""Kernel-side normalization applied before map extraction: `ShapeFix_Shape`
+ seam-splitting closed faces and edges (`ShapeUpgrade_ShapeDivideClosed`) --
the same step `src/imperial/brepcomplex/extract.py` used, but not something
`src/imperial/dartbrep/extract_darts.py` does (it takes a shape as given).

Without this, a full-circle hole boundary is one edge whose two darts share a
single vertex (chord length exactly 0) and a cylindrical or toroidal face is
periodic in its own parametrization -- both real cases this project has to
handle downstream (a per-edge local frame with no chord to align to; a
nesting check that cannot use signed area on a periodic domain). Splitting
upstream removes the cases instead: every loop becomes simple, so every edge
has two genuinely distinct endpoints and every face is non-periodic. This
runs before both `build_dart_cache.py` (topology) and `extract.py`
(geometry), so the two see the same, already-split map.
"""

from __future__ import annotations


def normalize_shape(shape):
    """ShapeFix + seam-splitting. Returns the normalized shape."""
    from OCC.Core.ShapeFix import ShapeFix_Shape
    from OCC.Core.ShapeUpgrade import ShapeUpgrade_ShapeDivideClosed

    fix = ShapeFix_Shape(shape)
    fix.Perform()
    shape = fix.Shape()

    div = ShapeUpgrade_ShapeDivideClosed(shape)
    div.Perform()
    return div.Result()


def read_step_normalized(path: str):
    """`extract_darts.read_step`, then `normalize_shape`."""
    import sys

    from . import config as cfg

    sys.path.insert(0, str(cfg.ROOT / "src/imperial/dartbrep"))
    from extract_darts import read_step

    return normalize_shape(read_step(str(path)))
