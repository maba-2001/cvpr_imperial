"""Kernel-side shape normalization applied before map extraction.

Seam-splits closed faces and edges (`ShapeUpgrade_ShapeDivideClosed`) so a
full-circle hole boundary (chord length exactly 0) and a periodic cylindrical
or toroidal face never reach the rest of the pipeline: every loop becomes
simple, every edge gets two distinct endpoints, every face is non-periodic.
Runs before both `build_dart_cache.py` (topology) and `extract.py`
(geometry), so both see the same already-split map.
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
    from .extract_darts import read_step

    return normalize_shape(read_step(str(path)))
