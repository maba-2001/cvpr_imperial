"""The canonical per-edge sample representation, shared by extract.py
(encode) and realize.py/viz_wireframe.py (decode) so the two directions are
exact inverses of each other.

World-axis-aligned deviation from the chord, not a rotated local frame. An
earlier version rotated samples into a chord-aligned frame (origin at the
start vertex, x-axis toward the end vertex) and scaled by the edge's own arc
length. Two problems with that, in order of severity:

  1. The roll about the chord axis was never picked by anything geometric --
     the other two frame axes were an arbitrary deterministic completion
     (the old `local_frame`'s `|x[0]| < 0.9` switch), and by the hairy-ball
     theorem no continuous rule assigns a perpendicular pair to every chord
     direction. That switch was itself a discontinuity: two edges with chord
     directions on either side of |x[0]| = 0.9 got targets that don't vary
     smoothly into each other, even though the edges themselves do.
  2. Scaling by arc length decouples shape from size, but arc length isn't
     recoverable from the two endpoints -- it has to be predicted too, and
     the natural alternative (chord length) is exactly 0 for a closed edge
     (full-circle hole boundary) and blows up for anything near-closed
     (measured on the v2 cache: worst case 1e12x the chord).

Chord-deviation sidesteps both. Samples are resampled at uniform *arc length*
first (`extract._sample_curve_arclength`), so the same shape gives the same
target regardless of how the CAD kernel parameterized it, and a straight
line's samples land exactly on the chord. Each sample's deviation from the
chord point at the same arc fraction is then expressed directly in *world*
axes -- no rotation, so no roll convention and no discontinuity -- and scaled
by `D = c + delta` (chord length `c`, known from the endpoints, plus `delta`,
how far the curve reaches beyond its chord). `delta` is 0 for every line and
every arc up to a semicircle, so it -- not arc length -- is the one scalar
that needs predicting, and it stays bounded (measured max |deviation| 0.997
over the full v2 cache) where chord-relative scaling does not.
"""

from __future__ import annotations

import numpy as np


def lerp(p0: np.ndarray, p1: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Chord points at arc-length fractions `s` (n,): p0 + s * (p1 - p0)."""
    return p0 + s[:, None] * (p1 - p0)


def to_deviation(points: np.ndarray, p0: np.ndarray, p1: np.ndarray, eps: float = 1e-6):
    """World points sampled at uniform arc-length fractions (endpoints
    included, i.e. `points[0] == p0`, `points[-1] == p1`) -> (dev, D).

    `dev` (n, 3): each sample's world-axis deviation from the chord point at
    the same arc fraction, divided by `D` -- exactly 0 at the two endpoints.
    `D`: the chord length plus how far the curve reaches beyond it (the
    scale that keeps `dev` bounded for both lines and near-closed arcs).
    """
    n = len(points)
    s = np.linspace(0.0, 1.0, n)
    raw = points - lerp(p0, p1, s)
    reach = max(float(np.linalg.norm(points - p0, axis=1).max()),
                float(np.linalg.norm(points - p1, axis=1).max()))
    D = max(reach, eps)
    return raw / D, D


def from_deviation(dev: np.ndarray, p0: np.ndarray, p1: np.ndarray, D: float) -> np.ndarray:
    """Inverse of `to_deviation`: world points at the same uniform
    arc-length fractions, given the stored/generated scale `D`."""
    n = len(dev)
    s = np.linspace(0.0, 1.0, n)
    return lerp(p0, p1, s) + D * dev
