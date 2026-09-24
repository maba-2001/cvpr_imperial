"""The canonical per-edge sample representation, shared by `data/extract.py`
(encode) and `realize.py`/`viz_wireframe.py` (decode) so the two directions
are exact inverses of each other. See the README's "Edge and face geometry"
section for the full rationale; summary:

World-axis-aligned deviation from the chord, not a rotated local frame --
rotating into a chord-aligned frame leaves the roll about that axis with no
geometric pick (by the hairy-ball theorem, no continuous rule assigns a
perpendicular pair to every chord direction, so any deterministic completion
is a discontinuity somewhere), and scaling by arc length needs arc length
predicted separately, where the natural alternative (chord length) is exactly
0 for a closed edge and blows up near one. Chord-deviation sidesteps both:
samples are resampled at uniform arc length, each one stored as its
deviation from the chord point at the same arc fraction, in world axes, over
`D = c + delta` (chord length `c`, known from the endpoints, plus `delta`,
how far the curve reaches beyond its chord -- 0 for every line and every arc
up to a semicircle, and bounded even in the closed limit).
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
