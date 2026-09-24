"""The representation: an oriented combinatorial map with a face partition.

A B-rep shell is  M = (D, alpha, phi, Phi)  where

    D      darts (coedges): one traversal of one edge by one loop
    alpha  fixed-point-free involution pairing the two uses of an edge
    phi    permutation whose orbits are the loops (wires)
    Phi    partition of loops into B-rep faces

and  sigma = phi o alpha  is the rotation around a dart's origin vertex.
Cells are orbits: vertices = orbits(sigma), edges = orbits(alpha),
loops = orbits(phi), faces = blocks of Phi.

Everything the penalty-based formulations enforce approximately is a theorem
here:

    loop closure          orbits of a permutation are cycles
    edge used twice       alpha is an involution
    opposite orientation  alpha is fixed-point free; the map is oriented
    vertex manifoldness   the link of a vertex IS its sigma-orbit, one cycle
    wire cyclic order     the phi-orbit, read off directly

so `check` has nothing left to verify at tier 0 -- it is a regression guard,
not a filter. The genuinely contingent facts live at tiers 1-3.

Euler characteristic of the realized surface S (faces are l-holed spheres,
not disks):   chi(S) = V - E + 2F - L.
Derivation: capping every loop with a disk gives the map's surface S_map with
chi = V - E + L; replacing the l caps of a face by one l-holed sphere changes
chi by (2 - l) - l, and summing 2 - 2*l_f over faces gives 2F - 2L.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


def orbits(perm: np.ndarray) -> np.ndarray:
    """Orbit id per element, numbered by first occurrence."""
    n = len(perm)
    out = np.full(n, -1, dtype=np.int64)
    k = 0
    for i in range(n):
        if out[i] >= 0:
            continue
        j = i
        while out[j] < 0:
            out[j] = k
            j = perm[j]
        k += 1
    return out


def _union_find(n: int, pairs) -> np.ndarray:
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)])


@dataclass(frozen=True)
class CMap:
    """alpha, phi over n darts; face_of_loop over the phi-orbits."""

    alpha: np.ndarray        # (n,)
    phi: np.ndarray          # (n,)
    face_of_loop: np.ndarray  # (L,)

    # -------------------------------------------------------------- cells
    @property
    def n_darts(self) -> int:
        return len(self.alpha)

    @cached_property
    def sigma(self) -> np.ndarray:
        return self.phi[self.alpha]

    @cached_property
    def loop_of_dart(self) -> np.ndarray:
        return orbits(self.phi)

    @cached_property
    def vertex_of_dart(self) -> np.ndarray:
        return orbits(self.sigma)

    @cached_property
    def edge_of_dart(self) -> np.ndarray:
        return orbits(self.alpha)

    @cached_property
    def face_of_dart(self) -> np.ndarray:
        return self.face_of_loop[self.loop_of_dart]

    def counts(self) -> tuple[int, int, int, int]:
        """(V, E, F, L)."""
        return (
            int(self.vertex_of_dart.max()) + 1,
            int(self.edge_of_dart.max()) + 1,
            int(self.face_of_loop.max()) + 1,
            len(self.face_of_loop),
        )

    def loop_cycle(self, loop: int) -> np.ndarray:
        """Darts of a loop in traversal order -- the wire, exactly."""
        start = int(np.nonzero(self.loop_of_dart == loop)[0][0])
        out, d = [start], int(self.phi[start])
        while d != start:
            out.append(d)
            d = int(self.phi[d])
        return np.array(out, dtype=np.int64)

    # --------------------------------------------------------- invariants
    def components(self) -> np.ndarray:
        """Component id per dart of the *realized* surface: darts joined by
        alpha, phi, and by sharing a face (a multi-loop face can join what the
        map alone separates)."""
        n = self.n_darts
        pairs = [(i, int(self.alpha[i])) for i in range(n)]
        pairs += [(i, int(self.phi[i])) for i in range(n)]
        first = {}
        fod = self.face_of_dart
        for i in range(n):
            f = int(fod[i])
            if f in first:
                pairs.append((i, first[f]))
            else:
                first[f] = i
        roots = _union_find(n, pairs)
        _, comp = np.unique(roots, return_inverse=True)
        return comp

    def invariants(self) -> dict:
        """Exact topological invariants of the realized surface."""
        comp = self.components()
        lod, vod, eod = self.loop_of_dart, self.vertex_of_dart, self.edge_of_dart
        chis, genera = [], []
        for c in range(int(comp.max()) + 1):
            sel = comp == c
            v = len(np.unique(vod[sel]))
            e = len(np.unique(eod[sel]))
            loops = np.unique(lod[sel])
            f = len(np.unique(self.face_of_loop[loops]))
            chi = v - e + 2 * f - len(loops)
            chis.append(chi)
            genera.append((2 - chi) // 2)
        b0 = len(chis)
        return {
            "b0": b0,
            "b1": int(2 * sum(genera)),
            "b2": b0,                      # closed orientable
            "chi": int(sum(chis)),
            "genus": int(sum(genera)),
            "genus_per_component": sorted(genera, reverse=True),
            "n_faces": self.counts()[2],
        }

    # ------------------------------------------------------------ validity
    def check(self) -> dict:
        """The validity ladder. Tier 0 is guaranteed by construction and kept
        as a regression guard; tiers 1-2 are the contingent combinatorial
        facts; tier 3 (kernel) lives in realize.py."""
        n = self.n_darts
        a, p = self.alpha, self.phi
        idx = np.arange(n)
        t0 = {
            "alpha_involution": bool(np.array_equal(a[a], idx)),
            "alpha_fixed_point_free": bool(np.all(a != idx)),
            "phi_permutation": bool(len(np.unique(p)) == n),
            "faces_nonempty": bool(
                len(np.unique(self.face_of_loop)) == int(self.face_of_loop.max()) + 1
            ) if len(self.face_of_loop) else False,
        }
        inv = self.invariants()
        # Tier 1 reduces to connectivity: for a connected oriented map,
        # chi(S) = chi(S_map) - 2(L - F) <= chi(S_map) <= 2, so genus >= 0 is
        # automatic too. Generation enforces connectivity by construction
        # (every new loop is entered through an already-open dart), so this is
        # also a guard rather than a filter.
        t1 = {"connected": inv["b0"] == 1}
        loop_len = np.bincount(self.loop_of_dart, minlength=len(self.face_of_loop))
        loops_per_face = np.bincount(self.face_of_loop)
        return {
            "tier0": t0, "tier0_pass": all(t0.values()),
            "tier1": t1, "tier1_pass": all(t1.values()),
            "invariants": inv,
            # geometric-risk diagnostics, not validity: short loops are legal
            # (a full circle bounds a loop of one dart) but stress the fitter.
            "diagnostics": {
                "n_monogon_loops": int((loop_len == 1).sum()),
                "n_bigon_loops": int((loop_len == 2).sum()),
                "max_loops_per_face": int(loops_per_face.max()),
                "n_multiloop_faces": int((loops_per_face > 1).sum()),
            },
        }


def from_arrays(alpha, phi, face_of_dart) -> CMap:
    """Build a CMap from an extraction that labels darts, not loops."""
    alpha = np.asarray(alpha, dtype=np.int64)
    phi = np.asarray(phi, dtype=np.int64)
    lod = orbits(phi)
    n_loops = int(lod.max()) + 1
    face_of_loop = np.zeros(n_loops, dtype=np.int64)
    face_of_loop[lod] = np.asarray(face_of_dart, dtype=np.int64)
    # renumber faces densely, by first appearance
    _, face_of_loop = np.unique(face_of_loop, return_inverse=True)
    return CMap(alpha, phi, face_of_loop.astype(np.int64))


def reindex_cells(values: np.ndarray, new_of_dart: np.ndarray,
                  old_of_dart: np.ndarray) -> np.ndarray:
    """Move a per-cell array from one dart-labelling of the cells to
    another: `values[old_of_dart[d]]` and `out[new_of_dart[d]]` are the same
    cell for every dart d, for any two per-dart cell-id arrays (e.g. an
    extraction's own edge/face numbering vs this module's orbit-derived
    one) covering the same map."""
    out = np.zeros((int(new_of_dart.max()) + 1, *values.shape[1:]), dtype=values.dtype)
    out[new_of_dart] = values[old_of_dart]
    return out
