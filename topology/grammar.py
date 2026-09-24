"""Canonical code: one object that is simultaneously the canonical labelling,
the isomorphism test, and the generation grammar. See the README's "canonical
code" and "vertex block" sections for the design rationale (why face-major,
why FACE_TYPE/EDGE_TYPE/LOOP_IS_OUTER ride along in the walk, why
LOOP_IS_OUTER also steers `preview`'s tie-break); this file is the mechanism.

Grammar (one group per face, faces in discovery order):

    NLOOPS k                 number of loops of this face
    FACE_TYPE t               this face's surface kind (plane/cyl/.../other)
    per loop:  LEN  l        number of darts in this loop
               LOOP_IS_OUTER o  is this loop the face's outer boundary?
               per dart:  ALPHA x     0 = OPEN (partner comes later)
                                      i-j = close currently-open dart j
                          EDGE_TYPE t  this dart's edge kind (line/circle/other)

The first dart of the entry loop of every face after the first is forced to
close the earliest open dart -- makes the walk breadth-first, makes the
result connected by construction, and makes termination structural: the code
ends exactly when a face group leaves no dart open, so there is no stop token
to get wrong and no complete sequence decodes to an invalid map.

An edge's two darts get the same EDGE_TYPE value by construction while
encoding; nothing forces a *sampled* sequence to agree at both, so `decode`
resolves a disagreement by taking whichever was emitted last (type never
affects grammar legality, unlike ALPHA's pointer consistency, so this is a
rare cosmetic fallback, not a validity concern).
"""

from __future__ import annotations

import time

import numpy as np

from ..data.cmap import CMap

NLOOPS, LEN, ALPHA, FACE_TYPE, EDGE_TYPE, LOOP_IS_OUTER = 0, 1, 2, 3, 4, 5   # token kinds
OPEN = 0                            # ALPHA value meaning "partner comes later"
N_FACE_TYPES = 6                    # plane, cylinder, cone, sphere, torus, other
N_EDGE_TYPES = 3                    # line, circle, other
N_LOOP_IS_OUTER = 2                 # inner (0), outer (1)
# One token per vertex, appended after the topology is complete (see
# topology/model.py) -- not part of the Builder grammar above: vertex count
# and order are fully determined by the finished map, so there is nothing
# for the grammar to constrain.
VERTEX = 6


# --------------------------------------------------------------- refinement

def _refine(m: CMap, col: np.ndarray, rounds: int) -> np.ndarray:
    """WL refinement along alpha, phi, phi^-1 and the face partition.

    Purely topological -- doesn't see FACE_TYPE/EDGE_TYPE at all. That's a
    deliberate scope cut, not an oversight: `preview`/`emit` already fold
    type into the token tuples that decide the actual code, so the code
    itself is correctly type-sensitive regardless. What `_refine` feeds --
    `canonical`'s candidate-root set and `tied`'s residual tie-break -- only
    affects how many equivalent candidates get tried before the (always
    correct) token-level comparison picks the minimum, i.e. a possible
    efficiency cost on highly symmetric parts, never a correctness one.
    """
    lod, fol = m.loop_of_dart, m.face_of_loop
    phi_inv = np.empty_like(m.phi)
    phi_inv[m.phi] = np.arange(m.n_darts)
    fod = fol[lod]
    n_faces = int(fol.max()) + 1
    for _ in range(rounds):
        fsum = np.bincount(fod, weights=col, minlength=n_faces).astype(np.int64)
        fmax = np.zeros(n_faces, dtype=np.int64)
        np.maximum.at(fmax, fod, col)
        # ascontiguousarray + explicit dtype: np.unique(axis=0) compares rows
        # via a byte-level view, which numpy has been observed to choke on
        # ("Cannot compare structured arrays...") for a non-C-contiguous or
        # mixed-width input built by np.stack over fancy-indexed columns.
        sig = np.ascontiguousarray(
            np.stack([col, col[m.alpha], col[m.phi], col[phi_inv],
                     fsum[fod], fmax[fod]], axis=1), dtype=np.int64)
        _, col = np.unique(sig, axis=0, return_inverse=True)
    return col


def dart_colors(m: CMap, rounds: int = 4) -> np.ndarray:
    """Isomorphism-invariant colouring of the darts."""
    lod, fol = m.loop_of_dart, m.face_of_loop
    init = np.ascontiguousarray(
        np.stack([np.bincount(lod)[lod], np.bincount(fol)[fol[lod]]], axis=1),
        dtype=np.int64)
    _, col = np.unique(init, axis=0, return_inverse=True)
    return _refine(m, col, rounds)


def state_colors(m: CMap, base: np.ndarray, new_of: np.ndarray,
                 rounds: int = 4) -> np.ndarray:
    """Individualization-refinement against the walk's own progress: every
    already-numbered dart is individualized by its number, then WL refines.

    This is what makes the walk deterministic *and* label-independent. Two
    candidates that still collide here agree on every invariant WL can see
    relative to the numbered prefix, i.e. they are interchangeable, so the
    tie may be broken arbitrarily."""
    return _refine(m, base * (m.n_darts + 2) + (new_of + 1), rounds)


# ------------------------------------------------------------------ encode

class _Walk:
    """State of an encoding walk; cloneable so residual ties can be branched."""

    def __init__(self, m: CMap, base: np.ndarray, edge_type: np.ndarray,
                face_type: np.ndarray, loop_is_outer: np.ndarray):
        self.m, self.base = m, base
        self.edge_type, self.face_type = edge_type, face_type
        self.loop_is_outer = loop_is_outer
        self.new_of = np.full(m.n_darts, -1, dtype=np.int64)
        self.order: list[int] = []
        self.open: list[int] = []          # new-indices with unassigned partner
        self.tokens: list[tuple[int, int]] = []
        self.rest: list[int] = []          # loops of the current face, not emitted
        self.bound: tuple[int, ...] | None = None   # best code so far, for pruning
        self.beaten = False                # True once this walk is strictly smaller
        self.loops_of_face: dict[int, list[int]] = {}
        for l, f in enumerate(m.face_of_loop):
            self.loops_of_face.setdefault(int(f), []).append(l)
        self.darts_of_loop = {l: np.nonzero(m.loop_of_dart == l)[0]
                              for l in range(len(m.face_of_loop))}

    def clone(self) -> "_Walk":
        w = _Walk.__new__(_Walk)
        w.m, w.base = self.m, self.base
        w.edge_type, w.face_type = self.edge_type, self.face_type
        w.loop_is_outer = self.loop_is_outer
        w.new_of = self.new_of.copy()
        w.order, w.open = list(self.order), list(self.open)
        w.tokens, w.rest = list(self.tokens), list(self.rest)
        w.loops_of_face, w.darts_of_loop = self.loops_of_face, self.darts_of_loop
        w.bound, w.beaten = self.bound, self.beaten
        return w

    def values(self) -> tuple[int, ...]:
        return tuple(v for _, v in self.tokens)

    def block_of(self, start: int) -> list[int]:
        """Darts of start's loop in traversal order, beginning at start."""
        out, d = [start], int(self.m.phi[start])
        while d != start:
            out.append(d)
            d = int(self.m.phi[d])
        return out

    def preview(self, start: int) -> tuple[int, ...]:
        """Comparison key for picking the next block: an outer-loop
        preference first (see the module docstring -- this makes the choice
        among a face's remaining loops, when genuinely free, prefer outer
        over inner regardless of length), then the tokens this block would
        actually emit, without mutating state."""
        block = self.block_of(start)
        base = len(self.order)
        seen = {d: base + i for i, d in enumerate(block)}
        outer_pref = 0 if self.loop_is_outer[self.m.loop_of_dart[start]] else 1
        vals = [outer_pref, len(block)]
        for i, d in enumerate(block):
            j = int(self.new_of[self.m.alpha[d]])
            if j < 0:
                j = seen.get(int(self.m.alpha[d]), -1)
            vals.append(OPEN if j < 0 or j >= base + i else base + i - j)
            vals.append(int(self.edge_type[self.m.edge_of_dart[d]]))
        return tuple(vals)

    def _bound_check(self, value: int) -> None:
        """Abort as soon as this walk's code is provably not the minimum."""
        if self.beaten or self.bound is None:
            return
        i = len(self.tokens) - 1
        if i >= len(self.bound) or value > self.bound[i]:
            raise _Pruned
        if value < self.bound[i]:
            self.beaten = True

    def emit(self, start: int) -> None:
        block = self.block_of(start)
        self.tokens.append((LEN, len(block)))
        self._bound_check(len(block))
        outer = int(self.loop_is_outer[self.m.loop_of_dart[start]])
        self.tokens.append((LOOP_IS_OUTER, outer))
        self._bound_check(outer)
        for d in block:
            i = len(self.order)
            self.new_of[d] = i
            self.order.append(d)
            j = int(self.new_of[self.m.alpha[d]])
            if j < 0 or j == i:
                self.tokens.append((ALPHA, OPEN))
                self.open.append(i)
            else:
                self.tokens.append((ALPHA, i - j))
                self.open.remove(j)
            self._bound_check(self.tokens[-1][1])
            et = int(self.edge_type[self.m.edge_of_dart[d]])
            self.tokens.append((EDGE_TYPE, et))
            self._bound_check(et)
        self.rest.remove(int(self.m.loop_of_dart[start]))

    def enter_face(self, entry: int) -> None:
        """Open the face group of `entry` and emit its entry loop."""
        face = int(self.m.face_of_loop[self.m.loop_of_dart[entry]])
        self.rest = list(self.loops_of_face[face])
        self.tokens.append((NLOOPS, len(self.rest)))
        self._bound_check(len(self.rest))
        ft = int(self.face_type[face])
        self.tokens.append((FACE_TYPE, ft))
        self._bound_check(ft)
        self.emit(entry)

    def tied(self) -> list[int]:
        """Minimal next blocks of the current face; ties resolved by
        refinement against the already-numbered prefix."""
        keyed = [(self.preview(int(d)), int(d))
                 for l in self.rest for d in self.darts_of_loop[l]]
        best = min(k for k, _ in keyed)
        cand = [d for k, d in keyed if k == best]
        if len(cand) == 1:
            return cand
        col = state_colors(self.m, self.base, self.new_of)
        lod = self.m.loop_of_dart
        key = lambda d: (int(col[d]),
                         tuple(sorted(col[self.darts_of_loop[int(lod[d])]].tolist())))
        best2 = min(key(d) for d in cand)
        return [d for d in cand if key(d) == best2]


class _Pruned(Exception):
    """Raised to abandon a walk that cannot be the lexicographic minimum."""


def _best(w: _Walk, budget: list[int], deadline: float | None = None) -> _Walk:
    """Finish the walk, branching on residual ties while the budget and
    (wall-clock) deadline last.

    The deadline is checked only here, between Python-level decisions, never
    inside a numpy call -- interrupting numpy's C loops with a signal-based
    timeout has been observed to corrupt their internal state and raise
    unrelated errors (e.g. a bogus "cannot compare structured arrays" from
    `np.unique`) instead of cleanly propagating. This check is the safe
    equivalent: once time runs out, branching simply stops and every
    remaining tie is broken by taking the first candidate, same as
    `quick_encode`.
    """
    while True:
        if w.rest:
            cand = w.tied()
            if (len(cand) > 1 and budget[0] > 0
                    and (deadline is None or time.time() < deadline)):
                budget[0] -= len(cand)
                outs, local = [], w.bound
                for d in cand:
                    w2 = w.clone()
                    w2.bound = local
                    try:
                        w2.emit(d)
                        outs.append(_best(w2, budget, deadline))
                    except _Pruned:
                        continue
                    v = outs[-1].values()
                    local = v if local is None else min(local, v)
                if not outs:
                    raise _Pruned
                return min(outs, key=lambda x: x.values())
            w.emit(cand[0])
        elif w.open:
            w.enter_face(int(w.m.alpha[w.order[w.open[0]]]))
        else:
            return w


def quick_encode(m: CMap, edge_type: np.ndarray, face_type: np.ndarray,
                 loop_is_outer: np.ndarray, root: int = 0,
                 base_colors: np.ndarray | None = None):
    """(tokens, order) for one root, breaking every tie by taking the first
    candidate -- O(n) in the number of darts, never branches.

    This is what the training data uses: the autoregressive model only needs
    *a* fixed, deterministic order per shape, not a canonical one, exactly as
    GraphRNN-style graph generators walk from an arbitrary root rather than
    canonicalizing. Two isomorphic maps can get different codes from this
    function; that only matters for isomorphism testing, which `canonical`
    is for.

    `edge_type`/`face_type`/`loop_is_outer`: (n_edges,)/(n_faces,)/(n_loops,)
    integer/bool arrays, indexed by `m`'s own `edge_of_dart`/`face_of_dart`/
    `loop_of_dart` numbering (not an extraction's raw numbering -- reindex
    first, see `cmap.reindex_cells`).
    """
    base = dart_colors(m) if base_colors is None else base_colors
    w = _Walk(m, base, edge_type, face_type, loop_is_outer)
    w.enter_face(int(root))
    while w.rest:
        w.emit(w.tied()[0])
    while w.open:
        w.enter_face(int(m.alpha[w.order[w.open[0]]]))
        while w.rest:
            w.emit(w.tied()[0])
    assert len(w.order) == m.n_darts, "walk did not cover the map"
    return w.tokens, np.array(w.order, dtype=np.int64)


def encode(m: CMap, edge_type: np.ndarray, face_type: np.ndarray,
           loop_is_outer: np.ndarray, root: int,
           base_colors: np.ndarray | None = None,
           budget: int | list[int] = 4000, bound: tuple[int, ...] | None = None,
           deadline: float | None = None):
    """(tokens, order) for one root, or None if `bound` or an exhausted
    `budget` prunes it away.

    The walk is deterministic except at residual ties, which are branched over
    up to `budget` expansions -- a plain int for a fresh per-call budget, or a
    mutable `[int]` shared across several calls (`canonical` passes one, so
    consumption on one root carries over to the next) -- and until
    `deadline` (a `time.time()`-comparable wall-clock bound). Exact
    (isomorphism-invariant) while budget and time remain; degrades to
    `quick_encode`-like behaviour once either runs out, rather than raising or
    hanging.
    """
    base = dart_colors(m) if base_colors is None else base_colors
    w = _Walk(m, base, edge_type, face_type, loop_is_outer)
    w.bound = bound
    budget_box = budget if isinstance(budget, list) else [budget]
    try:
        w.enter_face(int(root))
        w = _best(w, budget_box, deadline)
    except _Pruned:
        return None
    assert len(w.order) == m.n_darts, "walk did not cover the map"
    return w.tokens, np.array(w.order, dtype=np.int64)


def canonical(m: CMap, edge_type: np.ndarray, face_type: np.ndarray,
             loop_is_outer: np.ndarray, total_budget: int = 20000,
             time_budget_s: float = 3.0):
    """Lexicographically minimal code over the invariant candidate roots.

    Exact (isomorphism-invariant) whenever it stays within budget: `code_key(a)
    == code_key(b) iff a ~= b`. Expensive on highly symmetric parts (many
    candidate roots, many ties per step), so `total_budget` is shared across
    *all* roots, not reset per root, and a wall-clock `time_budget_s` bounds
    the whole call regardless -- on the rare part that still exceeds both, the
    result degrades to a deterministic-but-not-necessarily-canonical code
    (safe to use, just not guaranteed isomorphism-exact against a
    differently-labelled duplicate) instead of hanging. Meant for evaluation
    (novelty, uniqueness, memorisation), not for building training data; see
    `quick_encode` for that.
    """
    base = dart_colors(m)
    best = None
    budget = [total_budget]
    deadline = time.time() + time_budget_s
    for r in np.nonzero(base == base.min())[0]:
        if budget[0] <= 0 or time.time() >= deadline:
            break
        got = encode(m, edge_type, face_type, loop_is_outer, int(r), base,
                     budget=budget, deadline=deadline,
                     bound=None if best is None else best[0])
        if got is None:
            continue
        key = tuple(v for _, v in got[0])
        if best is None or key < best[0]:
            best = (key, got[0], got[1])
    return best[1], best[2]


def code_key(m: CMap, edge_type: np.ndarray, face_type: np.ndarray,
            loop_is_outer: np.ndarray) -> tuple[int, ...]:
    """Hashable canonical form. Equality == isomorphism (topology + type)."""
    return tuple(v for _, v in canonical(m, edge_type, face_type, loop_is_outer)[0])


def isomorphic(a: CMap, a_edge_type: np.ndarray, a_face_type: np.ndarray,
              a_loop_is_outer: np.ndarray, b: CMap, b_edge_type: np.ndarray,
              b_face_type: np.ndarray, b_loop_is_outer: np.ndarray) -> bool:
    return (code_key(a, a_edge_type, a_face_type, a_loop_is_outer)
            == code_key(b, b_edge_type, b_face_type, b_loop_is_outer))


# ------------------------------------------------------------------ decode

def decode(tokens):
    """Inverse of `encode`; total on any complete token stream.

    Returns (CMap, edge_type, face_type, loop_is_outer), indexed by the
    decoded map's own numbering: faces and loops by discovery order (this
    function assigns both ids by simple sequential append, in the same
    order `CMap`'s own `face_of_loop` and `orbits(phi)` end up using, since
    each loop's darts form one contiguous, monotonically-placed block);
    edges by `CMap.edge_of_dart` orbit order, via a reindexing scatter.
    """
    alpha: dict[int, int] = {}
    phi = []
    face_of_loop: list[int] = []
    open_darts: list[int] = []
    edge_type_of_dart: dict[int, int] = {}
    face_type_list: list[int] = []
    loop_is_outer_list: list[int] = []
    n, n_faces = 0, 0
    it = iter(tokens)
    for kind, n_loops in it:
        assert kind == NLOOPS
        kind, ft = next(it)
        assert kind == FACE_TYPE
        face_type_list.append(ft)
        for _ in range(n_loops):
            kind, length = next(it)
            assert kind == LEN
            kind, outer = next(it)
            assert kind == LOOP_IS_OUTER
            loop_is_outer_list.append(outer)
            block = list(range(n, n + length))
            n += length
            phi += [block[(i + 1) % length] for i in range(length)]
            face_of_loop.append(n_faces)
            for d in block:
                kind, av = next(it)
                assert kind == ALPHA
                if av == OPEN:
                    open_darts.append(d)
                else:
                    j = d - av
                    alpha[d], alpha[j] = j, d
                    open_darts.remove(j)
                kind, et = next(it)
                assert kind == EDGE_TYPE
                edge_type_of_dart[d] = et
        n_faces += 1
    assert not open_darts, "unclosed darts"
    m = CMap(
        np.array([alpha[d] for d in range(n)], dtype=np.int64),
        np.array(phi, dtype=np.int64),
        np.array(face_of_loop, dtype=np.int64),
    )
    # Both darts of an edge were *encoded* with the same value (see module
    # docstring); a sampled sequence isn't guaranteed to agree, so a plain
    # scatter (later dart wins on disagreement) is the resolution here.
    edge_type = np.zeros(int(m.edge_of_dart.max()) + 1, dtype=np.int64)
    edge_type[m.edge_of_dart] = [edge_type_of_dart[d] for d in range(n)]
    face_type = np.array(face_type_list, dtype=np.int64)
    loop_is_outer = np.array(loop_is_outer_list, dtype=np.int64)
    return m, edge_type, face_type, loop_is_outer


# -------------------------------------------------------- sampling grammar

class Builder:
    """Incremental decoder used at sampling time. `allowed()` returns the
    admissible values for the next token, so a masked sampler can only ever
    produce a code that decodes to a valid map. FACE_TYPE/EDGE_TYPE/
    LOOP_IS_OUTER are always fully free choices (`range(N_*)`) -- none of
    them affect topological validity, only NLOOPS/LEN/ALPHA do."""

    def __init__(self, max_darts: int, max_loop_len: int, max_loops_per_face: int):
        self.max_darts = max_darts
        self.max_loop_len = max_loop_len
        self.max_loops_per_face = max_loops_per_face
        self.tokens: list[tuple[int, int]] = []
        self.open: list[int] = []
        self.n = 0
        self.loops_left = 0
        self.block_left = 0
        self.entry_dart = False
        self.need = NLOOPS
        self.done = False

    def _budget(self) -> int:
        return self.max_darts - self.n

    def allowed(self) -> tuple[int, list[int]]:
        """(kind, admissible values) for the next token."""
        if self.need == NLOOPS:
            # every extra loop needs at least one dart, plus room to close
            room = self._budget() - max(len(self.open), 1)
            return NLOOPS, list(range(1, max(min(self.max_loops_per_face, room + 1), 1) + 1))
        if self.need == FACE_TYPE:
            return FACE_TYPE, list(range(N_FACE_TYPES))
        if self.need == LEN:
            hi = self._budget() - (self.loops_left - 1) - max(len(self.open) - 1, 0)
            return LEN, list(range(1, max(min(self.max_loop_len, hi), 1) + 1))
        if self.need == LOOP_IS_OUTER:
            return LOOP_IS_OUTER, list(range(N_LOOP_IS_OUTER))
        if self.need == EDGE_TYPE:
            return EDGE_TYPE, list(range(N_EDGE_TYPES))
        if self.entry_dart and self.open:
            return ALPHA, [self.n - self.open[0]]          # forced BFS entry
        vals = [self.n - j for j in self.open]
        # OPEN is legal only if every dart left open can still be closed
        if len(self.open) + 1 <= self._budget() - 1:
            vals.insert(0, OPEN)
        return ALPHA, vals or [self.n - self.open[0]]

    def push(self, value: int) -> None:
        kind = self.need
        self.tokens.append((kind, value))
        if kind == NLOOPS:
            self.loops_left = value
            self.first_loop = True
            self.need = FACE_TYPE
        elif kind == FACE_TYPE:
            self.need = LEN
        elif kind == LEN:
            self.block_left = value
            self.entry_dart = self.first_loop and bool(self.open)
            self.first_loop = False
            self.need = LOOP_IS_OUTER
        elif kind == LOOP_IS_OUTER:
            self.need = ALPHA
        elif kind == ALPHA:
            if value == OPEN:
                self.open.append(self.n)
            else:
                self.open.remove(self.n - value)
            self.n += 1
            self.entry_dart = False
            self.need = EDGE_TYPE
        else:                                               # EDGE_TYPE
            self.block_left -= 1
            if self.block_left == 0:
                self.loops_left -= 1
                if self.loops_left:
                    self.need = LEN
                elif not self.open:
                    self.done = True
                else:
                    self.need = NLOOPS
            else:
                self.need = ALPHA

    def build(self):
        """(CMap, edge_type, face_type, loop_is_outer) -- see `decode`."""
        return decode(self.tokens)
