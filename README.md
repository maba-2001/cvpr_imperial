# cvpr_imperial — generative B-reps on combinatorial maps

Topology is generated as an **oriented combinatorial map with a face
partition**, so every sample is a closed, connected, orientable 2-manifold *by
construction*. There are no validity penalties, no straight-through relaxations
of a boundary matrix, no rounding threshold and no repair heuristic, because
there is nothing left to repair.

## The representation

A B-rep shell is `M = (D, α, φ, Φ)`:

| symbol | meaning | cells |
|---|---|---|
| `D` | darts = coedges (one traversal of one edge by one loop) | — |
| `α` | fixed-point-free involution pairing an edge's two uses | edges = α-orbits |
| `φ` | permutation whose orbits are the wires | loops = φ-orbits |
| `σ = φ∘α` | rotation about a dart's origin vertex | vertices = σ-orbits |
| `Φ` | partition of loops into faces | faces = Φ-blocks |

Everything a penalty-based formulation enforces approximately is a theorem
here:

| enforced elsewhere as a penalty / ILP / repair | here |
|---|---|
| loop closure (`∂` of a wire vanishes) | orbits of a permutation are cycles |
| every edge used exactly twice | `α` is an involution |
| the two uses have opposite sign | `α` is fixed-point free; the map is oriented |
| vertex manifoldness (no pinch points) | a vertex **is** its σ-orbit, one cycle |
| recovering the wire's cyclic order | read off the φ-orbit |
| seam handling | a dart's α-partner may lie in its own loop |

`chi(S) = V - E + 2F - L`, because a face with `l` loops is an `l`-holed
sphere, not a disk. Capping every loop with a disk gives the map's surface with
`chi = V - E + L`; replacing a face's `l` caps by one `l`-holed sphere changes
`chi` by `2 - 2l`. Genus, Betti numbers and components follow exactly, per
component, with no GF(2) rank computation and no bridge construction.

## The canonical code — one object, three jobs

`code.py` walks the map from a root and numbers darts in discovery order. The
lexicographic minimum over an invariantly chosen root set is a canonical form,
so **`code_key(M) == code_key(M') iff M ≅ M'`**, and the same integer sequence
is the token stream the generator is trained on. Consequences:

* **No Hungarian matching.** Set prediction has nothing to match — the order is
  a property of the object, not a choice the loss has to discover.
* **Exact novelty and memorisation.** A set lookup on canonical codes, not a
  quantised geometry hash, and not an approximate WL signature with VF2 fallback.
* **Validity by construction.** The grammar's masks (`code.Builder.allowed`)
  admit exactly the token values that keep the partial map completable.

The walk is **face-major**, because `(α, φ)` alone is disconnected on real
B-reps: a pocket reaches the plate it sits in only through the inner loop of the
plate's top face, i.e. through `Φ`. Emitting a face's loops together restores
connectivity of the walk *and* makes `Φ` implicit — the face partition is the
block structure of the code, never a separate prediction to get wrong.

Grammar, one group per face, faces in discovery order:

```
NLOOPS k            number of loops of this face
  LEN l             darts in this loop
  ALPHA × l         0 = OPEN (partner comes later) | i-j = close open dart j
```

The first dart of each face group after the first is *forced* to close the
earliest open dart. That makes the walk breadth-first, makes the result
connected, and makes termination structural — the code ends exactly when a face
group leaves no dart open. There is no stop token to get wrong.

## Modules

| file | contents |
|---|---|
| `cmap.py` | the representation, orbit cells, exact invariants, the validity ladder |
| `code.py` | canonical code: labelling, isomorphism test, generation grammar |
| `build_dart_cache.py` | `dartbrep.extract_map` run over Fusion360's ~35,680 STEP files (dataset-generic; MFCAD++ already has one from `src/imperial/dartbrep`) |
| `data.py` | loads a dart cache for the configured dataset (`config.DATASET`); canonicalises splits |
| `topology_model.py` | causal transformer over the code; grammar-masked sampling |
| `extract.py` | reuses `dartbrep/extract_darts.extract_map`, adds sampled curves/surfaces |
| `geometry_model.py` | map message passing (gather along permutations) + rectified-flow geometry |
| `realize.py` | primitive fitting, orientation-derived outer loops, sewing, BRepCheck |
| `evaluate.py` | the ladder, exact novelty/uniqueness, conditioning accuracy, min-k |

Message passing on a map is a gather along a permutation: a dart's neighbours
are `α(d)`, `φ(d)`, `φ⁻¹(d)`, `σ(d)` and its face. No adjacency matrices, no
degree normalisation, no sign conventions — `MapConv` is eight lines.

## The validity ladder

Published "validity" numbers usually mean "OCCT did not crash". This separates:

* **tier 0 — structural.** `α` a fixed-point-free involution, `φ` a
  permutation, `Φ` a partition. *Guaranteed by construction*; measured as a
  regression guard.
* **tier 1 — topological.** Connected. For a connected oriented map
  `chi(S) ≤ chi(S_map) ≤ 2`, so genus ≥ 0 is automatic too; connectivity is
  enforced by the walk. *Also guaranteed*, also measured.
* **tier 2 — embedding.** Inner loops nested inside the outer loop in the
  face's own parameter domain. Genuinely contingent; checked in `realize.py`
  and reported, with periodic surfaces reported as unchecked rather than
  silently passed.
* **tier 3 — kernel.** Sewn into a closed shell, `BRepCheck_Analyzer` valid.

Tiers 0 and 1 being 100% is the point, not the result. The scientific content
moves to distribution quality, conditioning, and tiers 2–3.

## Geometry

`realize.py` fits the cheapest analytic primitive under tolerance — line,
circle, plane, cylinder — and falls back to a spline only when none fits.
Fitting a degree-3 B-spline through 16 samples of a straight edge is the main
reason kernel validity trails topological validity elsewhere.

Outer loops are **derived from orientation**, not from a bounding box: each
loop is projected into the face's parameter domain, the largest-area loop is
the outer one, and its traversal direction fixes the surface normal. That is
precisely the signed information the diffusion lineage discards.

Edges share one OCCT vertex per map vertex, so wires close exactly rather than
within a fit tolerance.

## Dataset

**Fusion360** (the Fusion 360 Gallery reconstruction dataset: 35,680 STEP
files, real designed CAD, not a segmentation benchmark) is the target dataset.
MFCAD++ (machining-feature segmentation, already cached by
`src/imperial/dartbrep`) is kept only for the cheap smoke tests referenced
below (`data.py --stats`, `data.py --build --dataset mfcad++`) — it is not an
appropriate distribution for a generative B-rep paper and no headline number
should come from it. `config.DATASET` picks the active one; `build_dart_cache.py`
builds a dart cache for any dataset in `config.DATASETS`.

## Status (measured, this machine)

* Canonical code: round-trip exact and invariant under random dart relabelling
  on the models tested; ~35 ms median per model after caching orbit
  computations, with a bounded tie-branching fallback for symmetric parts.
* Fusion360 dart extraction: `extract_darts.extract_map`, unchanged, reused
  via `build_dart_cache.py`; see the run log for yield and closed-manifold
  rate on this dataset.
* **An untrained, randomly-initialised model emits 100% tier-0/tier-1 valid
  maps.** That is the representation, not the training.
* Ground-truth realization round-trip on MFCAD++ (40 test models, the ceiling
  for the generated case, before the Fusion360 switch): 39/40 sewn, 28/40
  `BRepCheck`-valid. The gap is the geometric failure mode every method in
  this space inherits, and is reported separately from topological validity
  by design. To be re-measured on Fusion360.

## Running it

```bash
python -m cvpr_imperial.build_dart_cache --dataset fusion360   # map extraction (OCCT, once)
python -m cvpr_imperial.data --stats 400           # scope / capacity report
python -m cvpr_imperial.data --build train         # canonicalise a split (parallel)
python -m cvpr_imperial.train_topology --steps 40000
python -m cvpr_imperial.evaluate --n 512 --genus 2 --faces 20

python -m cvpr_imperial.extract --n 20000          # geometry cache (OCCT)
python -m cvpr_imperial.train_geometry --steps 30000
python -m cvpr_imperial.evaluate --n 256 --geometry
```

## Known limits

* Edges with one or three-plus face uses (open shells, non-manifold junctions)
  are out of scope: `α` would not be a fixed-point-free involution. Fusion360
  includes intermediate reconstruction-sequence states that are open shells,
  not solids -- these are filtered by `data.load_map`, not repaired. Weiler's
  radial structure generalises `α` to a k-cycle and the grammar would follow,
  but the surface is then not a manifold and `σ`-orbits are no longer vertices.
* Multi-shell solids (cavities) are excluded by the connectivity requirement.
* Sampling recomputes the full forward pass per token; a KV cache would make it
  linear.
* Nesting is unchecked on periodic surfaces.
* DeepCAD/ABC need only a new entry in `config.DATASETS` plus a
  `build_dart_cache.py --dataset` run; comparison against BrepGen / HoLa /
  DTGBrepGen is not implemented here.
