# cvpr_imperial — generative B-reps on combinatorial maps

Topology is generated as an **oriented combinatorial map with a face
partition**, so every sample is a closed, connected, orientable 2-manifold *by
construction*. There are no validity penalties, no straight-through relaxations
of a boundary matrix, no rounding threshold and no repair heuristic, because
there is nothing left to repair.

Generation is three stages, each conditioned on the last:

1. **Topology + primitive type + outer/inner role** — an autoregressive
   transformer over the canonical code (`topology/`).
2. **Vertex positions** — one token per vertex, appended to the same
   autoregressive sequence once the topology is complete (`topology/model.py`,
   `cfg.VERTEX_MODE`).
3. **Edge and face geometry** — a rectified flow conditioned on the map and
   the vertices stage 2 placed (`geometry/`).

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

`Φ` is the one thing this representation has that a plain rotation system
(`D, α, φ`) does not — see `topology/grammar.py`'s module docstring for why a
face with a hole needs it, and why `(α, φ)` alone is even *disconnected* on
real B-reps.

## The canonical code — one object, three jobs

`topology/grammar.py` walks the map from a root and numbers darts in discovery
order. The lexicographic minimum over an invariantly chosen root set is a
canonical form, so **`code_key(M) == code_key(M') iff M ≅ M'`**, and the same
integer sequence is the token stream the generator is trained on. Consequences:

* **No Hungarian matching.** Set prediction has nothing to match — the order is
  a property of the object, not a choice the loss has to discover.
* **Exact novelty and memorisation.** A set lookup on canonical codes, not a
  quantised geometry hash, and not an approximate WL signature with VF2 fallback.
* **Validity by construction.** The grammar's masks (`grammar.Builder.allowed`)
  admit exactly the token values that keep the partial map completable.

The walk is **face-major**, because `(α, φ)` alone is disconnected on real
B-reps: a pocket reaches the plate it sits in only through the inner loop of the
plate's top face, i.e. through `Φ`. Emitting a face's loops together restores
connectivity of the walk *and* makes `Φ` implicit — the face partition is the
block structure of the code, never a separate prediction to get wrong.

Grammar, one group per face, faces in discovery order:

```
NLOOPS k                  number of loops of this face
FACE_TYPE t                this face's surface kind (plane/cyl/cone/sphere/torus/other)
  LEN l                   darts in this loop
  LOOP_IS_OUTER o          is this loop the face's outer boundary?
  ALPHA × l               0 = OPEN (partner comes later) | i-j = close open dart j
  EDGE_TYPE t              this dart's edge kind (line/circle/other)
```

`FACE_TYPE`/`EDGE_TYPE`/`LOOP_IS_OUTER` ride along at fixed points in the same
walk that already determines topology, so the statistical grammar the model
imitates is over (topology, primitive type, outer/inner role) jointly — a
box's "perpendicular planar faces" pattern is a token in the stream it
imitates, not left for the geometry stage to rediscover. None of them affect
grammar legality (type never blocks a token the way ALPHA's pointer does), but
they do participate in the canonicalization walk itself: two candidates that
are topologically tied but differ in surface/edge kind or outer/inner role are
correctly told apart, not silently merged.

The first dart of each face group after the first is *forced* to close the
earliest open dart. That makes the walk breadth-first, makes the result
connected, and makes termination structural — the code ends exactly when a face
group leaves no dart open. There is no stop token to get wrong.

## The vertex block

After the topology is complete, one `VERTEX` token per vertex is appended, in
the map's own σ-orbit order — not interleaved into the topology walk, because
a vertex's identity isn't settled until its last incident dart is closed.
Each vertex token carries a *structural slot*: the mean positional embedding
of its incident darts' ALPHA tokens, so attention can retrieve exactly that
vertex's full incidence context before placing it.

`cfg.VERTEX_MODE` selects the per-token distribution
(`geometry/vertex_head.py`):

* `"quant"` — categorical, `cfg.VERTEX_BITS`-bit (default 10) per axis,
  factorised within the token as `p(x) p(y|x) p(z|x,y)` (PolyGen/BrepGPT-style
  quantization). A categorical puts point mass on exact-coincidence bins —
  same x as another vertex, coplanar with three others — which a continuous
  density assigns probability zero.
* `"flow"` — MAR-style continuous per-token diffusion/flow head. No
  quantization error, but no point masses either.

Measured on held-out Fusion360 shapes (≤ 15 faces, greedy decoding): `quant`
reproduces 83% of ground-truth axis-aligned edges and 98% of planar faces;
`flow` reproduces 4% and 33%. `quant` is the default. See
`scripts/eval_vertices.py`.

## Edge and face geometry

`geometry/frame.py` — each edge is resampled at uniform *arc length* first
(`data/extract.py`), so the same curve shape gives the same target regardless
of how the CAD kernel parameterized it, and a straight line lands exactly on
its chord. Every interior sample is then stored as its **world-axis deviation
from the chord point at the same arc fraction**, divided by `D = c + delta`:
`c` is the chord length (already known from the two endpoints stage 2 placed),
and `delta` — the one scalar that has to be predicted — is how far the curve
reaches beyond its chord: 0 for every line and every arc up to a semicircle,
and bounded even in the closed-edge limit (measured max |deviation| 0.997 over
the full cache, vs. a chord-relative scaling that blows up to 10¹²× the chord
on the same data). No rotation, no arc-length scalar, no roll convention to
pick for a frame axis — see `frame.py`'s module docstring for why an earlier
rotated-local-frame version was replaced.

`geometry/flow.py` conditions on the map via `MapConv` — message passing is a
gather along a permutation: a dart's neighbours are `α(d)`, `φ(d)`, `φ⁻¹(d)`,
`σ(d)` and its face. No adjacency matrices, no degree normalisation, no sign
conventions. Vertex positions from stage 2 are injected as per-dart Fourier
features of each edge's (origin, head) pair, so an edge sees its own oriented
endpoints and a face its own corners.

`geometry/realize.py` fits the cheapest analytic primitive under tolerance —
line, circle, plane, cylinder — and falls back to a spline only when none
fits. Outer loops are **derived from orientation**, not a bounding box: each
loop is projected into the face's parameter domain, the largest-area loop is
the outer one, and its traversal direction fixes the surface normal.

## Package layout

```
cvpr_imperial/
  config.py                capacities, paths, VERTEX_MODE
  topology/
    grammar.py              canonical code: labelling, isomorphism test, generation grammar
    model.py                 causal transformer over the code + vertex block; grammar-masked sampling
    signatures.py             exact canonical signatures of the training set, for novelty/memorisation
  geometry/
    frame.py                  chord-deviation edge sample representation
    flow.py                   map message passing + rectified-flow geometry
    vae.py                     per-cell latent VAE the flow matches instead of raw geometry
    vertex_head.py              quant / flow per-vertex-token heads
    realize.py                  primitive fitting, orientation-derived outer loops, sewing, BRepCheck
  data/
    cmap.py                     the representation itself: orbit cells, exact invariants, validity ladder
    dataset.py                   loads a dart cache for the configured dataset; canonicalises splits
    extract.py                    per-edge/per-face geometry cache (curves, surfaces, types)
    extract_darts.py               vendored dartbrep map extraction (phi from wire traversal, alpha from radial structure)
    normalize.py                    seam-splitting shape normalization shared by build_dart_cache.py and extract.py
    build_dart_cache.py              dart cache builder for any configured dataset
    abc1m.py                          ABC-1M (Hugging Face) ingestion
  scripts/
    train_topology.py, train_geometry.py, train_geometry_vae.py
    evaluate.py, eval_vertices.py
    viz_graph.py, viz_wireframe.py
```

## The validity ladder

Published "validity" numbers usually mean "OCCT did not crash". This separates:

* **tier 0 — structural.** `α` a fixed-point-free involution, `φ` a
  permutation, `Φ` a partition. *Guaranteed by construction*; measured as a
  regression guard.
* **tier 1 — topological.** Connected. For a connected oriented map
  `chi(S) ≤ chi(S_map) ≤ 2`, so genus ≥ 0 is automatic too; connectivity is
  enforced by the walk. *Also guaranteed*, also measured.
* **tier 2 — embedding.** Inner loops nested inside the outer loop in the
  face's own parameter domain. Genuinely contingent; checked in
  `geometry/realize.py` and reported, with periodic surfaces reported as
  unchecked rather than silently passed.
* **tier 3 — kernel.** Sewn into a closed shell, `BRepCheck_Analyzer` valid.

Tiers 0 and 1 being 100% is the point, not the result. The scientific content
moves to distribution quality, conditioning, and tiers 2-3.

## Dataset

**Fusion360** (the Fusion 360 Gallery reconstruction dataset: 35,680 STEP
files, real designed CAD, not a segmentation benchmark) is the primary
dataset. **ABC-1M** (`data/abc1m.py`, streamed from Hugging Face) is a second
source at much larger scale. MFCAD++ (machining-feature segmentation) is kept
only for cheap smoke tests (`dataset.py --stats`) — not an appropriate
distribution for a generative B-rep paper. `config.DATASET` picks the active
one; `build_dart_cache.py` builds a dart cache for any dataset in
`config.DATASETS`.

## Running it

```bash
python -m cvpr_imperial.data.build_dart_cache --dataset fusion360   # map extraction (OCCT, once)
python -m cvpr_imperial.data.dataset --stats 400        # scope / capacity report
python -m cvpr_imperial.data.dataset --build train       # canonicalise a split (parallel)
python -m cvpr_imperial.data.extract --n 20000            # geometry cache (OCCT)

python -m cvpr_imperial.scripts.train_topology --steps 40000
python -m cvpr_imperial.scripts.train_geometry_vae --steps 4000   # per-cell VAE, before the flow
python -m cvpr_imperial.scripts.train_geometry --steps 30000

python -m cvpr_imperial.scripts.evaluate --n 512 --genus 2 --faces 20
python -m cvpr_imperial.scripts.evaluate --n 256 --geometry
python -m cvpr_imperial.scripts.eval_vertices --n 256      # vertex-head structural metrics
python -m cvpr_imperial.scripts.viz_wireframe --n 6
```

`CVPR_VERTEX_MODE={quant,flow,none}` selects the vertex-block architecture and
redirects `config.RUN_DIR` to a matching `runs_vblock_<mode>/` (or plain
`runs/` for `none`, the pre-vertex-block architecture), so checkpoints and
caches for different architectures never collide.

## Known limits

* Edges with one or three-plus face uses (open shells, non-manifold junctions)
  are out of scope: `α` would not be a fixed-point-free involution. Fusion360
  includes intermediate reconstruction-sequence states that are open shells,
  not solids -- these are filtered by `dataset.load_map`, not repaired.
  Weiler's radial structure generalises `α` to a k-cycle and the grammar would
  follow, but the surface is then not a manifold and `σ`-orbits are no longer
  vertices.
* Multi-shell solids (cavities) are excluded by the connectivity requirement.
* Sampling recomputes the full forward pass per token; a KV cache would make it
  linear.
* Nesting is unchecked on periodic surfaces.
* The topology backbone is autoregressive; a non-autoregressive path (predict
  pairwise affinities for `α`/`φ`, project to an exact fixed-point-free
  involution / permutation via a matching algorithm, DETR-style Hungarian
  training loss) is a real option that preserves tier-0 validity for free and
  is not yet implemented.
