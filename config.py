"""Paths, capacities and hyperparameters.

Dataset: Fusion360 (the Fusion 360 Gallery reconstruction dataset), not
MFCAD++. MFCAD++ is a machining-feature *segmentation* benchmark -- fine for
de-risking the pipeline (data.py._stats, the smoke tests in the README), but
not an appropriate target for a generative B-rep paper. Fusion360 parts are
real designed CAD, which is what a reviewer expects to see samples of.

Capacities are p99 of the Fusion360 dart cache (`python -m cvpr_imperial.data
--stats --dataset fusion360`). Nothing here is a validity knob: validity is
structural, not thresholded -- these only bound the sequence/model shapes.
"""

import os
from pathlib import Path

ROOT = Path("/data_fast/users/maba")
RUN_DIR = Path(__file__).parent / "runs"      # redirected below when a vertex block is on

DATASETS = {
    "fusion360": {
        "root": ROOT / "data/fusion360",
        "step_dir": ROOT / "data/fusion360" / "step",
        "step_ext": ".stp",
        "dart_cache": ROOT / "data/fusion360" / "dartbrep_cache",   # built by extract_batch.py
        # built by extract.py. v2: edge/face type reindexed to the map's own
        # numbering (v1 saved them raw, misaligned) + loop_is_outer + frame.
        # v3: edges resampled at uniform arc length and stored as world-axis
        # chord deviations + delta (frame.py), replacing the rotated
        # local-frame + arc-length-scale representation ("edge_length" ->
        # "edge_delta"), incompatible with v2's cache.
        "geom_cache": ROOT / "data/fusion360" / "cvpr_geom_cache_v3",
    },
    "mfcad++": {
        "root": ROOT / "data/mfcad++",
        "step_dir": ROOT / "data/mfcad++" / "step",
        "step_ext": ".step",
        "dart_cache": ROOT / "data/mfcad++" / "dartbrep_cache",     # reused from dartbrep
        "geom_cache": ROOT / "data/mfcad++" / "cvpr_geom_cache",
    },
    "abc1m": {
        # ADSKAILab/ABC-1M on Hugging Face: per-solid .brep blobs in parquet
        # shards, not per-file STEP -- built by abc1m.py, not
        # build_dart_cache.py (see that module for why). "step_dir" holds
        # the unpacked per-shape .brep files under step/{split}/, the only
        # copy kept once a shard's parquet is deleted (abc1m.py's storage
        # policy: parquet is transient, never redundant with step_dir).
        "root": ROOT / "data/abc",
        "step_dir": ROOT / "data/abc" / "step",
        "step_ext": ".brep",
        "parquet_dir": ROOT / "data/abc" / "parquet",
        "dart_cache": ROOT / "data/abc" / "dartbrep_cache",
        "geom_cache": ROOT / "data/abc" / "cvpr_geom_cache",
    },
}
DATASET = "fusion360"

DATA_DIR = DATASETS[DATASET]["root"]
STEP_DIR = DATASETS[DATASET]["step_dir"]
STEP_EXT = DATASETS[DATASET]["step_ext"]
DART_CACHE = DATASETS[DATASET]["dart_cache"]
GEOM_CACHE = DATASETS[DATASET]["geom_cache"]

# kept for scripts that explicitly want the MFCAD++ smoke-test data
MFCAD_DIR = DATASETS["mfcad++"]["root"]

# ---- capacities (sequence model) ----
# Fusion360 p99 over a 5k-model sample (data.py --stats): darts 494 (max
# 1568), loop length 49 (max 80), loops per face 11 (max 21). MAX_DARTS is
# the binding cutoff -- parts above it are skipped by the loader; everything
# else follows from that with margin.
MAX_DARTS = 512
MAX_LOOP_LEN = 96
MAX_FACES = 192
MAX_LOOPS = 256
MAX_LOOPS_PER_FACE = 32    # inner loops of one face
# quick_encode token count, p99 1163 / max 1436 over the same sample --
# measured AFTER adding FACE_TYPE/EDGE_TYPE/LOOP_IS_OUTER (code.py), which
# roughly doubled tokens-per-structural-unit (a dart now costs an ALPHA +
# EDGE_TYPE pair, not one token; likewise LEN+LOOP_IS_OUTER per loop and
# NLOOPS+FACE_TYPE per face) versus the topology-only grammar this was
# originally sized for (was 768, p99 596 pre-type-tokens).
MAX_TOKENS = 1536
MAX_REL = 512              # pointer relative-distance embedding table

# ---- vertex block (stage 1 places vertices; stage 2 only edges/faces) ----
# None = topology only (vertices left to the geometry flow, the old split);
# "quant" = categorical per axis (vertex_head.QuantVertexHead);
# "flow"  = MAR-style continuous per-token head (vertex_head.FlowVertexHead).
VERTEX_MODE = {"none": None}.get(_m := os.environ.get("CVPR_VERTEX_MODE", "quant"), _m)
# Fraction of shapes with >= 2 distinct vertices collapsed into one cell, over
# 3k Fusion360 shapes: 6b 17.7%, 7b 9.0%, 8b 4.4%, 9b 2.3%, 10b 1.1% --
# roughly halving per bit; 10 is also BrepGPT's choice.
VERTEX_BITS = 10
VERTEX_LOSS_WEIGHT = 1.0
# topology tokens + one token per vertex (V p99 192, max 484)
MAX_SEQ = MAX_TOKENS + 512
# std of Gaussian noise on ground-truth vertices when conditioning the
# geometry flow in training (cascaded-diffusion conditioning augmentation):
# the flow is fed stage 1's *generated* vertices at sampling time, so it must
# not rely on them being exact. ~half a quantisation bin.
VERTEX_COND_NOISE = 0.5 / 2 ** VERTEX_BITS
# checkpoints/caches of the two architectures aren't interchangeable, so they
# don't share a directory; the edge/face VAEs are copied over (unchanged by
# where vertices come from)
if VERTEX_MODE is not None:
    RUN_DIR = RUN_DIR.parent / f"runs_vblock_{VERTEX_MODE}"

# ---- topology transformer ----
TOPO_DIM = 384
TOPO_LAYERS = 8
TOPO_HEADS = 6
COND_DROP = 0.1            # classifier-free guidance dropout on (genus, n_faces)

# ---- geometry model ----
N_CURVE_SAMPLES = 16       # 2 are deterministic (endpoints); 14 are the target
N_SURF_GRID = 8
GEOM_DIM = 384
GEOM_LAYERS = 6
GEOM_HEADS = 6
FLOW_STEPS = 50

# ---- geometry VAEs (per-cell latents the flow matches on, not raw geometry
# -- CLR-Wire's curve-VAE idea, adapted from a 1D-conv+cross-attn encoder to
# a plain MLP since our per-cell geometry is already a fixed-size vector) ----
VAE_LATENT_E = 8           # per-edge latent dim
VAE_LATENT_F = 16          # per-face latent dim
VAE_HIDDEN = 256
VAE_KL_WEIGHT = 1e-6       # matches CLR-Wire's curve VAE kl_weight

# ---- realization tolerances ----
PRIMITIVE_TOL = 1e-3       # fit residual below which a primitive replaces a B-spline
SEW_TOL = 1e-3
