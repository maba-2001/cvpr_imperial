"""Stage 2: geometry conditioned on a map.

Message passing on a combinatorial map is a gather along a permutation: a
dart's neighbours are alpha(d), phi(d), phi^-1(d), sigma(d), and its face.
Cell features are pooled from darts over their orbits (vertex = sigma-orbit,
edge = alpha-orbit, face = the face block), so the encoder is exactly as
expressive as the representation and no more.

The decoder is a rectified flow over per-cell geometry -- vertex positions,
sampled edge curves, sampled face surfaces -- conditioned on those cell
embeddings. Cells are ordered canonically by `code.canonical`, so, as in
stage 1, there is no set-matching problem to solve.

Batching across maps of different sizes follows PyTorch Geometric's graph-
batching trick: darts and cells of every map are concatenated with per-map
index offsets, so a cell id stays globally unique across the batch and
gather/scatter-mean message passing needs no further changes -- see
`map_tensors_batch`. Only the flow's attention (which must not let one map's
tokens attend to another's) needs an explicit pad + key-padding-mask step,
built by `_pad`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import code as C
from . import config as cfg
from .vertex_head import FourierEmbed

DIMS = {"v": 3, "e": (cfg.N_CURVE_SAMPLES - 2) * 3 + 1, "f": cfg.N_SURF_GRID ** 2 * 3}
RANKS = ("v", "e", "f")


def compute_scales(files: list, vaes: dict | None, device: str = "cpu",
                    n_files: int = 500) -> dict[str, float]:
    """Per-rank std of what `GeometryFlow` actually flow-matches (raw xyz for
    "v", VAE latent mu for "e"/"f"), over a sample of the geometry cache --
    see `GeometryFlow`'s `scales` argument for why this matters."""
    vs, es, fs = [], [], []
    for fp in files[:n_files]:
        d = np.load(fp)
        vs.append(d["vertex_xyz"].reshape(-1, 3))
        es.append(np.concatenate([d["edge_curve"].reshape(len(d["edge_curve"]), -1),
                                   d["edge_delta"][:, None]], axis=1))
        fs.append(d["face_surface"].reshape(len(d["face_surface"]), -1))
    raw = {"v": np.concatenate(vs), "e": np.concatenate(es), "f": np.concatenate(fs)}
    scales = {}
    for r, arr in raw.items():
        x = torch.from_numpy(arr.astype(np.float32)).to(device)
        if vaes is not None and r in vaes:
            with torch.no_grad():
                x, _ = vaes[r].encode(x)
        scales[r] = float(x.std().clamp(min=1e-6))
    return scales


def map_tensors(m, device="cpu", edge_type=None, face_type=None,
                loop_is_outer=None) -> dict:
    """Permutations and orbit ids of a single CMap as tensors. `edge_type`/
    `face_type`/`loop_is_outer`: optional (n_e,)/(n_f,)/(n_l,) class-index
    arrays in `m`'s own numbering (see `cmap.reindex_cells`), fed to
    `MapEncoder` as structural conditioning -- known before any geometry is
    generated, either from the training data's ground truth or, at sampling
    time, from stage 1's own FACE_TYPE/EDGE_TYPE/LOOP_IS_OUTER tokens."""
    phi_inv = np.empty_like(m.phi)
    phi_inv[m.phi] = np.arange(m.n_darts)
    t = {"alpha": m.alpha, "phi": m.phi, "phi_inv": phi_inv, "sigma": m.sigma,
         "v": m.vertex_of_dart, "e": m.edge_of_dart, "f": m.face_of_dart,
         "loop": m.loop_of_dart}
    out = {k: torch.as_tensor(v, dtype=torch.long, device=device) for k, v in t.items()}
    if edge_type is not None:
        out["edge_type"] = torch.as_tensor(edge_type, dtype=torch.long, device=device)
    if face_type is not None:
        out["face_type"] = torch.as_tensor(face_type, dtype=torch.long, device=device)
    if loop_is_outer is not None:
        out["loop_is_outer"] = torch.as_tensor(loop_is_outer, dtype=torch.long, device=device)
    return out


def map_tensors_batch(maps: list, device="cpu", edge_types: list | None = None,
                      face_types: list | None = None,
                      loop_is_outers: list | None = None,
                      vertex_pos: list | None = None) -> dict:
    """Batched permutations/orbit ids for several maps -- the disjoint-union
    trick: darts and cells from every map are concatenated with per-map index
    offsets, so a permutation still only ever points within its own map (the
    offset is added to both the array and the values it holds), and an orbit
    id stays globally unique across the batch. `MapConv`/`MapEncoder` need no
    batch-awareness at all as a result; only the flow's attention does, via
    the `counts` this returns (see `_pad`).

    `edge_types`/`face_types`/`loop_is_outers`: optional per-map lists of
    class-index arrays (see `map_tensors`) -- plain concatenation, no
    offset, since class ids share one vocabulary across the whole batch
    rather than being per-map identities.
    """
    d_off = v_off = e_off = f_off = l_off = 0
    alpha, phi, sigma, vod, eod, fod, lod = [], [], [], [], [], [], []
    counts = []  # (n_v, n_e, n_f) per map, in batch order
    for m in maps:
        n_v, n_e, n_f, n_l = m.counts()
        alpha.append(m.alpha + d_off)
        phi.append(m.phi + d_off)
        sigma.append(m.sigma + d_off)
        vod.append(m.vertex_of_dart + v_off)
        eod.append(m.edge_of_dart + e_off)
        fod.append(m.face_of_dart + f_off)
        lod.append(m.loop_of_dart + l_off)
        counts.append((n_v, n_e, n_f))
        d_off += m.n_darts
        v_off += n_v
        e_off += n_e
        f_off += n_f
        l_off += n_l
    phi_cat = np.concatenate(phi)
    phi_inv = np.empty_like(phi_cat)
    phi_inv[phi_cat] = np.arange(d_off)
    t = {"alpha": np.concatenate(alpha), "phi": phi_cat, "phi_inv": phi_inv,
         "sigma": np.concatenate(sigma), "v": np.concatenate(vod),
         "e": np.concatenate(eod), "f": np.concatenate(fod),
         "loop": np.concatenate(lod)}
    out = {k: torch.as_tensor(v, dtype=torch.long, device=device) for k, v in t.items()}
    out["counts"] = counts
    if edge_types is not None:
        out["edge_type"] = torch.as_tensor(np.concatenate(edge_types), dtype=torch.long,
                                           device=device)
    if face_types is not None:
        out["face_type"] = torch.as_tensor(np.concatenate(face_types), dtype=torch.long,
                                           device=device)
    if loop_is_outers is not None:
        out["loop_is_outer"] = torch.as_tensor(np.concatenate(loop_is_outers), dtype=torch.long,
                                               device=device)
    if vertex_pos is not None:
        # (V_i, 3) per map, concatenated in the same order as the vertex
        # offsets above, so t["v"] indexes straight into it
        out["vpos"] = torch.cat([torch.as_tensor(p, dtype=torch.float32, device=device)
                                 for p in vertex_pos])
    return out


def _unpad(padded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Inverse of `_pad`: boolean-indexing a (B, max_n, D) tensor row-major by
    its (B, max_n) validity mask recovers the (N_total, D) flat tensor in the
    same per-map concatenation order `map_tensors_batch` used, since `_pad`
    left exactly the first `count` positions of each row set."""
    return padded[mask]


def _pad(flat: torch.Tensor, counts: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(N_total, D) concatenated in map order + per-map counts -> (B, max_n, D)
    zero-padded and (B, max_n) validity mask."""
    parts = torch.split(flat, counts)
    padded = nn.utils.rnn.pad_sequence(parts, batch_first=True)
    B, max_n = len(counts), padded.shape[1]
    mask = torch.zeros(B, max_n, dtype=torch.bool, device=flat.device)
    for i, c in enumerate(counts):
        mask[i, :c] = True
    return padded, mask


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(n, src.shape[-1], device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    cnt = torch.zeros(n, 1, device=src.device, dtype=src.dtype)
    cnt.index_add_(0, index, torch.ones_like(src[:, :1]))
    return out / cnt.clamp(min=1)


class MapConv(nn.Module):
    """One layer of message passing along the map's permutations."""

    def __init__(self, d: int):
        super().__init__()
        self.perm = nn.ModuleDict(
            {p: nn.Linear(d, d) for p in ("alpha", "phi", "phi_inv", "sigma")})
        self.face = nn.Linear(d, d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h: torch.Tensor, t: dict) -> torch.Tensor:
        agg = sum(lin(h[t[p]]) for p, lin in self.perm.items())
        agg = agg + self.face(_scatter_mean(h, t["f"], int(t["f"].max()) + 1))[t["f"]]
        h = self.norm1(h + agg)
        return self.norm2(h + self.ffn(h))


class MapEncoder(nn.Module):
    """Darts -> per-cell context embeddings.

    Batch-agnostic: whether `t` describes one map or several concatenated
    with `map_tensors_batch`, this is exactly the same gather/scatter-mean
    computation -- cell ids are globally unique either way.

    When `t` carries `edge_type`/`face_type`/`loop_is_outer` (see
    `map_tensors_batch`), their embeddings are added to every dart of the
    corresponding cell before message passing -- known-before-geometry
    structural conditioning (a box's "these faces are planar and meet at
    right angles" is a token the encoder can read directly, likewise "this
    loop is the hole, not the boundary"), not something the flow has to
    infer purely from the map's abstract combinatorics. loop_is_outer isn't
    gated by a rank the way edge/face type are (no "loop" rank exists to
    gate on -- loops are an intermediate cell between edges and faces, not
    one of v/e/f), so its embedding is always created; it's a no-op to skip
    passing it, same as the others.
    """

    def __init__(self, d: int = cfg.GEOM_DIM, layers: int = cfg.GEOM_LAYERS,
                 ranks: tuple = RANKS, n_face_types: int = C.N_FACE_TYPES,
                 n_edge_types: int = C.N_EDGE_TYPES,
                 n_loop_outer: int = C.N_LOOP_IS_OUTER, vertex_cond: bool = False):
        super().__init__()
        self.ranks = ranks
        self.embed = nn.Parameter(torch.randn(d) * 0.02)
        self.layers = nn.ModuleList(MapConv(d) for _ in range(layers))
        self.rank_emb = nn.ParameterDict(
            {r: nn.Parameter(torch.randn(d) * 0.02) for r in ranks})
        self.face_type_emb = nn.Embedding(n_face_types, d) if "f" in ranks else None
        self.edge_type_emb = nn.Embedding(n_edge_types, d) if "e" in ranks else None
        self.loop_outer_emb = nn.Embedding(n_loop_outer, d)
        if vertex_cond:
            self.vpos = FourierEmbed(d)
            self.vpair = nn.Linear(2 * d, d)

    def forward(self, t: dict) -> dict:
        h = self.embed.expand(t["alpha"].shape[0], -1).contiguous()
        if self.face_type_emb is not None and "face_type" in t:
            h = h + self.face_type_emb(t["face_type"][t["f"]])
        if self.edge_type_emb is not None and "edge_type" in t:
            h = h + self.edge_type_emb(t["edge_type"][t["e"]])
        if "loop_is_outer" in t:
            h = h + self.loop_outer_emb(t["loop_is_outer"][t["loop"]])
        if "vpos" in t:
            # each dart sees its (origin, head) vertex pair -- after message
            # passing an edge knows its oriented endpoints, a face its corners
            p = self.vpos(t["vpos"])
            h = h + self.vpair(torch.cat([p[t["v"]], p[t["v"][t["alpha"]]]], dim=-1))
        for layer in self.layers:
            h = layer(h, t)
        return {r: _scatter_mean(h, t[r], int(t[r].max()) + 1) + self.rank_emb[r]
                for r in self.ranks}


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=1)


class DiTBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        s1, b1, g1, s2, b2, g2 = self.ada(c).unsqueeze(1).chunk(6, dim=2)
        h = self.norm1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, h, h, key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        h = self.norm2(x) * (1 + s2) + b2
        return x + g2 * self.ffn(h)


class VNLinear(nn.Module):
    """Vector Neurons (Deng et al. 2021) linear layer: mixes vector CHANNELS
    with an ordinary Linear, never touching the 3-vector axis itself. Since
    the mixing weights don't depend on the vectors' orientation, this
    commutes with any rotation applied to the input -- exactly equivariant,
    no canonicalization needed."""

    def __init__(self, c_in: int, c_out: int, bias: bool = False):
        super().__init__()
        self.lin = nn.Linear(c_in, c_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., C_in, 3) -> (..., C_out, 3)."""
        return self.lin(x.transpose(-1, -2)).transpose(-1, -2)


class VNLeakyReLU(nn.Module):
    """Vector Neurons nonlinearity: a learned per-channel direction (itself
    an equivariant linear map of the input, so it rotates along with x), and
    the vector is only allowed to flip toward that direction, never rotated
    independently of it -- a nonlinearity that treats a vector as a vector,
    not as 3 independent scalars."""

    def __init__(self, c: int, slope: float = 0.2):
        super().__init__()
        self.dir = VNLinear(c, c)
        self.slope = slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.dir(x)
        d = d / (d.norm(dim=-1, keepdim=True) + 1e-6)
        dot = (x * d).sum(-1, keepdim=True)
        return torch.where(dot >= 0, x, x - (1 - self.slope) * dot * d)


class VNScalarGate(nn.Module):
    """Injects (rotation-invariant) scalar conditioning into a vector-channel
    stream. Scaling a vector by an invariant scalar is still equivariant --
    only adding an independent vector or mixing in orientation-dependent
    terms would break it -- so this is the safe way for the topology
    context (which never sees coordinates, hence is trivially invariant) to
    steer the per-vertex correction without compromising equivariance."""

    def __init__(self, ctx_dim: int, c: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(ctx_dim, c), nn.SiLU(), nn.Linear(c, c))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)   # starts at scale 1 (no-op)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        scale = self.mlp(ctx).unsqueeze(-1)          # (..., C, 1)
        return x * (1 + scale)


class VNAttention(nn.Module):
    """Cross-vertex equivariant attention (VN-Transformer, Assaad et al.
    2022): attention WEIGHTS come from squared distances between equivariant
    query/key vectors -- ||R.qk_i - R.qk_j||^2 = ||qk_i - qk_j||^2 for any
    rotation R, so the weights are exactly invariant -- while the aggregated
    VALUES stay literal vectors, and an invariant-weighted sum of vectors is
    itself equivariant. This is what lets one vertex's correction actually
    depend on every other vertex's position -- the relational-structure
    mechanism the plain per-vertex VN layers above don't have on their own.
    """

    def __init__(self, c: int):
        super().__init__()
        self.to_qk = VNLinear(c, c)
        self.to_v = VNLinear(c, c)
        self.scale = c ** 0.5

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x: (B, N, C, 3). valid: (B, N) bool. Returns (B, N, C, 3)."""
        qk = self.to_qk(x).flatten(-2, -1)            # (B, N, C*3)
        v = self.to_v(x)
        dist2 = torch.cdist(qk, qk) ** 2               # (B, N, N), invariant
        bias = torch.zeros_like(dist2)
        pair_invalid = ~(valid.unsqueeze(2) & valid.unsqueeze(1))
        # A finite (not inf) penalty: a fully-padded query row has every key
        # masked too, and inf-inf/softmax-of-all-inf is NaN, which a
        # downstream *0 mask multiply would NOT zero out (NaN*0 is NaN) --
        # silently corrupting the loss for the whole batch. A large finite
        # value instead gives a harmless uniform softmax on those dead rows.
        bias.masked_fill_(pair_invalid, 1e9)
        attn = torch.softmax(-dist2 / self.scale - bias, dim=-1)
        return torch.einsum("bij,bjcd->bicd", attn, v)


class VNVertexHead(nn.Module):
    """Equivariant correction to vertex positions: lift the raw (vector)
    position to several vector channels, alternate VN-Attention (cross-
    vertex, relational) with VN-Linear+VNLeakyReLU+VNScalarGate (per-vertex,
    steered by the invariant topology context), project back to one channel.
    Zero-init output, same no-op-at-start convention as the rest of this
    file's additive corrections."""

    def __init__(self, ctx_dim: int, channels: int = 16, layers: int = 3):
        super().__init__()
        self.lift = VNLinear(1, channels)
        self.attn = nn.ModuleList(VNAttention(channels) for _ in range(layers))
        self.lin = nn.ModuleList(VNLinear(channels, channels) for _ in range(layers))
        self.act = nn.ModuleList(VNLeakyReLU(channels) for _ in range(layers))
        self.gate = nn.ModuleList(VNScalarGate(ctx_dim, channels) for _ in range(layers))
        self.proj = VNLinear(channels, 1)
        nn.init.zeros_(self.proj.lin.weight)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x: (B, N, 3) positions. ctx: (B, N, ctx_dim) invariant scalar
        context. valid: (B, N) bool. Returns (B, N, 3)."""
        h = self.lift(x.unsqueeze(-2))                 # (B, N, C, 3)
        for attn, lin, act, gate in zip(self.attn, self.lin, self.act, self.gate):
            h = h + attn(h, valid)
            h = gate(act(lin(h)), ctx)
        return self.proj(h).squeeze(-2)


class EGNNCoordUpdate(nn.Module):
    """EGNN-style (Satorras et al. 2021) relative-vector coordinate update.

    A vertex token's only route to "this shape is a box" is dense attention
    over absolute coordinates -- nothing makes perpendicularity, parallelism
    or coplanarity between vertices an easy quantity to represent, so the
    network has to rediscover such relations from scratch per shape. Here
    the update to vertex i is a learned scalar-weighted sum of its relative
    vectors to every other vertex in the same map:

        delta_i = sum_j w_ij * (x_i - x_j),   w_ij = MLP(h_i, h_j, |x_i-x_j|^2)

    Built entirely from relative vectors, this is exactly rotation- and
    translation-equivariant (no canonicalization needed), and -- the actual
    point -- dot/cross products between the relative vectors the network
    already holds are exactly what perpendicularity/parallelism/coplanarity
    *are*, so those relations become directly representable instead of
    something dense attention over absolute coordinates must approximate.
    """

    def __init__(self, d: int, pair_dim: int = 48):
        super().__init__()
        # Project down to `pair_dim` before the O(N^2) pairwise concat --
        # at the full hidden width (e.g. 384) a single batch's (B,N,N,2D+1)
        # intermediate is several GiB and OOMs; EGNN-style implementations
        # standardly use a narrow edge-message width for exactly this reason.
        self.proj = nn.Linear(d, pair_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * pair_dim + 1, pair_dim), nn.SiLU(), nn.Linear(pair_dim, 1))
        # Zero-init, same convention as elsewhere: starts as a no-op.
        nn.init.zeros_(self.edge_mlp[-1].weight)
        nn.init.zeros_(self.edge_mlp[-1].bias)

    def forward(self, h: torch.Tensor, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """h: (B, N, D) vertex hidden states. x: (B, N, 3) current (noisy)
        positions. valid: (B, N) bool mask. Returns (B, N, 3)."""
        n = h.shape[1]
        hp = self.proj(h)                                              # (B, N, pair_dim)
        diff = x.unsqueeze(2) - x.unsqueeze(1)                       # (B, N, N, 3)
        dist2 = (diff ** 2).sum(-1, keepdim=True)                     # (B, N, N, 1)
        hi = hp.unsqueeze(2).expand(-1, -1, n, -1)
        hj = hp.unsqueeze(1).expand(-1, n, -1, -1)
        w = self.edge_mlp(torch.cat([hi, hj, dist2], dim=-1))         # (B, N, N, 1)
        not_self = ~torch.eye(n, device=h.device, dtype=torch.bool)
        pair_valid = (valid.unsqueeze(2) & valid.unsqueeze(1) & not_self).unsqueeze(-1).float()
        w = w * pair_valid
        denom = pair_valid.sum(dim=2).clamp(min=1.0)                  # (B, N, 1)
        return (w * diff).sum(dim=2) / denom


class GeometryFlow(nn.Module):
    """Rectified flow over the cells' geometry, conditioned on the map.

    `boundary_gather`: dense self-attention over every cell gives a face
    token no explicit signal for which edges are its own boundary versus
    another face's. Measured effect: generated edges sit ~0.5-0.6 units from
    their own face's fitted surface, on a face spanning only ~0.1 units
    (see realize.py's face-trim collapse). When true, this adds an explicit
    structural gather mirroring `MapConv.face` -- scatter-mean each face's
    boundary darts' current edge tokens into its own token, every flow step.
    Kept togglable for A/B comparison against the baseline.
    """

    def __init__(self, d: int = cfg.GEOM_DIM, boundary_gather: bool = True,
                 ranks: tuple | None = None, egnn: bool = False, vn: bool = False,
                 vaes: dict | None = None, scales: dict | None = None,
                 vertex_cond: bool = cfg.VERTEX_MODE is not None):
        super().__init__()
        # vertex_cond: stage 1 already placed the vertices (cfg.VERTEX_MODE),
        # so they're conditioning here, not a rank to generate
        if ranks is None:
            ranks = ("e", "f") if vertex_cond else RANKS
        if vertex_cond and "v" in ranks:
            raise ValueError("vertex_cond: vertices are given, not generated")
        self.ranks = ranks
        self.vertex_cond = vertex_cond
        # vaes: optional {rank: CellVAE}, frozen, one per latent rank (see
        # geometry_vae.py) -- when given, the flow matches each such rank's
        # *latent* (mu of the VAE's posterior) instead of its raw geometry;
        # ranks absent from `vaes` (vertices, always) still flow on raw xyz.
        self.vaes = nn.ModuleDict(vaes) if vaes else None
        if self.vaes is not None:
            for vae in self.vaes.values():
                vae.requires_grad_(False)
        self.flow_dim = {r: (self.vaes[r].latent_dim if self.vaes and r in self.vaes else DIMS[r])
                          for r in ranks}
        # scales: per-rank std of what's actually flow-matched (raw xyz for
        # "v", VAE latent mu for "e"/"f") -- divided out before noising so
        # every rank sits at the same ~unit variance the N(0,1) noise prior
        # does. Registered as buffers (not plain floats) so a checkpoint
        # carries its own calibration and inference never has to recompute
        # or guess it; `compute_scales` does the one-time measurement at
        # train start. Without this, a rank whose natural/latent scale is
        # larger than the noise's (our VAE latents are ~1.6-2.9 std vs.
        # vertex xyz's ~0.27) dominates the shared network's gradient and
        # starves the other ranks' training signal.
        for r in ranks:
            s = float(scales[r]) if scales and r in scales else 1.0
            self.register_buffer(f"scale_{r}", torch.tensor(s))
        self.encoder = MapEncoder(d, ranks=ranks, vertex_cond=vertex_cond)
        self.inp = nn.ModuleDict({r: nn.Linear(self.flow_dim[r], d) for r in ranks})
        self.out = nn.ModuleDict({r: nn.Linear(d, self.flow_dim[r]) for r in ranks})
        for r in ranks:
            nn.init.zeros_(self.out[r].weight)
            nn.init.zeros_(self.out[r].bias)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(DiTBlock(d, cfg.GEOM_HEADS) for _ in range(cfg.GEOM_LAYERS))
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.d = d
        # boundary_gather pools each face's boundary edges -- meaningless
        # without both ranks present.
        self.boundary_gather = boundary_gather and "f" in ranks and "e" in ranks
        if self.boundary_gather:
            self.boundary = nn.Linear(d, d)
            # Zero-init, same convention as `out[r]`: the new term starts as
            # a no-op (face tokens identical to the no-gather baseline at
            # step 0) so it can only help once training turns it on, instead
            # of injecting an untrained random signal into every face token
            # from the first step.
            nn.init.zeros_(self.boundary.weight)
            nn.init.zeros_(self.boundary.bias)
        # egnn: relative-vector coordinate update for vertices -- see
        # EGNNCoordUpdate's docstring. Meaningless without "v".
        self.egnn = egnn and "v" in ranks
        if self.egnn:
            self.egnn_head = EGNNCoordUpdate(d)
        # vn: equivariant correction, conditioned on `ctx["v"]` specifically
        # (not the post-attention hidden state, which has absorbed absolute
        # coordinate information through many rounds of dense attention and
        # so is no longer provably rotation-invariant) -- ctx comes straight
        # from MapEncoder, which never sees geometry at all.
        self.vn = vn and "v" in ranks
        if self.vn:
            self.vn_head = VNVertexHead(ctx_dim=d)

    def scale(self, r: str) -> torch.Tensor:
        return getattr(self, f"scale_{r}")

    @classmethod
    def load(cls, path, vaes: dict | None = None, device: str = "cuda") -> "GeometryFlow":
        """Rebuild with the ranks and vertex conditioning the checkpoint was
        trained with (read off its parameter names), then load it."""
        state = torch.load(path, weights_only=True, map_location=device)
        ranks = tuple(r for r in RANKS if f"out.{r}.weight" in state)
        net = cls(vaes=vaes, ranks=ranks,
                  vertex_cond=any(k.startswith("encoder.vpos.") for k in state))
        net.load_state_dict(state)
        return net.to(device).eval()

    def velocity(self, x: dict, t: torch.Tensor, ctx: dict, mask: dict,
                 dart_e: torch.Tensor | None = None, dart_f: torch.Tensor | None = None,
                 counts: list | None = None) -> dict:
        """x/ctx[r]: (B, max_n_r, ·) padded. t: (B,). mask[r]: (B, max_n_r).
        dart_e/dart_f/counts: only needed when `boundary_gather` is on --
        dart-level (edge-of-dart, face-of-dart), globally offset the same way
        `map_tensors_batch` offsets everything else, plus the per-map
        (n_v, n_e, n_f) counts used to re-pad the per-face result."""
        tok = {r: self.inp[r](x[r]) + ctx[r] for r in self.ranks}
        if self.boundary_gather:
            e_flat = _unpad(tok["e"], mask["e"])              # (n_e_total, d)
            n_f = int(dart_f.max()) + 1
            boundary = _scatter_mean(e_flat[dart_e], dart_f, n_f)  # (n_f_total, d)
            boundary_p, _ = _pad(self.boundary(boundary), [c[2] for c in counts])
            tok["f"] = tok["f"] + boundary_p

        tokens = torch.cat([tok[r] for r in self.ranks], dim=1)
        valid = torch.cat([mask[r] for r in self.ranks], dim=1)
        c = self.t_mlp(timestep_embedding(t, self.d))
        h = tokens
        for blk in self.blocks:
            h = blk(h, c, key_padding_mask=~valid)
        h = self.norm(h)
        sizes = [x[r].shape[1] for r in self.ranks]
        parts = dict(zip(self.ranks, h.split(sizes, dim=1)))
        out = {r: self.out[r](p) for r, p in parts.items()}
        if self.egnn:
            out["v"] = out["v"] + self.egnn_head(parts["v"], x["v"], mask["v"])
        if self.vn:
            out["v"] = out["v"] + self.vn_head(x["v"], ctx["v"], mask["v"])
        return out

    def loss(self, geoms: list[dict], maps: list, edge_types: list | None = None,
             face_types: list | None = None,
             loop_is_outers: list | None = None) -> torch.Tensor:
        """geoms[i][r]: (n_r_i, DIMS[r]) geometry of map i, unpadded. One
        random flow timestep per map (standard batched rectified-flow
        training), loss averaged over real (non-pad) cells only.
        `edge_types`/`face_types`/`loop_is_outers`: optional per-map
        class-index arrays (see `map_tensors_batch`), the training data's
        ground-truth primitive types and outer/inner loop roles. With
        `vertex_cond`, geoms[i]["v"] is the ground-truth conditioning,
        jittered by cfg.VERTEX_COND_NOISE: at sampling time these vertices
        come from stage 1, so the flow must not rely on them being exact."""
        device = geoms[0]["v"].device
        vpos = None
        if self.vertex_cond:
            vpos = [g["v"] + cfg.VERTEX_COND_NOISE * torch.randn_like(g["v"]) for g in geoms]
        t_map = map_tensors_batch(maps, device, edge_types, face_types, loop_is_outers, vpos)
        counts = t_map["counts"]
        ctx = self.encoder(t_map)

        B = len(maps)
        t = torch.rand(B, device=device)
        x, geom_p, noise_p, mask = {}, {}, {}, {}
        for r in self.ranks:
            flat_geom = torch.cat([g[r] for g in geoms], dim=0)
            if self.vaes is not None and r in self.vaes:
                with torch.no_grad():
                    flat_geom, _ = self.vaes[r].encode(flat_geom)   # latent mean, LDM-style
            flat_geom = flat_geom / self.scale(r)
            flat_noise = torch.randn_like(flat_geom)
            counts_r = [c[RANKS.index(r)] for c in counts]  # counts is always the full (v,e,f) triple
            geom_p[r], mask[r] = _pad(flat_geom, counts_r)
            noise_p[r], _ = _pad(flat_noise, counts_r)
            ctx[r], _ = _pad(ctx[r], counts_r)
            tt = t.view(B, 1, 1)
            x[r] = (1 - tt) * noise_p[r] + tt * geom_p[r]

        v = self.velocity(x, t, ctx, mask, t_map["e"], t_map["f"], counts)
        total, n_terms = 0.0, 0
        for r in self.ranks:
            m = mask[r].unsqueeze(-1).float()
            diff = (v[r] - (geom_p[r] - noise_p[r])) ** 2
            total = total + (diff * m).sum() / (m.sum() * self.flow_dim[r]).clamp(min=1)
            n_terms += 1
        return total / n_terms

    @torch.no_grad()
    def sample(self, m, edge_type=None, face_type=None, loop_is_outer=None,
               vertices=None, device="cuda", steps: int = cfg.FLOW_STEPS) -> dict:
        """Single-map convenience wrapper around `sample_batch`."""
        wrap = lambda a: None if a is None else [a]
        return self.sample_batch([m], wrap(edge_type), wrap(face_type), wrap(loop_is_outer),
                                 wrap(vertices), device, steps)[0]

    @torch.no_grad()
    def sample_batch(self, maps: list, edge_types: list | None = None,
                      face_types: list | None = None,
                      loop_is_outers: list | None = None,
                      vertex_pos: list | None = None, device="cuda",
                      steps: int = cfg.FLOW_STEPS) -> list[dict]:
        """One flow integration for the whole pool of maps at once, instead
        of a Python loop of B=1 calls -- each `velocity` call is overhead-
        dominated at B=1 (see viz_wireframe.py's profiling: ~350ms/call
        regardless of the network's actual size), so batching turns N
        sequential tiny forward passes into one appropriately-sized one.
        `map_tensors_batch`/`_pad`'s key-padding-mask already makes the
        underlying network batch-agnostic; this just stops calling it with
        B=1 every time.

        With `vertex_cond`, `vertex_pos` (per-map (V, 3), from stage 1) is
        required and is passed through as each result's "v", so callers get
        one complete geometry dict either way.
        """
        if self.vertex_cond and vertex_pos is None:
            raise ValueError("vertex_cond flow needs stage 1's vertex positions")
        t_map = map_tensors_batch(maps, device, edge_types, face_types, loop_is_outers,
                                  vertex_pos if self.vertex_cond else None)
        counts = t_map["counts"]
        ctx = self.encoder(t_map)
        mask = {}
        for r in self.ranks:
            counts_r = [c[RANKS.index(r)] for c in counts]
            ctx[r], mask[r] = _pad(ctx[r], counts_r)
        B = len(maps)
        x = {r: torch.randn(B, ctx[r].shape[1], self.flow_dim[r], device=device)
             for r in self.ranks}
        t1 = torch.zeros(B, device=device)
        for i in range(steps):
            t1.fill_(i / steps)
            v = self.velocity(x, t1, ctx, mask, t_map["e"], t_map["f"], counts)
            x = {r: x[r] + v[r] / steps for r in self.ranks}
        for r in self.ranks:
            x[r] = x[r] * self.scale(r)
        if self.vaes is not None:
            for r in self.vaes:
                if r in x:
                    x[r] = self.vaes[r].decode(x[r])
        out = [{r: x[r][i, :c[RANKS.index(r)]].cpu().numpy() for r in self.ranks}
               for i, c in enumerate(counts)]
        if self.vertex_cond:
            for o, p in zip(out, vertex_pos):
                o["v"] = np.asarray(p, dtype=np.float32)
        return out
