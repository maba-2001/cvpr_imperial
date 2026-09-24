"""Stage 1: autoregressive model of the canonical code, plus the vertex block.
See the README's "canonical code" and "vertex block" sections for the design
rationale; this is the mechanism.

A causal transformer over the token stream of `grammar.py`, with six heads --
NLOOPS, LEN, FACE_TYPE, EDGE_TYPE, LOOP_IS_OUTER (plain classification) and a
pointer head for ALPHA that attends over the darts currently open. Sampling
runs `grammar.Builder` alongside the model and masks the logits to the
grammar, so every sample decodes to a closed, connected, orientable
2-manifold map -- validity is a property of the output space, not a metric
of this model.

Vertex block (`cfg.VERTEX_MODE`): after the topology, one VERTEX token per
vertex, in the decoded map's own order (σ-orbits numbered by first dart in
walk order -- a function of the prefix, so no ordering to learn). Appended
rather than interleaved: a vertex's identity isn't settled until its last
incident dart closes, so an interleaved token couldn't be tied to one vertex
exactly. Each vertex token carries a *structural slot* -- the mean
positional embedding of its incident darts' ALPHA tokens -- added both at the
position that predicts it (so attention can retrieve its incidence context)
and to its own content embedding (so later vertices can see where it landed).
The per-token distribution is `vertex_head.make_head(cfg.VERTEX_MODE)`.

Conditioning is (genus, n_faces) with classifier-free guidance. n_faces is
exact by construction (the number of face groups emitted); genus is verified
exactly and for free by `CMap.invariants`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import grammar as C
from .. import config as cfg
from ..data.cmap import CMap
from ..geometry.vertex_head import make_head


class TopoSample(NamedTuple):
    m: CMap
    edge_type: np.ndarray
    face_type: np.ndarray
    loop_is_outer: np.ndarray
    vertices: np.ndarray | None     # (V, 3) in the unit-box frame, or None


def targets_from_tokens(kinds: np.ndarray, values: np.ndarray):
    """Per-position supervision and the pointer mask implied by the grammar.

    Returns (tgt, open_mask) where tgt[t] is the class index for the head
    selected by kinds[t] (ALPHA targets index the *token position* of the dart
    being closed, or T for OPEN) and open_mask[t, j] marks dart-token j as open
    just before step t. VERTEX tokens get no class target; they are
    supervised by the vertex head on their coordinates.
    """
    T = len(kinds)
    tgt = np.full(T, -100, dtype=np.int64)
    open_mask = np.zeros((T, T), dtype=bool)
    open_pos: list[int] = []
    n = 0
    dart_pos: dict[int, int] = {}          # dart index -> token position
    for t, (k, v) in enumerate(zip(kinds, values)):
        if k == C.ALPHA:
            open_mask[t, open_pos] = True
            if v == C.OPEN:
                tgt[t] = T                 # the "open a new edge" class
                open_pos.append(t)
            else:
                j = dart_pos[n - v]
                tgt[t] = j
                open_pos.remove(j)
            dart_pos[n] = t
            n += 1
        elif k != C.VERTEX:
            tgt[t] = v
    return tgt, open_mask


def vertex_slots(kinds: np.ndarray, vertex_of_dart: np.ndarray) -> list[tuple[int, int, float]]:
    """(t, j, w): the vertex token at position t averages the positional
    embedding of ALPHA token j, one per incident dart, with w = 1/degree.
    `vertex_of_dart` is in the decoded map's numbering, whose dart i is the
    i-th ALPHA token of the sequence."""
    alpha_pos = np.nonzero(kinds == C.ALPHA)[0]
    vpos = np.nonzero(kinds == C.VERTEX)[0]
    deg = np.bincount(vertex_of_dart, minlength=len(vpos))
    return [(int(vpos[v]), int(alpha_pos[i]), 1.0 / deg[v])
            for i, v in enumerate(vertex_of_dart)]


class CodeTransformer(nn.Module):
    def __init__(self, vertex_mode: str | None = cfg.VERTEX_MODE):
        super().__init__()
        d = cfg.TOPO_DIM
        self.vertex_mode = vertex_mode
        self.kind_emb = nn.Embedding(C.VERTEX + 1, d)
        self.value_emb = nn.Embedding(cfg.MAX_REL + 2, d)
        n_pos = cfg.MAX_SEQ if vertex_mode else cfg.MAX_TOKENS
        self.pos_emb = nn.Parameter(torch.randn(n_pos, d) * 0.02)
        self.cond_mlp = nn.Sequential(nn.Linear(2, d), nn.SiLU(), nn.Linear(d, d))
        self.null_cond = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(
            d, cfg.TOPO_HEADS, 4 * d, dropout=0.0, batch_first=True,
            norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, cfg.TOPO_LAYERS)
        self.norm = nn.LayerNorm(d)
        self.nloops_head = nn.Linear(d, cfg.MAX_LOOPS_PER_FACE + 1)
        self.len_head = nn.Linear(d, cfg.MAX_LOOP_LEN + 1)
        self.facetype_head = nn.Linear(d, C.N_FACE_TYPES)
        self.edgetype_head = nn.Linear(d, C.N_EDGE_TYPES)
        self.loopouter_head = nn.Linear(d, C.N_LOOP_IS_OUTER)
        self.ptr_q = nn.Linear(d, d)
        self.ptr_k = nn.Linear(d, d)
        self.open_logit = nn.Linear(d, 1)
        if vertex_mode:
            self.vhead = make_head(vertex_mode, d, cfg.VERTEX_BITS)
            self.slot_q = nn.Linear(d, d, bias=False)     # "which vertex to place here"
            self.slot_in = nn.Linear(d, d, bias=False)    # "which vertex was placed here"

    def _cond(self, cond: torch.Tensor, drop: bool, force_null: bool = False) -> torch.Tensor:
        """The conditioning token. `force_null` deterministically selects the
        learned null embedding -- what unconditional sampling and the
        unconditional branch of classifier-free guidance actually want, as
        opposed to a real (genus, n_faces) pair of (0, 0), which is what
        passing a zero cond tensor through `cond_mlp` would compute instead."""
        if force_null:
            return self.null_cond.unsqueeze(0).expand(cond.shape[0], -1)
        c = self.cond_mlp(torch.stack(
            [cond[:, 0].float() / 8.0, cond[:, 1].float() / 32.0], dim=1))
        if drop and cfg.COND_DROP > 0:
            m = torch.rand(c.shape[0], 1, device=c.device) < cfg.COND_DROP
            c = torch.where(m, self.null_cond, c)
        return c

    def _slots(self, B: int, T: int, slot_idx, slot_w) -> torch.Tensor:
        s = self.pos_emb.new_zeros(B, T, self.pos_emb.shape[1])
        if slot_idx is not None and len(slot_idx):
            b, t, j = slot_idx.unbind(1)
            keep = t < T
            s.index_put_((b[keep], t[keep]), self.pos_emb[j[keep]] * slot_w[keep, None],
                         accumulate=True)
        return s

    def forward(self, kinds, values, cond, drop_cond: bool = False,
                force_null: bool = False, coords=None, slot_idx=None,
                slot_w=None) -> torch.Tensor:
        """Hidden states aligned so that h[:, t] predicts token t.
        coords: (B, >=T, 3) vertex coordinates at VERTEX positions;
        slot_idx/slot_w: flat (b, t, j) / weight lists from `vertex_slots`."""
        B, T = kinds.shape
        inp = self.kind_emb(kinds) + self.value_emb(values.clamp(max=cfg.MAX_REL + 1))
        slot = None
        if self.vertex_mode and coords is not None:
            slot = self._slots(B, T, slot_idx, slot_w)
            vin = self.kind_emb(kinds) + self.vhead.embed(coords[:, :T]) + self.slot_in(slot)
            inp = torch.where((kinds == C.VERTEX).unsqueeze(-1), vin, inp)
        h = torch.cat([self._cond(cond, drop_cond, force_null).unsqueeze(1),
                       inp[:, :-1]], dim=1)
        h = h + self.pos_emb[:T]
        if slot is not None:
            h = h + self.slot_q(slot)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), 1)
        return self.norm(self.blocks(h, mask=causal))

    def logits(self, h: torch.Tensor, kinds: torch.Tensor, open_mask: torch.Tensor):
        """Pointer logits for ALPHA steps: (B, T, T+1), last column = OPEN."""
        q, k = self.ptr_q(h), self.ptr_k(h)
        ptr = q @ k.transpose(1, 2) / q.shape[-1] ** 0.5
        ptr = ptr.masked_fill(~open_mask, -1e9)
        ptr = torch.cat([ptr, self.open_logit(h)], dim=2)
        return (self.nloops_head(h), self.len_head(h), ptr,
                self.facetype_head(h), self.edgetype_head(h), self.loopouter_head(h))

    def loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        h = self.forward(batch["kinds"], batch["values"], batch["cond"], drop_cond=True,
                         coords=batch.get("coords"), slot_idx=batch.get("slot_idx"),
                         slot_w=batch.get("slot_w"))
        nl, ln, ptr, ft, et, lo = self.logits(h, batch["kinds"], batch["open_mask"])
        tgt, kinds, mask = batch["tgt"], batch["kinds"], batch["mask"]
        terms, total, n_tok = {}, 0.0, 0
        for name, head, kind in (("nloops", nl, C.NLOOPS), ("len", ln, C.LEN),
                                 ("alpha", ptr, C.ALPHA), ("facetype", ft, C.FACE_TYPE),
                                 ("edgetype", et, C.EDGE_TYPE),
                                 ("loopouter", lo, C.LOOP_IS_OUTER)):
            sel = mask & (kinds == kind)
            if not sel.any():
                continue
            ce = F.cross_entropy(head[sel].float(), tgt[sel], reduction="sum")
            terms[name] = float(ce.detach()) / int(sel.sum())
            total = total + ce
            n_tok += int(sel.sum())
        loss = total / max(n_tok, 1)
        if self.vertex_mode and "coords" in batch:
            vsel = mask & (kinds == C.VERTEX)
            if vsel.any():
                vl = self.vhead.loss(h[vsel].float(), batch["coords"][vsel])
                terms["vertex"] = float(vl.detach())
                loss = loss + cfg.VERTEX_LOSS_WEIGHT * vl
        return loss, terms


def _step_logits(model, kinds, values, cond, t, force_null: bool = False, **vert):
    """Head logits for the next token, given the prefix kinds/values[:, :t+1],
    plus the hidden state at t (what the vertex head reads)."""
    h_all = model.forward(kinds[:, :t + 1], values[:, :t + 1], cond,
                          force_null=force_null, **vert)
    h = h_all[:, t]
    q = model.ptr_q(h)
    ptr = torch.einsum("bd,btd->bt", q, model.ptr_k(h_all)) / q.shape[-1] ** 0.5
    return (model.nloops_head(h), model.len_head(h), ptr,
            model.open_logit(h).squeeze(-1), model.facetype_head(h),
            model.edgetype_head(h), model.loopouter_head(h)), h


@torch.no_grad()
def sample(model: CodeTransformer, n: int, cond=None, guidance: float = 1.0,
           temperature: float = 1.0, device="cuda",
           vertex_temperature: float = 1.0) -> list[TopoSample]:
    """Grammar-masked ancestral sampling; every returned map is valid. With a
    vertex block, each sequence switches to placing its vertices as soon as
    its own topology closes (others in the batch may still be mid-topology).
    Vertex tokens use the conditional branch only -- guidance acts on the
    topology."""
    model.eval()
    vmode = model.vertex_mode
    T_max = cfg.MAX_SEQ if vmode else cfg.MAX_TOKENS
    builders = [C.Builder(cfg.MAX_DARTS, cfg.MAX_LOOP_LEN, cfg.MAX_LOOPS_PER_FACE)
                for _ in range(n)]
    # cond_t is a dummy placeholder when unconditional -- force_null makes
    # _cond ignore it and use the learned null embedding instead.
    cond_t = (torch.zeros(n, 2, dtype=torch.long, device=device) if cond is None
              else torch.as_tensor(cond, dtype=torch.long, device=device)
              .view(1, 2).expand(n, 2).contiguous())
    kinds = torch.zeros(n, T_max, dtype=torch.long, device=device)
    values = torch.zeros(n, T_max, dtype=torch.long, device=device)
    coords = torch.zeros(n, T_max, 3, device=device) if vmode else None
    slot_idx = torch.zeros(0, 3, dtype=torch.long, device=device)
    slot_w = torch.zeros(0, device=device)
    dart_pos: list[dict[int, int]] = [{} for _ in range(n)]
    topo: list = [None] * n        # decoded (m, et, ft, lo) once the topology closes
    plan: list = [None] * n        # per vertex: ALPHA positions of its incident darts
    placed: list[list[int]] = [[] for _ in range(n)]   # token positions of placed vertices
    live = list(range(n))
    direct_heads = {C.NLOOPS: 0, C.LEN: 1, C.FACE_TYPE: 4, C.EDGE_TYPE: 5,
                    C.LOOP_IS_OUTER: 6}

    for t in range(T_max):
        if not live:
            break
        vert_b = [b for b in live if topo[b] is not None]
        if vert_b:
            new = [(b, t, j, 1.0 / len(inc))
                   for b in vert_b for inc in [plan[b][len(placed[b])]] for j in inc]
            slot_idx = torch.cat([slot_idx, torch.tensor([r[:3] for r in new], device=device)])
            slot_w = torch.cat([slot_w, torch.tensor([r[3] for r in new], device=device)])
        vert = dict(coords=coords, slot_idx=slot_idx, slot_w=slot_w) if vmode else {}
        heads, h = _step_logits(model, kinds, values, cond_t, t, force_null=cond is None, **vert)
        if cond is not None and guidance != 1.0:
            null, _ = _step_logits(model, kinds, values, cond_t, t, force_null=True, **vert)
            heads = tuple(u + guidance * (c - u) for c, u in zip(heads, null))
        _, _, ptr, open_col, _, _, _ = heads

        if vert_b:
            xyz = model.vhead.sample(h[vert_b].float(), vertex_temperature)
            idx = torch.tensor(vert_b, device=device)
            kinds[idx, t] = C.VERTEX
            coords[idx, t] = xyz
            for b in vert_b:
                placed[b].append(t)

        still = []
        for b in live:
            if topo[b] is not None:
                if len(placed[b]) < len(plan[b]):
                    still.append(b)
                continue
            bd = builders[b]
            kind, allowed = bd.allowed()
            if kind in direct_heads:
                logit = heads[direct_heads[kind]][b]
                choices = torch.tensor(allowed, device=device)
                v = allowed[_categorical(logit[choices], temperature)]
            else:
                lg = torch.stack([open_col[b] if val == C.OPEN
                                  else ptr[b, dart_pos[b][bd.n - val]] for val in allowed])
                v = allowed[_categorical(lg, temperature)]
                dart_pos[b][bd.n] = t
            bd.push(v)
            kinds[b, t], values[b, t] = kind, min(v, cfg.MAX_REL + 1)
            if not bd.done:
                still.append(b)
            elif vmode:
                topo[b] = bd.build()
                vod = topo[b][0].vertex_of_dart
                pos = np.array([dart_pos[b][i] for i in range(len(vod))])
                plan[b] = [pos[vod == k].tolist() for k in range(int(vod.max()) + 1)]
                still.append(b)
        live = still

    out = []
    for b, bd in enumerate(builders):
        if not bd.done:
            continue
        if not vmode:
            out.append(TopoSample(*bd.build(), None))
        elif len(placed[b]) == len(plan[b]):       # else ran out of sequence budget
            out.append(TopoSample(*topo[b], coords[b, placed[b]].cpu().numpy()))
    return out


def _categorical(logits: torch.Tensor, temperature: float) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    return int(torch.multinomial(F.softmax(logits.float() / temperature, dim=-1), 1))
