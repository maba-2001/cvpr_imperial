"""Per-token vertex heads for the vertex block of stage 1.

Both heads read the transformer hidden state h at a VERTEX position and model
that vertex's (x, y, z) in the shared unit-box frame (extract.py: every shape
is centred and scaled so all its geometry lies in [-0.5, 0.5]^3). They differ
only in the per-token distribution, so swapping them is a clean ablation:

`QuantVertexHead` -- categorical over 2^bits bins per axis, factorised within
the token as p(x) p(y|x) p(z|x,y) (PolyGen / BrepGPT quantisation). A
categorical puts point masses on bins, so "same x as that other vertex" has
positive probability; CAD vertex sets are full of such exact coincidences
(axis-aligned faces, shared planes), which a continuous density assigns
probability zero.

`FlowVertexHead` -- MAR's per-token diffusion head (Li et al. 2024, "without
vector quantization"): a small AdaLN MLP conditioned on h, trained here with
rectified flow rather than DDPM to match the rest of this codebase. No
quantisation error, but no point masses either.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    q = 2 ** bits
    return ((x + 0.5) * q).floor().clamp(0, q - 1).long()


def dequantize(q: torch.Tensor, bits: int) -> torch.Tensor:
    return (q.float() + 0.5) / 2 ** bits - 0.5


class FourierEmbed(nn.Module):
    """xyz in [-0.5, 0.5] -> d, via sin/cos at octave frequencies."""

    def __init__(self, d: int, n_freq: int = 8):
        super().__init__()
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(n_freq).float())
        self.proj = nn.Linear(3 + 6 * n_freq, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x.unsqueeze(-1) * self.freqs                          # (..., 3, F)
        f = torch.cat([a.sin(), a.cos()], dim=-1).flatten(-2)     # (..., 6F)
        return self.proj(torch.cat([x, f], dim=-1))


def _mlp(d_in: int, d: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d_out))


class QuantVertexHead(nn.Module):
    def __init__(self, d: int, bits: int = 8):
        super().__init__()
        self.bits, q = bits, 2 ** bits
        self.head = nn.ModuleList(_mlp(d, d, q) for _ in range(3))
        # conditioning of y on x, and z on (x, y), inside the token
        self.cond = nn.ModuleList(nn.Embedding(q, d) for _ in range(3))
        # input side: identical bins -> identical embeddings, so copying a
        # previously placed coordinate is a lookup, not a regression
        self.inp = nn.ModuleList(nn.Embedding(q, d) for _ in range(3))

    def _logits(self, h: torch.Tensor, q: torch.Tensor, axis: int) -> torch.Tensor:
        c = h
        if axis >= 1:
            c = c + self.cond[0](q[..., 0])
        if axis == 2:
            c = c + self.cond[1](q[..., 1])
        return self.head[axis](c)

    def loss(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """h: (N, d), x: (N, 3) -> mean CE per coordinate."""
        q = quantize(x, self.bits)
        return sum(F.cross_entropy(self._logits(h, q, a), q[:, a]) for a in range(3)) / 3

    @torch.no_grad()
    def sample(self, h: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        q = torch.zeros(h.shape[0], 3, dtype=torch.long, device=h.device)
        for a in range(3):
            lg = self._logits(h, q, a).float()
            if temperature <= 0:
                q[:, a] = lg.argmax(-1)
            else:
                q[:, a] = torch.multinomial(F.softmax(lg / temperature, -1), 1).squeeze(-1)
        return dequantize(q, self.bits)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        q = quantize(x, self.bits)
        return sum(self.inp[a](q[..., a]) for a in range(3))


# -- MAR's SimpleMLPAdaLN (github/mar models/diffloss.py), velocity output only

def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


class _ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, c))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(c, 3 * c))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, y):
        shift, scale, gate = self.ada(y).chunk(3, dim=-1)
        return x + gate * self.mlp(_modulate(self.ln(x), shift, scale))


class _AdaMLP(nn.Module):
    def __init__(self, c_in: int, width: int, z: int, depth: int):
        super().__init__()
        self.inp = nn.Linear(c_in, width)
        self.t = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.c = nn.Linear(z, width)
        self.blocks = nn.ModuleList(_ResBlock(width) for _ in range(depth))
        self.ln = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.out = nn.Linear(width, c_in)
        for m in (self.ada[-1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, t, z):
        half = 128
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = (t * 1000)[:, None] * f[None]
        y = self.t(torch.cat([a.cos(), a.sin()], -1)) + self.c(z)
        x = self.inp(x)
        for b in self.blocks:
            x = b(x, y)
        shift, scale = self.ada(y).chunk(2, dim=-1)
        return self.out(_modulate(self.ln(x), shift, scale))


class FlowVertexHead(nn.Module):
    # xyz lives in [-0.5, 0.5]; x4 puts it at roughly the unit scale of the
    # N(0, 1) prior, same reasoning as GeometryFlow's per-rank scales
    SCALE = 4.0

    def __init__(self, d: int, width: int = 256, depth: int = 3,
                 batch_mul: int = 4, steps: int = 32):
        super().__init__()
        self.net = _AdaMLP(3, width, d, depth)
        self.batch_mul, self.steps = batch_mul, steps
        self.inp = FourierEmbed(d)

    def loss(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # MAR's trick: several noise draws per token amortise the (much more
        # expensive) transformer pass that produced h
        h = h.repeat(self.batch_mul, 1)
        x1 = (x * self.SCALE).repeat(self.batch_mul, 1)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        return F.mse_loss(self.net(xt, t, h), x1 - x0)

    @torch.no_grad()
    def sample(self, h: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        x = torch.randn(h.shape[0], 3, device=h.device) * temperature
        t = torch.zeros(h.shape[0], device=h.device)
        for i in range(self.steps):
            t.fill_(i / self.steps)
            x = x + self.net(x, t, h) / self.steps
        return (x / self.SCALE).clamp(-0.5, 0.5)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.inp(x)


def make_head(mode: str, d: int, bits: int) -> nn.Module:
    if mode == "quant":
        return QuantVertexHead(d, bits)
    if mode == "flow":
        return FlowVertexHead(d)
    raise ValueError(mode)
