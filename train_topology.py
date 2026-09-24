"""Train the canonical-code model.

    python -m cvpr_imperial.train_topology --steps 40000 --batch 16
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from . import config as cfg
from .data import CodeDataset, collate
from .topology_model import CodeTransformer, sample


def make_scheduler(opt, kind: str, lr: float, steps: int, warmup_frac: float = 0.03):
    if kind == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=warmup_frac)
    warmup_steps = max(1, int(steps * warmup_frac))
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup_steps))


def evaluate(model, loader, device, n_batches: int = 20) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = model.loss(batch)
            total += float(loss)
            n += 1
    model.train()
    return total / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-faces", type=int, default=None,
                    help="restrict training/val to shapes with at most this many faces")
    ap.add_argument("--sched", choices=["onecycle", "constant"], default="onecycle",
                    help="constant = linear warmup then flat lr, no decay-to-zero")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg.RUN_DIR.mkdir(parents=True, exist_ok=True)
    train = CodeDataset("train", limit=args.limit or None, max_faces=args.max_faces)
    val = CodeDataset("val", limit=2000, max_faces=args.max_faces)
    print(f"train {len(train)} | val {len(val)} sequences"
          + (f" (<= {args.max_faces} faces)" if args.max_faces else ""), flush=True)
    kw = dict(batch_size=args.batch, collate_fn=collate, num_workers=4, drop_last=True)
    tl = DataLoader(train, shuffle=True, **kw)
    vl = DataLoader(val, shuffle=False, **kw)

    model = CodeTransformer().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = make_scheduler(opt, args.sched, args.lr, args.steps)
    step, t0, best = 0, time.time(), float("inf")
    while step < args.steps:
        for batch in tl:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss, terms = model.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                msg = " ".join(f"{k}={v:.3f}" for k, v in terms.items())
                print(f"{step:6d} loss={float(loss):.4f} {msg} "
                      f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
            if step % 2000 == 0 or step == args.steps:
                v = evaluate(model, vl, args.device)
                print(f"{step:6d} val={v:.4f}", flush=True)
                if v < best:
                    best = v
                    torch.save(model.state_dict(), cfg.RUN_DIR / "topology.pt")
            if step >= args.steps:
                break
    print(f"best val {best:.4f}; samples from the final model:", flush=True)
    for m in (s.m for s in sample(model, 4, device=args.device)[:4]):
        inv = m.invariants()
        print(f"  V={m.counts()[0]} E={m.counts()[1]} F={inv['n_faces']} "
              f"genus={inv['genus']} valid={m.check()['tier1_pass']}")


if __name__ == "__main__":
    main()
