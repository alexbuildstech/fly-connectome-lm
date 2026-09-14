"""Transformer char-LM baseline (torch CPU) on the same corpus/protocol as FlyLM.

Two sizes:
  --size M : 2 layers, d=128, 4 heads, ffn 512  (~420k params) — "standard tiny GPT"
  --size S : 2 layers, d=96,  4 heads, ffn 384  (~235k params) — matched to FlyLM trainable budget
Trains for --steps with AdamW on random 64-char crops; reports val NLL/acc on the same
val split as the reservoir runs (chars [300k, 340k)).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reservoir_lib import load_corpus

import repo_paths
RESULTS = repo_paths.RESULTS
torch.set_num_threads(2)


class Block(nn.Module):
    def __init__(self, d, h, ffn):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, x, mask):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, V, d, n_layer, n_head, ffn, ctx=64):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, n_head, ffn) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        mask = torch.triu(torch.ones(ctx, ctx) * float("-inf"), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x, self.mask[:T, :T])
        return self.head(self.lnf(x))


def get_batch(ids, batch, ctx, rng):
    ix = rng.integers(0, len(ids) - ctx - 1, batch)
    x = np.stack([ids[i:i + ctx] for i in ix])
    y = np.stack([ids[i + 1:i + ctx + 1] for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def evaluate(model, val_ids, ctx, V):
    model.eval()
    nll, cnt, correct = 0.0, 0, 0
    for s in range(0, len(val_ids) - ctx - 1, 4096):
        chunk = val_ids[s:s + ctx + 4096]
        x = torch.from_numpy(np.stack([chunk[i:i + ctx] for i in range(0, len(chunk) - ctx, ctx)]))
        y = torch.from_numpy(np.stack([chunk[i + 1:i + ctx + 1] for i in range(0, len(chunk) - ctx, ctx)]))
        logits = model(x)
        nll += float(F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="sum"))
        correct += int((logits.argmax(-1) == y).sum())
        cnt += y.numel()
    model.train()
    return nll / cnt, correct / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="L", choices=["L", "M", "S"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-chars", type=int, default=300_000)
    ap.add_argument("--val-chars", type=int, default=40_000)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--max-seconds", type=int, default=480)
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    CKPT = repo_paths.CKPT
    os.makedirs(CKPT, exist_ok=True)
    ck = f"{CKPT}/transformer_{args.tag}_{args.size}_s{args.seed}.pt"

    text, ids, _, _ = load_corpus()
    V = len(set(text))
    train_ids = ids[:args.train_chars]
    val_ids = ids[args.train_chars:args.train_chars + args.val_chars]

    if args.size == "L":
        d, h, ffn, nl = 320, 5, 1280, 5
    elif args.size == "M":
        d, h, ffn, nl = 128, 4, 512, 2
    else:
        d, h, ffn, nl = 96, 4, 384, 2
    model = TinyGPT(V, d, nl, h, ffn, args.ctx)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[tf-{args.size}] params={n_params/1e3:.1f}k", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.05)
    hist = []
    start_step = 0
    if os.path.exists(ck):
        z = torch.load(ck, weights_only=False)
        model.load_state_dict(z["model"])
        opt.load_state_dict(z["opt"])
        sched.load_state_dict(z["sched"])
        start_step = z["step"]
        hist = z["hist"]
        rng_states = z.get("rng", None)
        if rng_states:
            torch.set_rng_state(rng_states[0])
            rng.bit_generator.state = rng_states[1]
        print(f"[tf-{args.size}] resumed at step {start_step}", flush=True)
    for step in range(start_step, args.steps):
        x, y = get_batch(train_ids, args.batch, args.ctx, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            hist.append({"step": step, "train_loss": float(loss)})
            print(f"[tf-{args.size}] step {step} loss {float(loss):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if (step + 1) % 200 == 0 or (step + 1) == args.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step + 1, "hist": hist,
                        "rng": (torch.get_rng_state(), rng.bit_generator.state)}, ck)
        if time.time() - t0 > args.max_seconds:
            print(f"[tf-{args.size}] budget reached at step {step}, checkpointed", flush=True)
            return

    nll, acc = evaluate(model, val_ids, args.ctx, V)
    res = {
        "model": f"transformer_{args.size}",
        "params": int(n_params),
        "steps": args.steps, "batch": args.batch, "ctx": args.ctx, "lr": args.lr,
        "train_chars": args.train_chars, "val_chars": args.val_chars, "seed": args.seed,
        "val_nll_nats_per_char": nll,
        "val_bits_per_char": nll / math.log(2),
        "val_acc": acc,
        "loss_curve": hist,
        "wall_seconds": time.time() - t0,
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = f"{RESULTS}/transformer_{args.tag}_{args.size}_s{args.seed}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    torch.save(model.state_dict(), f"{RESULTS}/transformer_{args.tag}_{args.size}_s{args.seed}.pt")
    print(json.dumps({k: v for k, v in res.items() if k != "loss_curve"}, indent=2))
    print(f"[tf-{args.size}] saved {out}", flush=True)


if __name__ == "__main__":
    main()
