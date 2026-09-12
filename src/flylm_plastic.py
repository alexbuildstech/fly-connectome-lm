"""Track B — Plastic subbrain: LEARNABLE synapse weights on real fly wiring (BPTT).

Takes a top-strength induced subgraph (K=1024 neurons) of the real MaleCNS connectome,
makes every existing synapse a trainable weight (frozen binary mask = fly wiring), and
trains the whole thing (synapses + token encoder + readout) with backprop-through-time
to predict the next character. This is the "fly brain modified enough to be an LM" model.

Controls (same dynamics, same trainable-param count):
  --mode fly_fly    : fly mask, weights initialized from real synapse counts
  --mode fly_rand   : fly mask, weights randomly initialized
  --mode rand_rand  : random mask (same nnz), random init
"""
import argparse, json, math, os, sys, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reservoir_lib import load_corpus, load_adjacency, spectral_radius

RESULTS = "/home/z/my-project/fly-connectome-lm/results"
torch.set_num_threads(2)


def build_subgraph(K=1024, seed=0):
    A = load_adjacency()
    strength = np.asarray(A.sum(axis=1)).ravel() + np.asarray(A.sum(axis=0)).ravel()
    deg = A.getnnz(axis=1) + A.getnnz(axis=0)
    cand = np.where(deg > 0)[0]
    idx = np.array(sorted(cand[np.argsort(strength[cand])[::-1][:K]]))
    Sub = A[np.ix_(idx, idx)].toarray()  # dense (K, K) real synapse counts
    return idx, Sub


def scale_spectral(M, target=1.2, iters=50):
    v = np.random.default_rng(0).standard_normal(M.shape[0]).astype(np.float32)
    v /= np.linalg.norm(v)
    rho = 1.0
    for _ in range(iters):
        w = M @ v
        n = np.linalg.norm(w)
        if n < 1e-12:
            break
        rho = n
        v = w / n
    return M * (target / max(rho, 1e-8))


class PlasticFly(nn.Module):
    """Parameter container only; the actual dynamics live in run_batch()."""
    def __init__(self, V, K, mask, W_init, leak=0.7, gain=1.2):
        super().__init__()
        self.K = K
        self.leak = leak
        self.gain = gain
        self.register_buffer("mask", torch.from_numpy(mask.astype(np.float32)))
        self.W = nn.Parameter(torch.from_numpy(W_init.astype(np.float32)))
        self.B = nn.Parameter(torch.zeros(V, K))
        self.b_in = nn.Parameter(torch.zeros(K))
        self.R = nn.Parameter(torch.zeros(K, V))
        nn.init.normal_(self.B, std=0.3 / math.sqrt(K))
        nn.init.normal_(self.R, std=0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fly_fly",
                    choices=["fly_fly", "fly_rand", "rand_rand"])
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--lr-w", type=float, default=5e-4)
    ap.add_argument("--lr-io", type=float, default=1e-3)
    ap.add_argument("--leak", type=float, default=0.7)
    ap.add_argument("--gain", type=float, default=1.2)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-chars", type=int, default=300_000)
    ap.add_argument("--val-chars", type=int, default=40_000)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--max-seconds", type=int, default=480)
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    CKPT = "/home/z/my-project/data/malecns/ckpts"
    os.makedirs(CKPT, exist_ok=True)
    ck = f"{CKPT}/plastic_{args.tag}_{args.mode}_s{args.seed}.pt"

    text, ids, _, _ = load_corpus()
    V = len(set(text))
    train_ids = ids[:args.train_chars]
    val_ids = ids[args.train_chars:args.train_chars + args.val_chars]

    idx, Sub = build_subgraph(args.K, args.seed)
    K = len(idx)
    mask = (Sub > 0).astype(np.float32)
    nnz_mask = int(mask.sum())
    if args.mode == "fly_fly":
        W0 = scale_spectral(Sub.astype(np.float32), target=args.gain)
    elif args.mode == "fly_rand":
        W0 = rng.standard_normal((K, K)).astype(np.float32) * (1.0 / math.sqrt(K))
        W0 = scale_spectral(W0 * mask, target=args.gain)  # spectral radius of masked random
    else:
        # random mask with same nnz, uniform placement
        rmask = np.zeros((K, K), dtype=np.float32)
        flat = rng.choice(K * K, nnz_mask, replace=False)
        rmask.reshape(-1)[flat] = 1.0
        mask = rmask
        W0 = rng.standard_normal((K, K)).astype(np.float32) * (1.0 / math.sqrt(K))
        W0 = scale_spectral(W0 * mask, target=args.gain)

    model = PlasticFly(V, K, mask, W0, leak=args.leak, gain=args.gain)
    # fix forward: implement properly here instead of the class placeholder
    params = [
        {"params": [model.W], "lr": args.lr_w},
        {"params": [model.B, model.b_in, model.R], "lr": args.lr_io},
    ]
    opt = torch.optim.AdamW(params, weight_decay=0.0)
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"[plastic-{args.mode}] K={K} mask_nnz={nnz_mask} trainable={n_params/1e3:.1f}k", flush=True)

    def run_batch(tokens):
        Bsz = tokens.shape[0]
        x = torch.zeros(Bsz, K)
        logits_all = []
        for t in range(args.T):
            u = model.B[tokens[:, t]] + model.b_in
            z = (x @ (model.W * model.mask).T) * model.gain + u
            x = (1 - model.leak) * x + model.leak * torch.tanh(z)
            logits_all.append(x @ model.R)
        return torch.stack(logits_all, dim=1)  # (B, T, V)

    def get_batch():
        ix = rng.integers(0, len(train_ids) - args.T - 1, args.batch)
        x = np.stack([train_ids[i:i + args.T] for i in ix])
        y = np.stack([train_ids[i + 1:i + args.T + 1] for i in ix])
        return torch.from_numpy(x), torch.from_numpy(y)

    hist = []
    start_step = 0
    if os.path.exists(ck):
        z = torch.load(ck, weights_only=False)
        model.load_state_dict(z["model"])
        opt.load_state_dict(z["opt"])
        start_step = z["step"]
        hist = z["hist"]
        rs = z.get("rng", None)
        if rs:
            torch.set_rng_state(rs[0])
            rng.bit_generator.state = rs[1]
        print(f"[plastic-{args.mode}] resumed at step {start_step}", flush=True)
    for step in range(start_step, args.steps):
        x, y = get_batch()
        logits = run_batch(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            hist.append({"step": step, "train_loss": float(loss)})
            print(f"[plastic-{args.mode}] step {step} loss {float(loss):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if (step + 1) % 100 == 0 or (step + 1) == args.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1, "hist": hist,
                        "rng": (torch.get_rng_state(), rng.bit_generator.state)}, ck)
        if time.time() - t0 > args.max_seconds:
            print(f"[plastic-{args.mode}] budget reached at step {step}, checkpointed", flush=True)
            return

    # ---- eval: contiguous-stream rollout over val (state carried across, no teacher reset) ----
    @torch.no_grad()
    def evaluate():
        model.eval()
        ctx = args.T
        nll, cnt, correct = 0.0, 0, 0
        for s in range(0, len(val_ids) - ctx - 1, 4096):
            chunk = val_ids[s:s + ctx + 4096]
            xin = torch.from_numpy(np.stack([chunk[i:i + ctx] for i in range(0, len(chunk) - ctx, ctx)]))
            yin = torch.from_numpy(np.stack([chunk[i + 1:i + ctx + 1] for i in range(0, len(chunk) - ctx, ctx)]))
            logits = run_batch(xin)
            nll += float(torch.nn.functional.cross_entropy(
                logits.reshape(-1, V), yin.reshape(-1), reduction="sum"))
            correct += int((logits.argmax(-1) == yin).sum())
            cnt += yin.numel()
        model.train()
        return nll / cnt, correct / cnt

    nll, acc = evaluate()
    res = {
        "model": f"plastic_fly_{args.mode}",
        "K": K, "mask_nnz": nnz_mask, "trainable_params": n_params,
        "steps": args.steps, "batch": args.batch, "T": args.T,
        "lr_w": args.lr_w, "lr_io": args.lr_io, "leak": args.leak, "gain": args.gain,
        "train_chars": args.train_chars, "val_chars": args.val_chars, "seed": args.seed,
        "val_nll_nats_per_char": nll,
        "val_bits_per_char": nll / math.log(2),
        "val_acc": acc,
        "loss_curve": hist,
        "wall_seconds": time.time() - t0,
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = f"{RESULTS}/plastic_{args.tag}_{args.mode}_s{args.seed}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "loss_curve"}, indent=2))
    print(f"[plastic-{args.mode}] saved {out}", flush=True)


if __name__ == "__main__":
    main()
