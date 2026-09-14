"""FlyLM v3 plastic — session-4 upgrade #1: BACKPROP-THROUGH-TIME ON TRAINABLE SYNAPSES
OF THE FULL CONNECTOME (critique #1: "only the linear readout is trained — the fly brain
is not learning anything").

Scales the session-2 appendix (1,024-neuron sub-brain, 61k trainable synapses, 3.252 bpc)
to the complete graph: all 211,577 neurons in the dynamics, synapse weights trainable by
BPTT on the frozen wiring pattern.

WHY A CUSTOM SPMM (and why --train-frac exists):
  torch's CPU sparse.mm backward materializes a DENSE (N x N) values-gradient -> OOM at
  N=211k. PlasticSpMM does it properly:
      forward : native CSR spmm (fast)
      grad_x  : native spmm with the precomputed A^T structure (values permuted CSR->CSC)
      grad_w  : chunked gather over the TRAINABLE edge subset only
  Default --train-frac 0.2 = a uniformly random 5.2M-synapse subset trains by BPTT (the
  rest stay frozen at their init values); --train-frac 1.0 trains all 26M (about 5x the
  gradient traffic — usable, slower per character).

Modes (appendix controls at full scale):
  fly_fly    : fly wiring, synapse init = real synapse counts (spectral radius -> 1.2)
  fly_rand   : fly wiring, random synapse init
  rand_rand  : config-model random wiring (same out-degree sequence), random init
  fly_frozen : fly wiring, real counts, synapses frozen (encoder/readout still train) —
               isolates what trainable synapses add in the identical protocol

Dynamics (appendix regime, kept for continuity): Z = (W @ X) * 1.2 + enc + b_in;
X = 0.3*X + 0.7*tanh(Z). Stateless TBPTT chunks (B x T), AdamW, clip, ckpt/resume.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from repo_paths import CKPT, PROCESSED, RESULTS

torch.set_num_threads(2)
V = 65
GAIN = 1.2


# ---------------------------------------------------------------- graph builders
def spectral_scale_values(indptr, indices, values, target=1.2, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    N = len(indptr) - 1
    v = rng.standard_normal(N).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    rho = 1.0
    A = sp.csr_matrix((values, indices, indptr), shape=(N, N))
    for _ in range(iters):
        w = A @ v
        n = float(np.linalg.norm(w))
        if n < 1e-12:
            return 1.0
        rho = n
        v = (w / n).astype(np.float32)
    return target / max(rho, 1e-12)


def build_mode(mode, seed=0):
    A = sp.load_npz(f"{PROCESSED}/adjacency.npz").tocsr().astype(np.float32)
    N = A.shape[0]
    if mode in ("fly_fly", "fly_frozen"):
        indptr, indices, values = A.indptr, A.indices, A.data.copy()
        values *= spectral_scale_values(indptr, indices, values, GAIN, seed=seed)
    elif mode == "fly_rand":
        rng = np.random.default_rng(seed + 10)
        indptr, indices = A.indptr, A.indices
        values = rng.standard_normal(A.nnz).astype(np.float32) * 0.05
        values *= spectral_scale_values(indptr, indices, values, GAIN, seed=seed)
    elif mode == "rand_rand":
        rng = np.random.default_rng(seed + 20)
        counts = A.getnnz(axis=1)
        rows = np.repeat(np.arange(N, dtype=np.int64), counts)
        cols = rng.integers(0, N, size=rows.size)
        C = sp.coo_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)),
                          shape=(N, N)).tocsr()
        C.sum_duplicates()
        indptr, indices = C.indptr, C.indices
        values = rng.standard_normal(C.nnz).astype(np.float32) * 0.05
        values *= spectral_scale_values(indptr, indices, values, GAIN, seed=seed)
        del C
    else:
        raise ValueError(mode)
    del A
    return (indptr.astype(np.int64), indices.astype(np.int64),
            values.astype(np.float32), N)


def transpose_structure(indptr, indices, nnz):
    """CSR structure of A^T + the permutation taking values from CSR order to CSC order.
    Built by running tocsc on a copy whose 'values' are 0..nnz-1."""
    N = len(indptr) - 1
    marker = sp.csr_matrix((np.arange(nnz, dtype=np.float64), indices, indptr),
                           shape=(N, N))
    At = marker.tocsc()
    permT = np.asarray(At.data, dtype=np.int64)          # values index in CSR order
    return (At.indptr.astype(np.int32), At.indices.astype(np.int32), permT)


def sensory_indices_original():
    import pandas as pd
    slim = pd.read_parquet(f"{PROCESSED}/annotations_slim.parquet")
    body_ids = np.load(f"{PROCESSED}/annotated_body_ids.npy")
    id2orig = {int(b): i for i, b in enumerate(body_ids)}
    KEY = ("olfactory", "visual", "auditory", "mechanosensory", "gustatory",
           "thermosensory", "hygrosensory", "chemosensory", "proprioceptive",
           "nociceptive", "sensory")
    sup = slim["superclass"].fillna("").astype(str).values
    cls = slim["class"].fillna("").astype(str).values
    keep = np.zeros(len(slim), dtype=bool)
    for i in range(len(slim)):
        c = cls[i].lower()
        if "sensory" in sup[i].lower() or any(k in c for k in KEY):
            keep[i] = True
    bodies = slim["bodyId"].values[keep]
    return np.array(sorted(id2orig[int(b)] for b in bodies if int(b) in id2orig),
                    dtype=np.int64)


# ---------------------------------------------------------------- custom spmm
class PlasticSpMM(torch.autograd.Function):
    """out = W @ x with trainable W values on a frozen pattern.

    forward  : native CSR spmm
    backward : grad_x = W^T @ g (native, via precomputed A^T structure)
               grad_w[e] = sum_b x[col_e, b] * g[row_e, b]  (trainable edges only,
               processed in chunks; frozen edges get exactly 0)
    """

    @staticmethod
    def forward(ctx, x, vals, crow, col, crowT, colT, permT, tr_pos, rowT, colT):
        W = torch.sparse_csr_tensor(crow, col, vals, size=(crow.numel() - 1,) * 2)
        out = torch.sparse.mm(W, x)
        ctx.save_for_backward(x, vals)
        ctx.crowT, ctx.colT, ctx.permT = crowT, colT, permT
        ctx.tr_pos, ctx.rowT, ctx.colT = tr_pos, rowT, colT
        ctx.shape = (crow.numel() - 1,)
        return out

    @staticmethod
    def backward(ctx, g):
        x, vals = ctx.saved_tensors
        N = ctx.shape[0]
        # grad_x = W^T @ g   (W^T = A^T in CSR with CSC-ordered values)
        WT = torch.sparse_csr_tensor(ctx.crowT, ctx.colT, vals[ctx.permT],
                                     size=(N, N))
        grad_x = torch.sparse.mm(WT, g)
        # grad_w on the trainable subset, chunked
        grad_vals = torch.zeros_like(vals)
        if ctx.tr_pos is not None and ctx.tr_pos.numel():
            CH = 1_000_000
            xt, gt = x.t().contiguous(), g.t().contiguous()   # (B, N)
            for s in range(0, ctx.tr_pos.numel(), CH):
                e = min(s + CH, ctx.tr_pos.numel())
                pos = ctx.tr_pos[s:e]
                r, c = ctx.rowT[s:e], ctx.colT[s:e]
                grad_vals[pos] = (xt[:, c] * gt[:, r]).sum(dim=0)
        return grad_x, grad_vals, None, None, None, None, None, None, None, None


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fly_fly",
                    choices=["fly_fly", "fly_rand", "rand_rand", "fly_frozen"])
    ap.add_argument("--train-frac", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--lr-w", type=float, default=5e-4)
    ap.add_argument("--lr-io", type=float, default=1e-3)
    ap.add_argument("--leak", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-chars", type=int, default=800_000)
    ap.add_argument("--val-chars", type=int, default=24_000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--max-seconds", type=int, default=100000)
    ap.add_argument("--tag", default="v3plastic")
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed)

    os.makedirs(CKPT, exist_ok=True)
    ck = f"{CKPT}/plastic_{args.tag}_{args.mode}_f{args.train_frac}_s{args.seed}.pt"
    res = f"{RESULTS}/plastic_{args.tag}_{args.mode}_f{args.train_frac}_s{args.seed}.json"
    freeze_syn = args.mode == "fly_frozen"
    train_syn = (not freeze_syn) and args.train_frac > 0

    ids = np.load(f"{PROCESSED}/corpus_ids.npy")
    train_ids = ids[:args.train_chars]
    val_ids = ids[args.train_chars:args.train_chars + args.val_chars]
    B, T = args.batch, args.T

    if os.path.exists(ck):
        z = torch.load(ck, weights_only=False)
        crow = z["crow"]; col = z["col"]; crowT = z["crowT"]; colT = z["colT"]
        permT = z["permT"]; tr_pos = z["tr_pos"]; rowT = z["rowT"]; colT2 = z["colT"]
        rowT = z["rowT"]
        N = int(z["N"]); n_sens = int(z["n_sens"]); in_idx = z["in_idx"]
        Wv = z["Wv"]; B_enc = z["B_enc"]; b_in = z["b_in"]; R = z["R"]
        start_step = z["step"]; hist = z["hist"]
        params = ([Wv] if train_syn else []) + [B_enc, b_in, R]
        opt = torch.optim.AdamW(
            ([{"params": [Wv], "lr": args.lr_w}] if train_syn else []) +
            [{"params": [B_enc, b_in, R], "lr": args.lr_io}], weight_decay=0.0)
        if z.get("opt") is not None:
            opt.load_state_dict(z["opt"])
        print(f"[{args.tag}:{args.mode}] resumed step {start_step}", flush=True)
    else:
        indptr, indices, values, N = build_mode(args.mode, args.seed)
        crow = torch.from_numpy(indptr.astype(np.int32))
        col = torch.from_numpy(indices.astype(np.int32))
        Wv = torch.tensor(values)
        del values
        nnz = Wv.numel()
        print(f"[{args.tag}:{args.mode}] building transpose structure...", flush=True)
        crowT_n, colT_n, permT_n = transpose_structure(indptr, indices, nnz)
        crowT, colT = torch.from_numpy(crowT_n), torch.from_numpy(colT_n)
        permT = torch.from_numpy(permT_n)
        del indptr, indices, crowT_n, colT_n, permT_n
        rng = np.random.default_rng(args.seed + 40)
        if train_syn:
            n_tr = int(round(nnz * args.train_frac))
            tr_pos = torch.from_numpy(np.sort(rng.choice(nnz, n_tr, replace=False))
                                      .astype(np.int64))
            e_row = np.repeat(np.arange(N), np.diff(crow.numpy()))
            rowT = torch.from_numpy(e_row[tr_pos.numpy()].astype(np.int64))
            colT = col[tr_pos].long()
            del e_row
        else:
            tr_pos = rowT = colT = torch.zeros(0, dtype=torch.int64)
        sens_idx = sensory_indices_original()
        n_sens = len(sens_idx)
        in_idx = torch.from_numpy(sens_idx)
        B_enc = (torch.randn(V, n_sens, generator=torch.Generator().manual_seed(args.seed + 3))
                 * 0.05)
        b_in = torch.zeros(N)
        R = torch.randn(N, V, generator=torch.Generator().manual_seed(args.seed + 4)) * 0.02
        start_step = 0; hist = []
        params = ([Wv] if train_syn else []) + [B_enc, b_in, R]
        opt = torch.optim.AdamW(
            ([{"params": [Wv], "lr": args.lr_w}] if train_syn else []) +
            [{"params": [B_enc, b_in, R], "lr": args.lr_io}], weight_decay=0.0)
        torch.save({"crow": crow, "col": col, "crowT": crowT, "colT": colT,
                    "permT": permT, "tr_pos": tr_pos, "rowT": rowT, "colT": colT,
                    "N": N, "n_sens": n_sens, "in_idx": in_idx, "Wv": Wv.detach(),
                    "B_enc": B_enc, "b_in": b_in, "R": R.detach(), "step": 0,
                    "hist": hist, "opt": None, "args": vars(args)}, ck)
        print(f"[{args.tag}:{args.mode}] fresh init N={N} nnz={Wv.numel()} "
              f"trainable_syn={0 if not train_syn else tr_pos.numel()}", flush=True)

    Wv.requires_grad_(train_syn)
    B_enc.requires_grad_(True); b_in.requires_grad_(True); R.requires_grad_(True)
    nnz = Wv.numel()
    n_train = sum(p.numel() for p in params if p.requires_grad)

    def run_batch(tok):
        x = torch.zeros(N, B, requires_grad=False)
        logits_all = []
        for t in range(T):
            if train_syn or freeze_syn:
                x = PlasticSpMM.apply(x, Wv, crow, col, crowT, colT, permT,
                                      tr_pos if train_syn else None, rowT, colT)
            else:
                W = torch.sparse_csr_tensor(crow, col, Wv, size=(N, N))
                x = torch.sparse.mm(W, x)
            x = x * GAIN
            x = x.index_add(0, in_idx, B_enc[tok[:, t]].T)
            x = x + b_in.unsqueeze(1)
            x = (1 - args.leak) * x + args.leak * torch.tanh(x)
            logits_all.append((x / (x.norm(dim=0, keepdim=True) + 1e-6)).T @ R)
        return torch.stack(logits_all, dim=1)

    rng = np.random.default_rng(args.seed)

    def get_batch():
        ix = rng.integers(0, len(train_ids) - T - 1, B)
        x = np.stack([train_ids[i:i + T] for i in ix])
        y = np.stack([train_ids[i + 1:i + T + 1] for i in ix])
        return torch.from_numpy(x), torch.from_numpy(y)

    for step in range(start_step, args.steps):
        xb, yb = get_batch()
        logits = run_batch(xb)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step()
        if step % 50 == 0:
            hist.append({"step": step, "loss": float(loss)})
            print(f"[{args.tag}:{args.mode}] step {step} loss {float(loss):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.steps:
            torch.save({"crow": crow, "col": col, "crowT": crowT, "colT": colT,
                        "permT": permT, "tr_pos": tr_pos, "rowT": rowT, "colT": colT,
                        "N": N, "n_sens": n_sens, "in_idx": in_idx,
                        "Wv": Wv.detach(), "B_enc": B_enc.detach(),
                        "b_in": b_in.detach(), "R": R.detach(), "step": step + 1,
                        "hist": hist, "opt": opt.state_dict(), "args": vars(args)}, ck)
        if time.time() - t0 > args.max_seconds:
            torch.save({"crow": crow, "col": col, "crowT": crowT, "colT": colT,
                        "permT": permT, "tr_pos": tr_pos, "rowT": rowT, "colT": colT,
                        "N": N, "n_sens": n_sens, "in_idx": in_idx,
                        "Wv": Wv.detach(), "B_enc": B_enc.detach(),
                        "b_in": b_in.detach(), "R": R.detach(), "step": step,
                        "hist": hist, "opt": opt.state_dict(), "args": vars(args)}, ck)
            print(f"[{args.tag}:{args.mode}] budget reached at step {step}", flush=True)
            return

    # ---- eval: contiguous-stream rollout, state carried across chunks ----
    @torch.no_grad()
    def evaluate():
        nll, cnt, correct = 0.0, 0, 0
        x = torch.zeros(N, 1)
        W = torch.sparse_csr_tensor(crow, col, Wv, size=(N, N))
        pos = 0
        while pos < len(val_ids) - 1:
            t_end = min(T, len(val_ids) - 1 - pos)
            for t in range(t_end):
                x = torch.sparse.mm(W, x) * GAIN
                x = x.index_add(0, in_idx, B_enc[val_ids[pos + t]].T)
                x = x + b_in.unsqueeze(1)
                x = (1 - args.leak) * x + args.leak * torch.tanh(x)
                xn = x / (x.norm(dim=0, keepdim=True) + 1e-6)
                logit = (xn.T @ R)[0]
                tgt = torch.tensor([val_ids[pos + t + 1]])
                nll += float(torch.nn.functional.cross_entropy(logit.unsqueeze(0), tgt,
                                                               reduction="sum"))
                correct += int(logit.argmax() == val_ids[pos + t + 1])
                cnt += 1
            pos += t_end
        return nll / max(cnt, 1), correct / max(cnt, 1)

    nll, acc = evaluate()
    out = {
        "model": f"plastic_v3_{args.mode}", "mode": args.mode,
        "train_frac": args.train_frac,
        "N": N, "nnz": int(nnz),
        "trainable_synapses": int(tr_pos.numel()) if train_syn else 0,
        "trainable_params": int(n_train),
        "steps_done": args.steps, "batch": B, "T": T,
        "lr_w": args.lr_w, "lr_io": args.lr_io, "leak": args.leak, "gain": GAIN,
        "train_chars": args.train_chars, "val_chars": args.val_chars, "seed": args.seed,
        "val_bits_per_char": nll / math.log(2), "val_acc": acc,
        "loss_curve": hist,
        "wall_seconds": time.time() - t0,
    }
    with open(res, "w") as f:
        json.dump(out, f, indent=2)
    print("RESULT", json.dumps({k: out[k] for k in
          ["model", "val_bits_per_char", "val_acc", "trainable_synapses",
           "trainable_params", "wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
