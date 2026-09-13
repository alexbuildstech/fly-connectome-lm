"""FlyLM-FULL v2 — full connectome LM, fixed optimization regime.

Changes vs v1 (both motivated by diagnosed failure):
  - v1's online per-batch updates (64 examples) noise-fit the 41M-param readout:
    final val 4.55 bpc, WORSE than its own bias (unigram ~4.3 bpc).
  - v2: gradient accumulation E=4 (256 examples/update) + AdamW decoupled wd 1e-2
    (anti-interference) + single readout (Adam is scale-invariant: stacked seeds
    provably collapse to identical weights - verified in v1 ckpt).
  - v2 optionally stores 4096-d random projections (32/row sparse) every PROJ_EVERY
    positions for (a) offline-fit comparison arm, (b) linear memory probes.

Dynamics unchanged (the proven core):
  Z = (rowsum-normalized connectome @ X) * gain; Z[sensory] += B_enc[tok].T
  X = (1-leak) X + leak*tanh(Z)
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

RESULTS = "/home/z/my-project/results"
CKPT = "/home/z/my-project/data/malecns/ckpts"
DATA = "/home/z/my-project/data/malecns/processed"
torch.set_num_threads(2)

V = 65


def load_fly_csr_cached():
    z = torch.load(f"{DATA}/fly_csr_int32.pt", weights_only=False)
    return torch.sparse_csr_tensor(z["indptr"], z["indices"], z["data"], size=tuple(z["shape"].tolist()))


def load_adjacency_rcm(variant="fly", seed=0):
    if variant != "fly":
        cache = f"{DATA}/adjacency_{variant}_s{seed}.npz"
        if os.path.exists(cache):
            Araw = sp.load_npz(cache).tocsr()
        else:
            Araw0 = sp.load_npz(f"{DATA}/adjacency.npz").tocsr().astype(np.float32)
            from reservoir_lib import random_graph_like, shuffled_weights
            Araw = (random_graph_like(Araw0, seed=seed) if variant == "random"
                    else shuffled_weights(Araw0, seed=seed))
            del Araw0
            sp.save_npz(cache, Araw.astype(np.float32))
    else:
        Araw = sp.load_npz(f"{DATA}/adjacency.npz").tocsr().astype(np.float32)
    rowsum = np.asarray(Araw.sum(axis=1)).ravel()
    A = (sp.diags(1.0 / np.maximum(rowsum, 1e-6)) @ Araw).tocsr()
    del Araw
    perm_path = f"{DATA}/rcm_perm.npy"
    perm = np.load(perm_path)
    P = sp.eye(A.shape[0], format="csr")[perm, :]
    A2 = (P @ A @ P.T).tocsr().astype(np.float32)
    return A2, perm


def sensory_indices_in_perm_space(perm):
    import pandas as pd
    slim = pd.read_parquet(f"{DATA}/annotations_slim.parquet")
    body_ids = np.load(f"{DATA}/annotated_body_ids.npy")
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
    orig = np.array([id2orig[int(b)] for b in bodies if int(b) in id2orig], dtype=np.int64)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return np.sort(inv[orig]), int(keep.sum()), int(len(orig))


def to_torch_csr(A):
    return torch.sparse_csr_tensor(
        torch.from_numpy(A.indptr.astype(np.int32)),
        torch.from_numpy(A.indices.astype(np.int32)),
        torch.from_numpy(A.data), size=A.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="fly", choices=["fly", "random", "shuffled"])
    ap.add_argument("--leak", type=float, default=0.7)
    ap.add_argument("--gain", type=float, default=1.6)
    ap.add_argument("--in-gain", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-chars", type=int, default=1_051_394)
    ap.add_argument("--val-chars", type=int, default=64_000)
    ap.add_argument("--burn-in", type=int, default=200)
    ap.add_argument("--streams", type=int, default=64)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--store-proj", action="store_true")
    ap.add_argument("--proj-every", type=int, default=4)
    ap.add_argument("--max-seconds", type=int, default=100000)
    ap.add_argument("--tag", default="full2")
    args = ap.parse_args()
    t_start = time.time()

    os.makedirs(CKPT, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    tag_id = f"{args.tag}_{args.variant}_leak{args.leak}_gain{args.gain}_ing{args.in_gain}_s{args.seed}"
    ck = f"{CKPT}/flylm_{tag_id}.pt"
    res = f"{RESULTS}/flylm_{tag_id}.json"
    print(f"[{args.tag}] variant={args.variant}", flush=True)

    ids = np.load(f"{DATA}/corpus_ids.npy")
    train_ids = ids[:args.train_chars]
    val_ids = ids[args.train_chars:args.train_chars + args.val_chars]
    B = args.streams
    S = 1  # single readout (Adam scale-invariance makes multi-init seeds replicas)

    def make_chunks(id_arr):
        L = len(id_arr) // B
        return torch.from_numpy(np.stack([id_arr[k * L:(k + 1) * L] for k in range(B)]))

    TR = make_chunks(train_ids)
    VA = make_chunks(val_ids)
    Ltr, Lva = TR.shape[1], VA.shape[1]

    z = None
    if os.path.exists(ck):
        z = torch.load(ck, weights_only=False)
        phase = z["phase"]; pos = z["pos"]; vpos = z.get("vpos", 0)
        in_idx = z["in_idx"]; B_enc = z["B_enc"]
        W = z["W"]; b0 = z["b"]
        if not W.requires_grad:
            W.requires_grad_(True)
        if not b0.requires_grad:
            b0.requires_grad_(True)
        X = z["X"]; hist = z["hist"]; perm = z["perm"]
        n_sens = int(z["n_sens"])
        print(f"[{args.tag}] resumed phase={phase} pos={pos} vpos={vpos}", flush=True)
    else:
        A2, perm = load_adjacency_rcm(args.variant, args.seed)
        sens_idx, n_sens_ann, n_sens = sensory_indices_in_perm_space(perm)
        in_idx = torch.from_numpy(sens_idx.astype(np.int64))
        B_enc = (np.random.default_rng(args.seed + 1).standard_normal((V, n_sens)) * args.in_gain).astype(np.float32)
        Nn = A2.shape[0]
        W = (torch.randn(Nn, S * V, generator=torch.Generator().manual_seed(100)) * 0.05)
        W.requires_grad_(True)
        b0 = torch.zeros(S * V, requires_grad=True)
        X = torch.zeros(Nn, B)
        pos = 0; vpos = 0; hist = []
        phase = "train"
        del A2
        torch.save({"phase": phase, "pos": pos, "vpos": vpos, "in_idx": in_idx,
                    "B_enc": B_enc, "W": W.detach(), "b": b0.detach(), "opt": None,
                    "X": X, "hist": hist, "perm": perm, "n_sens": n_sens,
                    "args": vars(args)}, ck)
        print(f"[{args.tag}] fresh init n_sens={n_sens}", flush=True)

    # sparse op
    if args.variant == "fly":
        At = load_fly_csr_cached()
        nnz = int(At.values().numel())
    else:
        A2, _ = load_adjacency_rcm(args.variant, args.seed)
        At = to_torch_csr(A2)
        nnz = int(A2.nnz)
        del A2
    Nn = At.shape[0]
    B_enc_t = torch.from_numpy(B_enc) if isinstance(B_enc, np.ndarray) else B_enc

    opt = torch.optim.AdamW([W, b0], lr=args.lr, weight_decay=args.wd)
    if z is not None and z.get("opt") is not None:
        opt.load_state_dict(z["opt"])

    steps_total = Ltr - 1

    def lr_factor(step):
        wu = 120  # warmup in UPDATES
        upd = step // args.accum
        total_upd = steps_total // args.accum
        if upd < wu:
            return (upd + 1) / wu
        p = (upd - wu) / max(1, total_upd - wu)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))

    def step_batch(tok_ids):
        Zt = torch.sparse.mm(At, X) * args.gain
        Zt.index_add_(0, in_idx, B_enc_t[tok_ids].T)
        torch.tanh(Zt, out=Zt)
        X.mul_(1 - args.leak).add_(Zt, alpha=args.leak)

    def readout_logits():
        Xn = X / (X.norm(dim=0, keepdim=True) + 1e-6)
        return (Xn.T @ W + b0).view(B, S, V), Xn

    if phase == "train":
        st = torch.arange(B)
        opt.zero_grad()
        while pos < steps_total:
            tok = TR[st, pos]
            nxt = TR[st, pos + 1]
            step_batch(tok)
            logits, _ = readout_logits()
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(B * S, V), nxt.repeat(S))
            (loss / args.accum).backward()
            if (pos + 1) % args.accum == 0 or pos == steps_total - 1:
                for g in opt.param_groups:
                    g["lr"] = args.lr * lr_factor(pos)
                torch.nn.utils.clip_grad_norm_([W, b0], args.clip)
                opt.step()
                opt.zero_grad()
            if pos % args.ckpt_every == 0:
                hist.append({"pos": pos, "loss": float(loss.detach())})
                torch.save({"phase": "train", "pos": pos, "vpos": 0, "in_idx": in_idx,
                            "B_enc": B_enc, "W": W.detach(), "b": b0.detach(),
                            "opt": opt.state_dict(), "X": X, "hist": hist,
                            "perm": perm, "n_sens": n_sens, "args": vars(args)}, ck)
            if pos % 1000 == 0:
                el = time.time() - t_start
                eta = el / max(pos, 1) * (steps_total - pos)
                print(f"[{args.tag}] train {pos}/{steps_total} loss={float(loss.detach()):.4f} "
                      f"({el:.0f}s, eta {eta/60:.0f}m)", flush=True)
            pos += 1
            if time.time() - t_start > args.max_seconds:
                torch.save({"phase": "train", "pos": pos, "vpos": 0, "in_idx": in_idx,
                            "B_enc": B_enc, "W": W.detach(), "b": b0.detach(),
                            "opt": opt.state_dict(), "X": X, "hist": hist,
                            "perm": perm, "n_sens": n_sens, "args": vars(args)}, ck)
                print("budget reached", flush=True)
                return
        phase = "val"
        torch.save({"phase": phase, "pos": 0, "vpos": 0, "in_idx": in_idx,
                    "B_enc": B_enc, "W": W.detach(), "b": b0.detach(),
                    "opt": opt.state_dict(), "X": X, "hist": hist,
                    "perm": perm, "n_sens": n_sens, "args": vars(args)}, ck)
        print(f"[{args.tag}] train complete -> val", flush=True)

    if phase == "val":
        st = torch.arange(B)
        nll = torch.zeros(S); cnt = 0; correct = torch.zeros(S)
        curve = []
        F_list, Y_list = [], []
        if args.store_proj:
            K = 32
            rngP = np.random.default_rng(7)
            rows = np.repeat(np.arange(Nn, dtype=np.int64), K)
            cols = rngP.integers(0, 4096, size=rows.size, dtype=np.int64)
            vals = (rngP.choice([-1.0, 1.0], size=rows.size) / math.sqrt(K)).astype(np.float32)
            projT = sp.coo_matrix((vals, (cols, rows)), shape=(4096, Nn),
                                  dtype=np.float32).tocsr()
            Pt = torch.sparse_csr_tensor(
                torch.from_numpy(projT.indptr.astype(np.int32)),
                torch.from_numpy(projT.indices.astype(np.int32)),
                torch.from_numpy(projT.data), size=projT.shape)
        while vpos < Lva - 1:
            tok = VA[st, vpos]
            nxt = VA[st, vpos + 1]
            step_batch(tok)
            if vpos >= args.burn_in:
                with torch.no_grad():
                    logits, _ = readout_logits()
                    ce_none = torch.nn.functional.cross_entropy(
                        logits.reshape(B * S, V), nxt.repeat(S),
                        reduction="none").view(B, S).sum(0)
                    nll += ce_none.cpu()
                    correct += (logits.argmax(-1) == nxt.unsqueeze(1)).sum(0).cpu()
                cnt += B
                curve.append(float(nll[-1] / max(cnt, 1)))
                if args.store_proj and (vpos % args.proj_every == 0):
                    F_list.append(torch.sparse.mm(Pt, X).T.to(torch.float16).cpu().numpy())
                    Y_list.append(nxt.cpu().numpy())
            vpos += 1
            if vpos % 200 == 0:
                print(f"[{args.tag}] val {vpos}/{Lva-1} ({time.time()-t_start:.0f}s)", flush=True)
            if time.time() - t_start > args.max_seconds:
                torch.save({"phase": "val", "pos": 0, "vpos": vpos, "in_idx": in_idx,
                            "B_enc": B_enc, "W": W.detach(), "b": b0.detach(),
                            "opt": opt.state_dict(), "X": X, "hist": hist,
                            "perm": perm, "n_sens": n_sens, "args": vars(args),
                            "_valpartial": {"nll": nll, "cnt": cnt, "correct": correct}}, ck)
                print("budget reached (val)", flush=True)
                return
        n_val = max(cnt, 1)
        bpc = (nll / n_val / math.log(2)).numpy()
        acc = (correct / n_val).numpy()
        if F_list:
            F = np.concatenate(F_list)
            Y = np.stack(Y_list)  # (n_stored, B)
            LAGS = [0, 1, 2, 4, 8, 16]
            Yl = np.stack([np.roll(Y, k, axis=0) for k in LAGS], axis=-1)
            Yl[:16, :, 0] = -1
            np.savez_compressed(f"{RESULTS}/probes_{args.tag}_{args.variant}.npz",
                                F=F, Yl=Yl, lags=np.array(LAGS),
                                proj_every=np.array(args.proj_every))
        wall = time.time() - t_start
        spmv_flops = steps_total * nnz * 2
        gemm_flops = (steps_total / args.accum) * B * Nn * (S * V) * 2 * 2
        out = {"model": f"flylm2-{args.variant}", "tag": args.tag, "variant": args.variant,
               "bits_per_char": [float(x) for x in bpc],
               "acc": [float(x) for x in acc],
               "val_positions": int(n_val), "burn_in": args.burn_in,
               "N_neurons": int(Nn), "nnz": nnz, "n_sensory": int(n_sens),
               "leak": args.leak, "gain": args.gain, "in_gain": args.in_gain,
               "accum": args.accum, "lr": args.lr, "wd": args.wd,
               "trainable_params": int(Nn * S * V + S * V),
               "frozen_params": int(nnz),
               "train_flops_est": int(spmv_flops + gemm_flops),
               "wall_seconds": float(wall),
               "train_chars": args.train_chars, "val_chars": args.val_chars,
               "loss_curve": curve[::5]}
        with open(res, "w") as f:
            json.dump(out, f, indent=2)
        print("RESULT", json.dumps({k: out[k] for k in
              ["model", "bits_per_char", "acc", "trainable_params", "wall_seconds"]}), flush=True)
        z2 = torch.load(ck, weights_only=False)
        z2["phase"] = "done"
        torch.save(z2, ck)


if __name__ == "__main__":
    main()
