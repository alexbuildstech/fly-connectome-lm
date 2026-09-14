"""FlyLM-FULL: the complete connectome as a language model — no subsampling anywhere.

Upgrades over prior session's flylm_esn2 (the "weak version"):
  - input drive into ALL sensory neurons (was: 30,000 random neurons)
  - readout from the FULL state of all N=211,577 neurons (was: 2,048 sampled)
  - min-weight 1: ALL 26,028,386 connections kept (was: >=2 only)
  - FULL corpus: 1,051,394 train chars (was: 300,000); val 64,000
  - 3 readout seeds trained simultaneously in one sweep (stacked-weight trick)
  - RCM locality ordering + int32 CSR for 1.55x faster spmv

Dynamics (same proven core as prior session):
  Z = (A_normalized @ X) * gain ; Z[sensory] += B_enc[tok].T ; X = (1-leak) X + leak*tanh(Z)
Readout: logits = X.T @ W_all + b  (N x 195 stacked 3 seeds x 65 vocab), CE loss, Adam.

Checkpointed phases: train -> val -> done.  Repeat invocation to resume.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reservoir_lib import load_corpus

from repo_paths import CKPT, PROCESSED, RESULTS

DATA = PROCESSED
torch.set_num_threads(2)
torch.set_warn_always(False)

V = 65  # tiny shakespeare vocab


def load_fly_csr_cached():
    z = torch.load(f"{DATA}/fly_csr_int32.pt", weights_only=False)
    return torch.sparse_csr_tensor(z["indptr"], z["indices"], z["data"], size=tuple(z["shape"].tolist()))


def load_adjacency_rcm(variant="fly", seed=0):
    """Load adjacency, apply variant transform, rowsum-normalize, RCM-reorder.
    Variants: fly (real), random (config-model rewiring), shuffled (permuted synapse counts).
    Variant matrices cached to disk."""
    if variant != "fly":
        cache = f"{DATA}/adjacency_{variant}_s{seed}.npz"
        if os.path.exists(cache):
            Araw = sp.load_npz(cache).tocsr()
        else:
            Araw0 = sp.load_npz(f"{DATA}/adjacency.npz").tocsr().astype(np.float32)
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    if os.path.exists(perm_path):
        perm = np.load(perm_path)
    else:
        import scipy.sparse.csgraph as csg
        perm = csg.reverse_cuthill_mckee((A + A.T).tocsr(), symmetric_mode=True)
        np.save(perm_path, perm.astype(np.int64))
    P = sp.eye(A.shape[0], format="csr")[perm, :]
    A2 = (P @ A @ P.T).tocsr().astype(np.float32)
    return A2, perm


def sensory_indices_in_perm_space(perm):
    """Return (n_sensory,) indices into permuted space for ALL sensory neurons."""
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
        s = sup[i].lower()
        c = cls[i].lower()
        if "sensory" in s or any(k in c for k in KEY):
            keep[i] = True
    bodies = slim["bodyId"].values[keep]
    orig = np.array([id2orig[int(b)] for b in bodies if int(b) in id2orig], dtype=np.int64)
    # orig -> permuted: new index j has old index perm[j]  =>  inv[orig] = j
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return np.sort(inv[orig]), int(keep.sum()), int(len(orig))


def to_torch_csr(A):
    return torch.sparse_csr_tensor(
        torch.from_numpy(A.indptr.astype(np.int32)),
        torch.from_numpy(A.indices.astype(np.int32)),
        torch.from_numpy(A.data),
        size=A.shape)


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
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--max-seconds", type=int, default=100000)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    t_start = time.time()

    os.makedirs(CKPT, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    tag_id = f"{args.tag}_{args.variant}_leak{args.leak}_gain{args.gain}_ing{args.in_gain}_s{args.seed}"
    ck = f"{CKPT}/flylmfull_{tag_id}.pt"
    res = f"{RESULTS}/flylmfull_{tag_id}.json"
    print(f"[{args.tag}] variant={args.variant}", flush=True)

    if os.path.exists(f"{DATA}/corpus_ids.npy"):
        ids = np.load(f"{DATA}/corpus_ids.npy")
    else:
        _, ids, _, _ = load_corpus()
    train_ids = ids[:args.train_chars]
    val_ids = ids[args.train_chars:args.train_chars + args.val_chars]
    B = args.streams
    S = args.n_seeds

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
        A2 = None  # rebuilt below
        in_idx = z["in_idx"]; B_enc = z["B_enc"]
        W_all = z["W_all"]; b_all = z["b_all"]
        if not W_all.requires_grad:
            W_all.requires_grad_(True)
        if not b_all.requires_grad:
            b_all.requires_grad_(True)
        opt_st = z["opt"]; X = z["X"]
        hist = z["hist"]
        perm = z["perm"]
        n_sens = int(z["n_sens"])
        print(f"[{args.tag}] resumed phase={phase} pos={pos} vpos={vpos}", flush=True)
    else:
        A2, perm = load_adjacency_rcm(args.variant, args.seed)
        Nn = A2.shape[0]
        sens_idx, n_sens_ann, n_sens = sensory_indices_in_perm_space(perm)
        _ = Nn
        in_idx = torch.from_numpy(sens_idx.astype(np.int64))
        B_enc = (np.random.default_rng(args.seed + 1).standard_normal((V, n_sens)) * args.in_gain).astype(np.float32)
        W_all = torch.zeros(Nn, S * V)
        # init scaled to full-state norm: initial logit std ~= 0.5
        w_init = 0.05
        for s in range(S):
            g = torch.Generator().manual_seed(100 + s)
            W_all[:, s * V:(s + 1) * V] = torch.randn(Nn, V, generator=g) * w_init
        W_all.requires_grad_(True)
        b_all = torch.zeros(S * V, requires_grad=True)
        X = torch.zeros(Nn, B)
        pos = 0; vpos = 0; hist = []
        phase = "train"
        torch.save({"phase": phase, "pos": pos, "vpos": vpos, "in_idx": in_idx,
                    "B_enc": B_enc, "W_all": W_all, "b_all": b_all, "opt": None,
                    "X": X, "hist": hist, "perm": perm, "n_sens": n_sens,
                    "args": vars(args)}, ck)
        A2_meta = {"N": int(A2.shape[0]), "nnz": int(A2.nnz), "n_sensory": n_sens,
                   "n_sensory_annotated": n_sens_ann}
        print(f"[{args.tag}] fresh init: {A2_meta}", flush=True)

    # build sparse op (fast path: pre-cached int32 CSR for the fly variant)
    if args.variant == "fly":
        At = load_fly_csr_cached()
        nnz = int(At.values().numel())
    else:
        A2, perm2 = load_adjacency_rcm(args.variant, args.seed)
        At = to_torch_csr(A2)
        nnz = int(A2.nnz)
        del A2
    Nn = At.shape[0]
    B_enc_t = torch.from_numpy(B_enc) if isinstance(B_enc, np.ndarray) else B_enc

    opt = torch.optim.Adam([W_all, b_all], lr=args.lr, weight_decay=1e-4)
    if z is not None and z.get("opt") is not None:
        opt.load_state_dict(z["opt"])

    steps_total = Ltr - 1

    def lr_factor(step):
        wu = 300
        if step < wu:
            return (step + 1) / wu
        p = (step - wu) / max(1, steps_total - wu)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))

    def step_batch(tok_ids):
        Zt = torch.sparse.mm(At, X) * args.gain
        Zt.index_add_(0, in_idx, B_enc_t[tok_ids].T)
        torch.tanh(Zt, out=Zt)
        X.mul_(1 - args.leak).add_(Zt, alpha=args.leak)

    if phase == "train":
        st = torch.arange(B)
        while pos < steps_total:
            tok = TR[st, pos]
            nxt = TR[st, pos + 1]
            step_batch(tok)
            # online readout update (stacked seeds); state L2-normalized per stream
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_factor(pos)
            Xn = X / (X.norm(dim=0, keepdim=True) + 1e-6)
            logits = (Xn.T @ W_all + b_all).view(B, S, V)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(B * S, V),
                torch.stack([nxt] * S, dim=1).reshape(B * S))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W_all, b_all], args.clip)
            opt.step()
            if pos % args.ckpt_every == 0:
                hist.append({"pos": pos, "loss": float(loss.detach())})
                torch.save({"phase": "train", "pos": pos, "vpos": 0, "in_idx": in_idx,
                            "B_enc": B_enc, "W_all": W_all.detach(), "b_all": b_all.detach(),
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
                            "B_enc": B_enc, "W_all": W_all.detach(), "b_all": b_all.detach(),
                            "opt": opt.state_dict(), "X": X, "hist": hist,
                            "perm": perm, "n_sens": n_sens, "args": vars(args)}, ck)
                print("budget reached", flush=True)
                return
        phase = "val"; pos = 0
        torch.save({"phase": phase, "pos": 0, "vpos": 0, "in_idx": in_idx,
                    "B_enc": B_enc, "W_all": W_all, "b_all": b_all,
                    "opt": opt.state_dict(), "X": X, "hist": hist,
                    "perm": perm, "n_sens": n_sens, "args": vars(args)}, ck)
        print(f"[{args.tag}] train complete -> val phase", flush=True)

    if phase == "val":
        st = torch.arange(B)
        nll = torch.zeros(S); cnt = 0; correct = torch.zeros(S)
        curve = []
        # probe storage: projected features for every PROBE_EVERY val position + labels lags
        PROBE_EVERY = 2
        # fixed sparse random projection N -> 4096 (32 nonzero +/- entries per neuron)
        K_PER_ROW = 32
        rngP = np.random.default_rng(7)
        rows = np.repeat(np.arange(Nn, dtype=np.int64), K_PER_ROW)
        cols = rngP.integers(0, 4096, size=rows.size, dtype=np.int64)
        vals = (rngP.choice([-1.0, 1.0], size=rows.size) / math.sqrt(K_PER_ROW)).astype(np.float32)
        proj = sp.coo_matrix((vals, (rows, cols)), shape=(Nn, 4096), dtype=np.float32).tocsr()
        proj.sum_duplicates()
        projT = proj.T.tocsr()  # (4096, Nn)
        Pt = torch.sparse_csr_tensor(
            torch.from_numpy(projT.indptr.astype(np.int32)),
            torch.from_numpy(projT.indices.astype(np.int32)),
            torch.from_numpy(projT.data), size=projT.shape)
        F_list, Y_list, full_list = [], [], []
        X.zero_() if isinstance(X, torch.Tensor) else None
        while vpos < Lva - 1:
            tok = VA[st, vpos]
            nxt = VA[st, vpos + 1]
            step_batch(tok)
            if vpos >= args.burn_in:
                with torch.no_grad():
                    Xn = X / (X.norm(dim=0, keepdim=True) + 1e-6)
                    logits = (Xn.T @ W_all + b_all).view(B, S, V)
                    ce_none = torch.nn.functional.cross_entropy(
                        logits.reshape(B * S, V),
                        torch.stack([nxt] * S, dim=1).reshape(B * S),
                        reduction="none").view(B, S).sum(0)
                    nll += ce_none.cpu()
                    correct += (logits.argmax(-1) == nxt.unsqueeze(1)).sum(0).cpu()
                cnt += B
                curve.append(float(nll[-1] / max(cnt, 1)))
                if (vpos - args.burn_in) % PROBE_EVERY == 0:
                    F_list.append(torch.sparse.mm(Pt, X).T.to(torch.float16).cpu().numpy())
                    Y_list.append(nxt.cpu().numpy())
                if vpos >= Lva - 4:
                    full_list.append(X.T.to(torch.float16).cpu().numpy())
            vpos += 1
            if vpos % 100 == 0:
                print(f"[{args.tag}] val {vpos}/{Lva-1} ({time.time()-t_start:.0f}s)", flush=True)
            if time.time() - t_start > args.max_seconds:
                return
        n_val = max(cnt, 1)
        bpc = (nll / n_val / math.log(2)).numpy()
        acc = (correct / n_val).numpy()
        F = np.concatenate(F_list) if F_list else np.zeros((0, 4096), np.float16)
        Y = np.concatenate(Y_list) if Y_list else np.zeros((0,), np.int16)
        np.savez_compressed(f"{RESULTS}/probes_{args.tag}_{args.variant}.npz",
                            F=F, Y=Y, probe_every=np.array(PROBE_EVERY))
        if full_list:
            np.savez_compressed(f"{RESULTS}/fullstate_{args.tag}_{args.variant}.npz",
                                F=np.concatenate(full_list))
        wall = time.time() - t_start
        spmv_flops = steps_total * nnz * 2
        gemm_flops = steps_total * B * Nn * (S * V) * 2 * 2
        out = {"model": f"flylm-full-{args.variant}", "tag": args.tag, "variant": args.variant,
               "seeds": list(range(S)), "bits_per_char": [float(x) for x in bpc],
               "nats": [float(x / math.log(2)) for x in bpc],
               "acc": [float(x) for x in acc],
               "val_positions": int(n_val), "burn_in": args.burn_in,
               "N_neurons": int(Nn), "nnz": nnz, "n_sensory": int(n_sens),
               "leak": args.leak, "gain": args.gain, "in_gain": args.in_gain,
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
