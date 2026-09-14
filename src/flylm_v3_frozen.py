"""FlyLM v3 frozen — session-4 "fair-fight" upgrades to the frozen-reservoir protocol,
each isolating ONE deficiency named in the external critique of v2:

  --variant flysigned|shuffledsigned|randomsigned : E/I signs from the neurotransmitter
      file (critique #2: "no inhibitory sign — a connectome without signs is a weighted
      averaging graph"). Sign of an edge = sign of its presynaptic neuron; inhibitory =
      {gaba, histamine}. Rows are |w|-normalized so gain retains its meaning.
  --leak multi:0.3,0.7,0.99 : per-neuron time constants, assigned in RCM-band thirds
      (critique #3/#4: "single leak = the whole memory story"; slow third extends the
      possible horizon to 0.99^lag instead of 0.3^lag).
  --delay-frac 0.3 : a random 30% of edges (post-normalization) deliver their input one
      character late (critique #4: "no delays").
  Everything else is byte-for-byte the v2 protocol (B=64 streams, rowsum-normalized
  operator, gain 1.6, in-gain 2.0, gradient-accum 4, AdamW, 1,051,394 train chars) so
  results are directly comparable to the committed v2 numbers.

Stores 4096-d fp16 random projections of the state every 4 val positions (lags up to 64)
for offline memory-horizon probes and the nonlinear-readout arm.
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
from flylm_full2 import sensory_indices_in_perm_space

torch.set_num_threads(2)
V = 65
SIGNED = ("flysigned", "shuffledsigned", "randomsigned")


def load_adjacency_v3(variant, seed=0):
    """Returns (A2 row/rowabs-normalized in RCM space, perm). Unsigned path = v2's."""
    if variant in SIGNED:
        Araw = sp.load_npz(f"{PROCESSED}/adjacency_{variant}_s{seed}.npz").tocsr().astype(np.float32)
        absr = np.abs(Araw)
        rowsum = np.asarray(absr.sum(axis=1)).ravel()
        A = (sp.diags(1.0 / np.maximum(rowsum, 1e-6)) @ Araw).tocsr()
        del Araw, absr
    else:
        if variant != "fly":
            cache = f"{PROCESSED}/adjacency_{variant}_s{seed}.npz"
            Araw = sp.load_npz(cache).tocsr().astype(np.float32)
        else:
            Araw = sp.load_npz(f"{PROCESSED}/adjacency.npz").tocsr().astype(np.float32)
        rowsum = np.asarray(Araw.sum(axis=1)).ravel()
        A = (sp.diags(1.0 / np.maximum(rowsum, 1e-6)) @ Araw).tocsr()
        del Araw
    perm = np.load(f"{PROCESSED}/rcm_perm.npy")
    P = sp.eye(A.shape[0], format="csr")[perm, :]
    A2 = (P @ A @ P.T).tocsr().astype(np.float32)
    return A2, perm


def to_torch_csr(A):
    return torch.sparse_csr_tensor(
        torch.from_numpy(A.indptr.astype(np.int32)),
        torch.from_numpy(A.indices.astype(np.int32)),
        torch.from_numpy(A.data), size=A.shape)


def parse_leak(s, N):
    """'0.7' -> scalar path; 'multi:0.3,0.7,0.99' -> per-neuron vector (RCM thirds)."""
    if s.startswith("multi:"):
        vals = [float(x) for x in s[6:].split(",")]
        seg = N // len(vals)
        lv = np.zeros(N, dtype=np.float32)
        for k, v in enumerate(vals):
            lo = k * seg
            hi = N if k == len(vals) - 1 else (k + 1) * seg
            lv[lo:hi] = v
        return None, torch.from_numpy(lv).unsqueeze(1), "+".join(f"{v:g}" for v in vals)
    return float(s), None, f"{float(s):g}"


def split_delays(A, frac, seed=0):
    """Post-normalization edge split: (1-frac) fast, frac one-step-delayed."""
    coo = A.tocoo()
    rng = np.random.default_rng(seed)
    m = rng.random(coo.nnz) >= frac
    N = A.shape[0]
    Af = sp.coo_matrix((coo.data[m], (coo.row[m], coo.col[m])), shape=(N, N)).tocsr()
    Ad = sp.coo_matrix((coo.data[~m], (coo.row[~m], coo.col[~m])), shape=(N, N)).tocsr()
    return Af, Ad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="fly",
                    choices=["fly", "random", "shuffled", "flysigned",
                             "shuffledsigned", "randomsigned"])
    ap.add_argument("--leak", default="0.7")
    ap.add_argument("--gain", type=float, default=1.6)
    ap.add_argument("--in-gain", type=float, default=2.0)
    ap.add_argument("--delay-frac", type=float, default=0.0)
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
    ap.add_argument("--store-proj", action="store_true", default=True)
    ap.add_argument("--proj-every", type=int, default=4)
    ap.add_argument("--max-seconds", type=int, default=100000)
    ap.add_argument("--tag", default="v3")
    args = ap.parse_args()
    t_start = time.time()

    os.makedirs(CKPT, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    leak_scalar, leak_vec, leak_name = None, None, None  # set after N known
    tag_id = (f"{args.tag}_{args.variant}_leak{args.leak.replace(':', '-').replace(',', '_')}"
              f"_dl{args.delay_frac}_gain{args.gain}_s{args.seed}")
    ck = f"{CKPT}/flylm_{tag_id}.pt"
    res = f"{RESULTS}/flylm_{tag_id}.json"
    print(f"[{args.tag}] variant={args.variant} leak={args.leak} delay={args.delay_frac}", flush=True)

    ids = np.load(f"{PROCESSED}/corpus_ids.npy")
    train_ids = ids[:args.train_chars]
    val_ids = ids[args.train_chars:args.train_chars + args.val_chars]
    B = args.streams
    S = 1

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
        X = z["X"]; Xprev = z.get("Xprev", X.clone())
        hist = z["hist"]; perm = z["perm"]; n_sens = int(z["n_sens"])
        leak_scalar, leak_vec, leak_name = z["leak_scalar"], z["leak_vec"], z["leak_name"]
        print(f"[{args.tag}] resumed phase={phase} pos={pos} vpos={vpos}", flush=True)
    else:
        A2, perm = load_adjacency_v3(args.variant, args.seed)
        Nn = A2.shape[0]
        leak_scalar, leak_vec, leak_name = parse_leak(args.leak, Nn)
        sens_idx, _, n_sens = sensory_indices_in_perm_space(perm)
        in_idx = torch.from_numpy(sens_idx.astype(np.int64))
        B_enc = (np.random.default_rng(args.seed + 1).standard_normal((V, n_sens))
                 * args.in_gain).astype(np.float32)
        W = (torch.randn(Nn, S * V, generator=torch.Generator().manual_seed(100)) * 0.05)
        W.requires_grad_(True)
        b0 = torch.zeros(S * V, requires_grad=True)
        X = torch.zeros(Nn, B)
        Xprev = torch.zeros(Nn, B)
        pos = 0; vpos = 0; hist = []
        phase = "train"
        del A2
        torch.save({"phase": phase, "pos": pos, "vpos": vpos, "in_idx": in_idx,
                    "B_enc": B_enc, "W": W.detach(), "b": b0.detach(), "opt": None,
                    "X": X, "Xprev": Xprev, "hist": hist, "perm": perm,
                    "n_sens": n_sens, "leak_scalar": leak_scalar, "leak_vec": leak_vec,
                    "leak_name": leak_name, "args": vars(args)}, ck)
        print(f"[{args.tag}] fresh init n_sens={n_sens} leak={leak_name}", flush=True)

    # graph(s)
    A2, perm2 = load_adjacency_v3(args.variant, args.seed)
    Nn = A2.shape[0]
    if leak_vec is None and leak_scalar is None:
        leak_scalar, leak_vec, leak_name = parse_leak(args.leak, Nn)
    if args.delay_frac > 0:
        Af, Ad = split_delays(A2, args.delay_frac, seed=args.seed)
        At_fast, At_del = to_torch_csr(Af), to_torch_csr(Ad)
        nnz = int(Af.nnz + Ad.nnz)
        del Af, Ad
    else:
        At_fast, At_del = to_torch_csr(A2), None
        nnz = int(A2.nnz)
    del A2
    B_enc_t = torch.from_numpy(B_enc) if isinstance(B_enc, np.ndarray) else B_enc
    lvec = leak_vec  # (N,1) or None
    lsc = leak_scalar

    opt = torch.optim.AdamW([W, b0], lr=args.lr, weight_decay=args.wd)
    if z is not None and z.get("opt") is not None:
        opt.load_state_dict(z["opt"])

    steps_total = Ltr - 1

    def lr_factor(step):
        wu = 120
        upd = step // args.accum
        total_upd = steps_total // args.accum
        if upd < wu:
            return (upd + 1) / wu
        p = (upd - wu) / max(1, total_upd - wu)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))

    def step_batch(tok):
        # returns nothing; updates X (and Xprev when delays on)
        if At_del is not None:
            Zt = (torch.sparse.mm(At_fast, X) + torch.sparse.mm(At_del, Xprev)) * args.gain
        else:
            Zt = torch.sparse.mm(At_fast, X) * args.gain
        Zt.index_add_(0, in_idx, B_enc_t[tok_ids_holder[0]].T)
        torch.tanh(Zt, out=Zt)
        if lvec is not None:
            Zt.mul_(lvec)
            X.mul_(1 - lvec)
            X.add_(Zt)
        else:
            X.mul_(1 - lsc).add_(Zt, alpha=lsc)
        if At_del is not None:
            Xprev.copy_(X_old[0])

    tok_ids_holder = [None]
    X_old = [None]

    def step_batch_v(tok):
        tok_ids_holder[0] = tok
        X_old[0] = X.clone()
        step_batch(tok)

    def readout_logits():
        Xn = X / (X.norm(dim=0, keepdim=True) + 1e-6)
        return (Xn.T @ W + b0).view(B, S, V), Xn

    if phase == "train":
        st = torch.arange(B)
        opt.zero_grad()
        while pos < steps_total:
            tok = TR[st, pos]
            nxt = TR[st, pos + 1]
            step_batch_v(tok)
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
                            "opt": opt.state_dict(), "X": X, "Xprev": Xprev,
                            "hist": hist, "perm": perm, "n_sens": n_sens,
                            "leak_scalar": leak_scalar, "leak_vec": leak_vec,
                            "leak_name": leak_name, "args": vars(args)}, ck)
            if pos % 1000 == 0:
                el = time.time() - t_start
                eta = el / max(pos, 1) * (steps_total - pos)
                print(f"[{args.tag}] train {pos}/{steps_total} loss={float(loss.detach()):.4f} "
                      f"({el:.0f}s, eta {eta/60:.0f}m)", flush=True)
            pos += 1
            if time.time() - t_start > args.max_seconds:
                torch.save({"phase": "train", "pos": pos, "vpos": 0, "in_idx": in_idx,
                            "B_enc": B_enc, "W": W.detach(), "b": b0.detach(),
                            "opt": opt.state_dict(), "X": X, "Xprev": Xprev,
                            "hist": hist, "perm": perm, "n_sens": n_sens,
                            "leak_scalar": leak_scalar, "leak_vec": leak_vec,
                            "leak_name": leak_name, "args": vars(args)}, ck)
                print("budget reached", flush=True)
                return
        phase = "val"
        torch.save({"phase": phase, "pos": 0, "vpos": 0, "in_idx": in_idx,
                    "B_enc": B_enc, "W": W.detach(), "b": b0.detach(),
                    "opt": opt.state_dict(), "X": X, "Xprev": Xprev,
                    "hist": hist, "perm": perm, "n_sens": n_sens,
                    "leak_scalar": leak_scalar, "leak_vec": leak_vec,
                    "leak_name": leak_name, "args": vars(args)}, ck)
        print(f"[{args.tag}] train complete -> val", flush=True)

    if phase == "val":
        st = torch.arange(B)
        nll = torch.zeros(S); cnt = 0; correct = torch.zeros(S)
        curve = []
        F_list, Y_list = [], []
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
            step_batch_v(tok)
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
                            "opt": opt.state_dict(), "X": X, "Xprev": Xprev,
                            "hist": hist, "perm": perm, "n_sens": n_sens,
                            "leak_scalar": leak_scalar, "leak_vec": leak_vec,
                            "leak_name": leak_name, "args": vars(args),
                            "_valpartial": {"nll": nll, "cnt": cnt, "correct": correct}}, ck)
                print("budget reached (val)", flush=True)
                return
        n_val = max(cnt, 1)
        bpc = (nll / n_val / math.log(2)).numpy()
        acc = (correct / n_val).numpy()
        if F_list:
            F = np.concatenate(F_list)
            Y = np.stack(Y_list)
            LAGS = [0, 1, 2, 4, 8, 16, 32, 64]
            LAGS = [k for k in LAGS if k * args.proj_every < len(Y) - 16]
            Yl = np.stack([np.roll(Y, k * args.proj_every, axis=0) for k in LAGS], axis=-1)
            Yl[:16, :, 0] = -1
            np.savez_compressed(f"{RESULTS}/probes_{args.tag}_{args.variant}_{leak_name}"
                                f"_dl{args.delay_frac}.npz",
                                F=F, Yl=Yl, lags=np.array(LAGS),
                                proj_every=np.array(args.proj_every))
        wall = time.time() - t_start
        out = {"model": f"flylm-v3frozen-{args.variant}", "tag": args.tag,
               "variant": args.variant,
               "bits_per_char": [float(x) for x in bpc],
               "acc": [float(x) for x in acc],
               "val_positions": int(n_val), "burn_in": args.burn_in,
               "N_neurons": int(Nn), "nnz": nnz, "n_sensory": int(n_sens),
               "leak": leak_name, "gain": args.gain, "in_gain": args.in_gain,
               "delay_frac": args.delay_frac,
               "signed": args.variant in SIGNED,
               "accum": args.accum, "lr": args.lr, "wd": args.wd,
               "trainable_params": int(Nn * S * V + S * V),
               "frozen_params": int(nnz),
               "wall_seconds": float(wall),
               "train_chars": args.train_chars, "val_chars": args.val_chars,
               "loss_curve": curve[::5]}
        with open(res, "w") as f:
            json.dump(out, f, indent=2)
        print("RESULT", json.dumps({k: out[k] for k in
              ["model", "bits_per_char", "acc", "leak", "delay_frac", "wall_seconds"]}),
              flush=True)
        z2 = torch.load(ck, weights_only=False)
        z2["phase"] = "done"
        torch.save(z2, ck)


if __name__ == "__main__":
    main()
