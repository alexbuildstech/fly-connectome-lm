"""FlyLM — full-brain Echo-State language model on the REAL MaleCNS v1.0 connectome.

CHECKPOINTED version: the run is a phase machine (train -> val_moments -> lambda ->
final_eval -> done) that saves state to disk and exits when --max-seconds is reached.
Call the same command repeatedly until it prints RESULT line / writes the output JSON.
This is required because background processes do not survive between tool calls on this
box; each tool call is a ~10-minute compute window.

Usage (repeat until done):
  python flylm_esn.py --variant fly --norm global --leak 0.7 --gain 1.2 --min-weight 2 \
      --seed 0 --train-chars 300000 --tag main
"""
import os as _os
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repo root
import argparse, json, os, sys, time
import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reservoir_lib import (load_corpus, load_adjacency, normalize_connectome,
                           random_graph_like, shuffled_weights, DATA)

RESULTS = f"{_R}/fly-connectome-lm/results"
CKPT = f"{_R}/data/malecns/ckpts"
torch.set_num_threads(2)


def to_torch_csr(A):
    return torch.sparse_csr_tensor(
        torch.from_numpy(A.indptr.astype(np.int64)),
        torch.from_numpy(A.indices.astype(np.int64)),
        torch.from_numpy(A.data.astype(np.float32)),
        size=A.shape)


def pick_input_neurons(body_ids, N, n_in, seed=0, pool="random_active", active=None):
    """Injection pool: 'sensory' (real sensory neurons) or 'random_active' (random neurons
    with >=1 incoming edge — the "modified fly" that stimulates the recurrent core directly)."""
    rng = np.random.default_rng(seed)
    if pool == "sensory":
        import pandas as pd
        ann = pd.read_parquet(f"{DATA}/annotations_slim.parquet")[["bodyId", "superclass", "class"]]
        sup = dict(zip(ann.bodyId.values, ann.superclass.fillna("").astype(str).values))
        cls = dict(zip(ann.bodyId.values, ann["class"].fillna("").astype(str).values))
        sens = np.array([i for i, b in enumerate(body_ids)
                         if "sensory" in sup.get(b, "") or "sensory" in cls.get(b, "")])
        idx = sens
    else:
        idx = active
    if len(idx) < 256:
        pad = rng.choice(N, 256 - len(idx), replace=False)
        idx = np.concatenate([idx, pad])
    if len(idx) > n_in:
        idx = rng.choice(idx, n_in, replace=False)
    return np.sort(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="fly", choices=["fly", "random", "shuffled"])
    ap.add_argument("--norm", default="row", choices=["global", "deg", "log", "row"])
    ap.add_argument("--pool", default="random_active", choices=["random_active", "sensory"])
    ap.add_argument("--pool-size", type=int, default=30000)
    ap.add_argument("--leak", type=float, default=0.7)
    ap.add_argument("--gain", type=float, default=1.2)
    ap.add_argument("--in-gain", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-chars", type=int, default=300_000)
    ap.add_argument("--val-chars", type=int, default=40_000)
    ap.add_argument("--burn-in", type=int, default=200)
    ap.add_argument("--streams", type=int, default=64)
    ap.add_argument("--in-neurons", type=int, default=1024)
    ap.add_argument("--readout", type=int, default=2048)
    ap.add_argument("--subsample", type=int, default=3)
    ap.add_argument("--min-weight", type=int, default=1)
    ap.add_argument("--max-seconds", type=int, default=480)
    ap.add_argument("--tag", default="dev")
    args = ap.parse_args()
    t_start = time.time()

    os.makedirs(CKPT, exist_ok=True)
    tag_id = f"{args.tag}_{args.variant}_{args.norm}_leak{args.leak}_gain{args.gain}_ing{args.in_gain}_{args.pool}{args.pool_size}_s{args.seed}_mw{args.min_weight}_tc{args.train_chars}"
    ck = f"{CKPT}/flylm_{tag_id}.npz"

    text, ids, _, _ = load_corpus()
    V = len(set(text))
    train_ids = ids[:args.train_chars]
    val_ids = ids[args.train_chars:args.train_chars + args.val_chars]
    B = args.streams
    D = args.readout + 1  # state features + 1 bias feature
    st_t = torch.arange(B)

    def make_chunks(id_arr):
        L = len(id_arr) // B
        return np.stack([id_arr[k * L:(k + 1) * L] for k in range(B)])

    TR = torch.from_numpy(make_chunks(train_ids))
    VA = torch.from_numpy(make_chunks(val_ids))
    Ltr, Lva = TR.shape[1], VA.shape[1]
    T_train_total = Ltr - 1 - args.burn_in
    T_val_total = Lva - 1 - args.burn_in

    # ---------- load or init checkpoint ----------
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=True)
        phase = str(z["phase"])
        G = torch.from_numpy(z["G"]); C = torch.from_numpy(z["C"])
        X = torch.from_numpy(z["X"]); pos = int(z["pos"])
        n_collect = int(z["n_collect"])
        Gv = torch.from_numpy(z["Gv"]) if "Gv" in z else None
        Cv = torch.from_numpy(z["Cv"]) if "Cv" in z else None
        lam_scores = dict(z["lam_scores"].item()) if "lam_scores" in z else {}
        print(f"[{args.tag}] resumed {tag_id} at phase={phase} pos={pos}", flush=True)
    else:
        # fresh init: build metadata, save empty checkpoint (matrix built below, shared path)
        A_probe = load_adjacency()
        Nn = A_probe.shape[0]
        del A_probe
        body_ids = np.load(f"{DATA}/annotated_body_ids.npy")
        A_probe = load_adjacency()
        indeg = A_probe.getnnz(axis=0)
        del A_probe
        active = np.where(indeg >= 1)[0]
        pool_n = args.pool_size if args.pool == "random_active" else args.in_neurons
        in_idx_np = pick_input_neurons(body_ids, Nn, pool_n, seed=args.seed,
                                       pool=args.pool, active=active)
        n_in = len(in_idx_np)
        B_enc = (np.random.default_rng(args.seed + 1).standard_normal((V, n_in)) * args.in_gain).astype(np.float32)
        # feature neurons: sample from neurons that actually RECEIVE connections
        A_probe = load_adjacency()
        indeg = A_probe.getnnz(axis=0)
        del A_probe
        active = np.where(indeg >= 1)[0]
        feat_np = np.sort(np.random.default_rng(args.seed + 2).choice(active, D - 1, replace=False))
        phase = "train"
        G = torch.zeros(D, D, dtype=torch.float64)
        C = torch.zeros(D, V, dtype=torch.float64)
        Gv = Cv = None
        X = torch.zeros(Nn, B)
        pos = 0
        n_collect = 0
        lam_scores = {}
        np.savez(ck, phase=phase, G=G.numpy(), C=C.numpy(), X=X.numpy(), pos=pos,
                 n_collect=n_collect, in_idx=in_idx_np, B_enc=B_enc, feat_idx=feat_np,
                 lam_scores=np.array(lam_scores, dtype=object), N=Nn)
        print(f"[{args.tag}] fresh init saved", flush=True)

    # reload runtime tensors from ckpt each call ( matrices not persisted across phases )
    z = np.load(ck, allow_pickle=True)
    in_idx_np = z["in_idx"]; B_enc = z["B_enc"]; feat_np = z["feat_idx"]; Nn = int(z["N"])
    in_idx = torch.from_numpy(in_idx_np.astype(np.int64))
    B_enc_t = torch.from_numpy(B_enc)
    feat = torch.from_numpy(feat_np.astype(np.int64))
    n_in = len(in_idx_np)

    # rebuild the sparse matrix each call (fast: load npz + normalize)
    A_raw = load_adjacency()
    if args.min_weight > 1:
        coo = A_raw.tocoo()
        keep = coo.data >= args.min_weight
        A_raw = sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])),
                              shape=A_raw.shape).tocsr()
        del coo
    if args.variant == "fly":
        A = normalize_connectome(A_raw, args.norm, seed=args.seed)
    elif args.variant == "random":
        A = normalize_connectome(random_graph_like(A_raw, seed=args.seed), args.norm, seed=args.seed)
    else:
        A = normalize_connectome(shuffled_weights(A_raw, seed=args.seed), args.norm, seed=args.seed)
    del A_raw
    At = to_torch_csr(A)
    del A

    def step_batch(tok_ids):
        nonlocal pos
        tok = tok_ids if torch.is_tensor(tok_ids) else torch.as_tensor(tok_ids)
        Z = torch.sparse.mm(At, X) * args.gain
        Z.index_add_(0, in_idx, B_enc_t[tok].T)
        X.mul_(1 - args.leak).add_(torch.tanh(Z), alpha=args.leak)
        pos += 1

    def reset_streams(chunks, burn):
        pos = 0
        X.zero_()
        for _ in range(burn):
            step_batch(chunks[st_t, pos])

    def save(phase_name, **extra):
        np.savez(ck, phase=phase_name, G=G.numpy(), C=C.numpy(), X=X.numpy(), pos=pos,
                 n_collect=n_collect, in_idx=in_idx_np, B_enc=B_enc, feat_idx=feat_np,
                 lam_scores=np.array(lam_scores, dtype=object), N=Nn,
                 **extra)

    done = False
    if phase == "train":
        started = pos > 0
        if not started:
            reset_streams(TR, args.burn_in)   # advances pos to burn_in, X warmed
        # X / pos are already at the right point on resume; just continue
        t0 = time.time()
        while pos < T_train_total:
            tok = TR[st_t, pos]
            nxt = TR[st_t, pos + 1]
            step_batch(tok)
            if (pos - args.burn_in) % args.subsample == 0 and pos >= args.burn_in:
                F = torch.cat([X[feat].T.double(), torch.ones(B, 1, dtype=torch.float64)], dim=1)
                Y = torch.zeros(B, V, dtype=torch.float64)
                Y[st_t, nxt] = 1.0
                G += F.T @ F
                C += F.T @ Y
                n_collect += B
            if pos % 400 == 0:
                el = time.time() - t0
                print(f"[{args.tag}] train {pos}/{T_train_total} (+{el:.0f}s)", flush=True)
            if time.time() - t_start > args.max_seconds:
                break
        if pos >= T_train_total:
            phase = "val_moments"
            pos = 0
            save(phase)
        else:
            save("train")
            done = True

    if phase == "val_moments" and not done:
        if Gv is None:
            Gv = torch.zeros(D, D, dtype=torch.float64)
            Cv = torch.zeros(D, V, dtype=torch.float64)
        zr = np.load(ck, allow_pickle=True)
        vpos = int(zr["vpos"]) if "vpos" in zr else 0
        if vpos == 0:
            reset_streams(VA, args.burn_in)
        t0 = time.time()
        while vpos < T_val_total:
            tok = VA[st_t, vpos]
            nxt = VA[st_t, vpos + 1]
            step_batch(tok)
            if vpos % args.subsample == 0:
                F = torch.cat([X[feat].T.double(), torch.ones(B, 1, dtype=torch.float64)], dim=1)
                Y = torch.zeros(B, V, dtype=torch.float64)
                Y[st_t, nxt] = 1.0
                Gv += F.T @ F
                Cv += F.T @ Y
            vpos += 1
            if time.time() - t_start > args.max_seconds:
                break
        if vpos >= T_val_total:
            phase = "lambda"
            pos = 0
            save(phase, Gv=Gv.numpy(), Cv=Cv.numpy())
        else:
            save("val_moments", Gv=Gv.numpy(), Cv=Cv.numpy(), vpos=vpos)
            done = True

    if phase == "lambda" and not done:
        # solve all lambdas, evaluate exact val NLL for each in ONE pass, pick best by NLL
        Gs, Cs = G.numpy(), C.numpy()
        lams = [3e-5, 3e-4, 3e-3, 3e-2, 3e-1, 3.0]
        Ws = np.stack([np.linalg.solve(Gs + lam * np.eye(D), Cs) for lam in lams]).astype(np.float32)
        Wt_all = torch.from_numpy(Ws)                      # (L, D, V)
        reset_streams(VA, args.burn_in)   # deterministic full re-roll (cheap: ~300 steps)
        nlls = np.zeros(len(lams)); cnt = 0
        Fv = []          # (rows, D)
        yv = []          # (rows,)
        for t in range(T_val_total):
            tok = VA[st_t, pos]
            nxt = VA[st_t, pos + 1]
            step_batch(tok)
            if t % args.subsample == 0:
                Fv.append(torch.cat([X[feat].T, torch.ones(B, 1)], dim=1))
                yv.append(nxt.clone())
            if time.time() - t_start > args.max_seconds:
                break
        if len(Fv) and pos >= T_val_total:
            F = torch.cat(Fv, dim=0)                       # (rows, D)
            yy = torch.cat(yv, dim=0)                      # (rows,)
            logits = torch.einsum("rd,ldv->lrv", F, Wt_all)  # BLAS-backed bmm
            logits -= logits.max(dim=2, keepdim=True).values
            p = torch.softmax(logits, dim=2)
            for li in range(len(lams)):
                nlls[li] = float(-torch.log(p[li, torch.arange(len(yy)), yy] + 1e-12).mean())
            cnt = len(yy)
            for lam, nl in zip(lams, nlls):
                lam_scores[str(lam)] = nl
                print(f"[{args.tag}] lambda={lam:g} val_nll={nl:.4f} nats", flush=True)
            lam_best = min(lam_scores, key=lam_scores.get)
            W = np.linalg.solve(Gs + float(lam_best) * np.eye(D), Cs)
            phase = "final_eval"
            pos = 0
            save(phase, W=W.astype(np.float32), lam_best=np.array(lam_best))
            done = True
        else:
            save("lambda")
            done = True

    if phase == "final_eval" and not done:
        z = np.load(ck, allow_pickle=True)
        W = torch.from_numpy(z["W"])
        lam_best = str(z["lam_best"].item()) if hasattr(z["lam_best"], "item") else str(z["lam_best"])
        Gs = G.numpy()
        lam_scores = {k: float(v) for k, v in lam_scores.items()}
        lam_best_val = float(min(lam_scores, key=lam_scores.get))
        Wt = W.float()
        reset_streams(VA, args.burn_in)
        nll, cnt, correct = 0.0, 0, 0
        t0 = time.time()
        for t in range(T_val_total):
            tok = VA[st_t, pos]
            nxt = VA[st_t, pos + 1]
            step_batch(tok)
            if t % args.subsample == 0:
                F = torch.cat([X[feat].T, torch.ones(B, 1)], dim=1)
                logits = F @ Wt
                logits -= logits.max(dim=1, keepdim=True).values
                p = torch.softmax(logits, dim=1)
                nll += float(-torch.log(p[st_t, nxt] + 1e-12).sum())
                correct += int((logits.argmax(dim=1) == nxt).sum())
                cnt += B
            if time.time() - t_start > args.max_seconds:
                print(f"[{args.tag}] WARNING: final eval exceeded budget, partial results",
                      flush=True)
                break
        res = {
            "variant": args.variant, "norm": args.norm, "leak": args.leak, "gain": args.gain,
            "in_gain": args.in_gain, "seed": args.seed, "train_chars": args.train_chars,
            "val_chars": args.val_chars, "streams": B, "in_neurons": int(n_in),
            "readout": D, "subsample": args.subsample, "ridge_samples": int(n_collect),
            "min_weight": args.min_weight,
            "val_nll_nats_per_char": nll / cnt,
            "val_bits_per_char": (nll / cnt) / np.log(2),
            "val_acc": correct / cnt,
            "lambda": lam_best_val,
            "lambda_grid_val_mse": lam_scores,
            "input_neuron_source": f"{args.pool} (n={n_in})",
            "wall_seconds": time.time() - t_start,
        }
        os.makedirs(RESULTS, exist_ok=True)
        out = f"{RESULTS}/flylm_{args.tag}_{args.variant}_{args.norm}_leak{args.leak}_gain{args.gain}_{args.pool}{args.pool_size}_s{args.seed}.json"
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        np.savez(f"{RESULTS}/flylm_state_{args.tag}_{args.variant}_{args.norm}_leak{args.leak}_gain{args.gain}_s{args.seed}.npz",
                 W=W.numpy(), feat_idx=feat_np, in_idx=in_idx_np, B_enc=B_enc,
                 leak=args.leak, gain=args.gain, lam=lam_best_val)
        print("RESULT " + json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}))
        print(f"[{args.tag}] saved {out}", flush=True)
        phase = "done"
        save(phase)

    if phase == "done":
        print(f"[{args.tag}] DONE", flush=True)


if __name__ == "__main__":
    main()
