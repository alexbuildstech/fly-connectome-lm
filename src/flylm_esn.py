"""FlyLM — full-brain Echo-State language model on the REAL MaleCNS v1.0 connectome.

tokens -> (fixed random encoder into ~1024 real SENSORY neurons) -> frozen fly wiring
(A scaled to spectral radius 1, leaky-tanh recurrent dynamics) -> state -> ridge readout
-> next-char distribution.

Variants (--variant): fly | random | shuffled  (identical dynamics/size/protocol).
Dynamics run on torch sparse CSR (multithreaded CPU). Ridge stats accumulated streaming;
lambda selected by val MSE (computable from second moments), exact NLL for the winner.

Usage:
  python flylm_esn.py --variant fly --norm global --leak 0.7 --gain 1.2 --seed 0 \
      --train-chars 300000 --streams 64 --tag main
"""
import argparse, json, os, sys, time
import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reservoir_lib import (load_corpus, load_adjacency, normalize_connectome,
                           random_graph_like, shuffled_weights, DATA)

RESULTS = "/home/z/my-project/fly-connectome-lm/results"
torch.set_num_threads(2)


def to_torch_csr(A):
    return torch.sparse_csr_tensor(
        torch.from_numpy(A.indptr.astype(np.int64)),
        torch.from_numpy(A.indices.astype(np.int64)),
        torch.from_numpy(A.data.astype(np.float32)),
        size=A.shape)


def pick_input_neurons(body_ids, N, n_in, seed=0):
    """Prefer REAL sensory neurons (superclass/class containing 'sensory')."""
    import pandas as pd
    ann = pd.read_parquet(f"{DATA}/annotations_slim.parquet")[["bodyId", "superclass", "class"]]
    sup = dict(zip(ann.bodyId.values, ann.superclass.fillna("").astype(str).values))
    cls = dict(zip(ann.bodyId.values, ann["class"].fillna("").astype(str).values))
    sens = np.zeros(N, dtype=bool)
    for i, b in enumerate(body_ids):
        if ("sensory" in sup.get(b, "")) or ("sensory" in cls.get(b, "")):
            sens[i] = True
    idx = np.where(sens)[0]
    rng = np.random.default_rng(seed)
    if len(idx) < 256:
        pad = rng.choice(N, 256 - len(idx), replace=False)
        idx = np.concatenate([idx, pad])
    if len(idx) > n_in:
        idx = rng.choice(idx, n_in, replace=False)
    return np.sort(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="fly", choices=["fly", "random", "shuffled"])
    ap.add_argument("--norm", default="global", choices=["global", "deg", "log"])
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
    ap.add_argument("--min-weight", type=int, default=1,
                    help="keep only connections with >= k synapses (1 = full connectome)")
    ap.add_argument("--tag", default="dev")
    args = ap.parse_args()
    t_start = time.time()

    text, ids, _, _ = load_corpus()
    V = len(set(text))
    train_ids = ids[:args.train_chars]
    val_ids = ids[args.train_chars:args.train_chars + args.val_chars]

    A_raw = load_adjacency()
    if args.min_weight > 1:
        coo = A_raw.tocoo()
        keep = coo.data >= args.min_weight
        A_raw = sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])),
                              shape=A_raw.shape).tocsr()
        del coo
        print(f"[{args.tag}] pruned to weight>={args.min_weight}: nnz={A_raw.nnz}", flush=True)
    if args.variant == "fly":
        A = normalize_connectome(A_raw, args.norm, seed=args.seed)
    elif args.variant == "random":
        A = normalize_connectome(random_graph_like(A_raw, seed=args.seed), args.norm, seed=args.seed)
    else:
        A = normalize_connectome(shuffled_weights(A_raw, seed=args.seed), args.norm, seed=args.seed)
    del A_raw
    N = A.shape[0]
    At = to_torch_csr(A)
    del A
    print(f"[{args.tag}] variant={args.variant} norm={args.norm} N={N} nnz={At.values().numel()} "
          f"({time.time()-t_start:.0f}s)", flush=True)

    body_ids = np.load(f"{DATA}/annotated_body_ids.npy")
    in_idx_np = pick_input_neurons(body_ids, N, args.in_neurons, seed=args.seed)
    n_in = len(in_idx_np)
    in_idx = torch.from_numpy(in_idx_np.astype(np.int64))
    B_enc = (np.random.default_rng(args.seed + 1).standard_normal((V, n_in)) / np.sqrt(V)).astype(np.float32)
    B_enc *= args.in_gain
    B_enc_t = torch.from_numpy(B_enc)
    feat_np = np.sort(np.random.default_rng(args.seed + 2).choice(N, args.readout, replace=False))
    feat = torch.from_numpy(feat_np.astype(np.int64))

    B = args.streams
    st_t = torch.arange(B)

    def make_chunks(id_arr):
        L = len(id_arr) // B
        return np.stack([id_arr[k * L:(k + 1) * L] for k in range(B)])  # (B, L)

    TR = torch.from_numpy(make_chunks(train_ids))
    VA = torch.from_numpy(make_chunks(val_ids))
    Ltr, Lva = TR.shape[1], VA.shape[1]

    X = torch.zeros(N, B)
    pos = 0

    def step_batch(tok_ids):
        nonlocal pos
        tok = tok_ids if torch.is_tensor(tok_ids) else torch.as_tensor(tok_ids)
        Z = torch.sparse.mm(At, X)                       # (N, B)
        Z.index_add_(0, in_idx, B_enc_t[tok].T)          # inject currents into sensory neurons
        X.mul_(1 - args.leak).add_(torch.tanh(Z), alpha=args.leak)
        pos += 1

    def reset_streams(chunks, burn):
        nonlocal pos
        pos = 0
        with torch.no_grad():
            X.zero_()
            for _ in range(burn):
                step_batch(chunks[st_t, pos])

    # ---------------- training pass: streaming ridge ----------------
    D = args.readout
    G = torch.zeros(D, D, dtype=torch.float64)
    C = torch.zeros(D, V, dtype=torch.float64)
    n_collect = 0
    t0 = time.time()
    reset_streams(TR, args.burn_in)
    for t in range(Ltr - 1 - args.burn_in):
        tok = TR[st_t, pos]
        nxt = TR[st_t, pos + 1]
        step_batch(tok)
        if t % args.subsample == 0:
            F = X[feat].T.double()                       # (B, D)
            Y = torch.zeros(B, V, dtype=torch.float64)
            Y[st_t, nxt] = 1.0
            G += F.T @ F
            C += F.T @ Y
            n_collect += B
        if t % 200 == 0:
            el = time.time() - t0
            print(f"[{args.tag}] train {t}/{Ltr-1-args.burn_in} {el:.0f}s "
                  f"({el/max(t,1)*1000:.1f} ms/step)", flush=True)
    Gs, Cs = G.numpy(), C.numpy()

    # ---------------- val second moments (for cheap lambda selection) ----------------
    def val_moments():
        Gv = torch.zeros(D, D, dtype=torch.float64)
        Cv = torch.zeros(D, V, dtype=torch.float64)
        reset_streams(VA, args.burn_in)
        for t in range(Lva - 1 - args.burn_in):
            tok = VA[st_t, pos]
            nxt = VA[st_t, pos + 1]
            step_batch(tok)
            if t % args.subsample == 0:
                F = X[feat].T.double()
                Y = torch.zeros(B, V, dtype=torch.float64)
                Y[st_t, nxt] = 1.0
                Gv += F.T @ F
                Cv += F.T @ Y
        return Gv.numpy(), Cv.numpy()

    Gv, Cv = val_moments()

    lams = [3e-3, 3e-2, 3e-1, 3.0]
    lam_scores = {}
    Dm = float((Cv * Cv).sum())  # constant term proxy for MSE ranking
    for lam in lams:
        W = np.linalg.solve(Gs + lam * np.eye(D), Cs)
        mse = np.trace(W.T @ Gv @ W) - 2.0 * np.trace(W.T @ Cv) + Dm
        lam_scores[str(lam)] = float(mse)
        print(f"[{args.tag}] lambda={lam:g} val_mse={mse:.5f}", flush=True)
    lam_best = min(lam_scores, key=lam_scores.get)

    # ---------------- exact val NLL + accuracy for best lambda ----------------
    W = np.linalg.solve(Gs + float(lam_best) * np.eye(D), Cs)
    Wt = torch.from_numpy(W.astype(np.float32))

    def val_nll_acc():
        reset_streams(VA, args.burn_in)
        nll, cnt, correct = 0.0, 0, 0
        for t in range(Lva - 1 - args.burn_in):
            tok = VA[st_t, pos]
            nxt = VA[st_t, pos + 1]
            step_batch(tok)
            if t % args.subsample == 0:
                F = X[feat].T                             # (B, D) float32
                logits = F @ Wt
                logits -= logits.max(dim=1, keepdim=True).values
                p = torch.softmax(logits, dim=1)
                nll += float(-torch.log(p[st_t, nxt] + 1e-12).sum())
                correct += int((logits.argmax(dim=1) == nxt).sum())
                cnt += B
        return nll / cnt, correct / cnt

    nll, acc = val_nll_acc()

    # save state for sampling
    np.savez(f"{RESULTS}/flylm_state_{args.tag}_{args.variant}_{args.norm}_s{args.seed}.npz",
             W=W.astype(np.float32), feat_idx=feat_np, in_idx=in_idx_np,
             B_enc=B_enc, leak=args.leak, gain=args.gain, lam=float(lam_best))

    res = {
        "variant": args.variant, "norm": args.norm, "leak": args.leak, "gain": args.gain,
        "in_gain": args.in_gain, "seed": args.seed, "train_chars": args.train_chars,
        "val_chars": args.val_chars, "streams": B, "in_neurons": int(n_in),
        "readout": D, "subsample": args.subsample, "ridge_samples": int(n_collect),
        "val_nll_nats_per_char": nll,
        "val_bits_per_char": nll / np.log(2),
        "val_acc": acc,
        "lambda": float(lam_best),
        "lambda_grid_val_mse": lam_scores,
        "input_neuron_source": "sensory neurons (superclass/class contains 'sensory')",
        "wall_seconds": time.time() - t_start,
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = f"{RESULTS}/flylm_{args.tag}_{args.variant}_{args.norm}_s{args.seed}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=2))
    print(f"[{args.tag}] saved {out}", flush=True)


if __name__ == "__main__":
    main()
