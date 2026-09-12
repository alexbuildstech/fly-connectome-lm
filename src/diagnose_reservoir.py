"""Diagnose reservoir signal propagation: are feature states modulated by the input at all?

Checks (on real fly connectome, pruned >=2, global vs deg normalization):
 1. temporal variance of feature states (dead vs saturated vs modulated)
 2. linear decodability of the PREVIOUS char from current features (memory probe)
 3. decodability of the CURRENT char (input presence)
"""
import os, sys
import numpy as np
import torch
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reservoir_lib import load_corpus, load_adjacency, normalize_connectome, DATA

torch.set_num_threads(2)
text, ids, _, _ = load_corpus()
V = len(set(text))
T = 3000
stream = ids[300000:300000 + T]

for norm in ["global", "deg"]:
    A_raw = load_adjacency()
    coo = A_raw.tocoo()
    keep = coo.data >= 2
    A_raw = sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=A_raw.shape).tocsr()
    A = normalize_connectome(A_raw, norm, seed=0)
    At = torch.sparse_csr_tensor(torch.from_numpy(A.indptr.astype(np.int64)),
                                 torch.from_numpy(A.indices.astype(np.int64)),
                                 torch.from_numpy(A.data.astype(np.float32)), size=A.shape)
    N = A.shape[0]
    # input neurons: sensory
    import pandas as pd
    ann = pd.read_parquet(f"{DATA}/annotations_slim.parquet")[["bodyId", "superclass", "class"]]
    body_ids = np.load(f"{DATA}/annotated_body_ids.npy")
    sup = dict(zip(ann.bodyId.values, ann.superclass.fillna("").astype(str).values))
    cls = dict(zip(ann.bodyId.values, ann["class"].fillna("").astype(str).values))
    sens = np.array([i for i, b in enumerate(body_ids)
                     if "sensory" in sup.get(b, "") or "sensory" in cls.get(b, "")])
    rng = np.random.default_rng(0)
    in_idx = np.sort(rng.choice(sens, 1024, replace=False))
    B_enc = (rng.standard_normal((V, 1024)) / np.sqrt(V) * 1.2).astype(np.float32)
    indeg = A_raw.getnnz(axis=0)
    feat = np.sort(rng.choice(np.where(indeg >= 1)[0], 2048, replace=False))

    leak, gain = 0.7, 1.2
    X = torch.zeros(N, 1)
    in_t = torch.from_numpy(in_idx.astype(np.int64))
    Bt = torch.from_numpy(B_enc)
    states = np.zeros((T, 2048), dtype=np.float32)
    with torch.no_grad():
        for t in range(T):
            Z = torch.sparse.mm(At, X)
            Z.index_add_(0, in_t, Bt[torch.tensor([stream[t]])].T)
            X.mul_(1 - leak).add_(torch.tanh(Z), alpha=leak)
            if t >= T - 2000:
                states[t - (T - 2000)] = X[feat, 0].numpy()

    filled = states[T - 2000:T]        # exactly 2000 recorded rows
    sl = filled[200:]                  # skip transient; row j corresponds to t = 1000 + 200 + j
    t0s = 1000 + 200
    prev_chars = stream[t0s - 1: t0s - 1 + len(sl)]
    cur_chars = stream[t0s: t0s + len(sl)]
    # one-hot regression with closed form ridge
    def decode(targets):
        Y = np.zeros((len(targets), V)); Y[np.arange(len(targets)), targets] = 1.0
        F = sl.astype(np.float64)
        Gm = F.T @ F + 1e-2 * np.eye(2048)
        Cm = F.T @ Y
        W = np.linalg.solve(Gm, Cm)
        pred = F @ W
        acc = (pred.argmax(1) == targets).mean()
        return acc
    print(f"norm={norm}: state std across time (mean over features) = {sl.std(0).mean():.4f} | "
          f"frac features std<1e-3 = {(sl.std(0) < 1e-3).mean():.2f} | "
          f"frac |mean|>0.95 = {(np.abs(sl.mean(0)) > 0.95).mean():.2f}")
    print(f"  decode PREV char acc = {decode(prev_chars):.3f} (chance 1/65=0.015, bigram-margin ~0.15-0.25)")
    print(f"  decode CURR char acc = {decode(cur_chars):.3f}")
