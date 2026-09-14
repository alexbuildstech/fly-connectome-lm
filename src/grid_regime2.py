"""Grid search v2: all configs as parallel columns of one reservoir batch (fast)."""
import os as _os
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repo root
import numpy as np, scipy.sparse as sp, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reservoir_lib import load_corpus, load_adjacency, DATA

torch.set_num_threads(2)
text, ids, _, _ = load_corpus(); V = len(set(text))
A_raw = load_adjacency()
coo = A_raw.tocoo(); keep = coo.data >= 2
A_raw = sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=A_raw.shape).tocsr()
outd = np.maximum(A_raw.getnnz(axis=1), 1).astype(np.float32)
A = (sp.diags(1.0 / outd) @ A_raw).tocsr()
At = torch.sparse_csr_tensor(torch.from_numpy(A.indptr.astype(np.int64)),
                             torch.from_numpy(A.indices.astype(np.int64)),
                             torch.from_numpy(A.data.astype(np.float32)), size=A.shape)
N = A.shape[0]
import pandas as pd
ann = pd.read_parquet(f"{DATA}/annotations_slim.parquet")[["bodyId", "superclass", "class"]]
body_ids = np.load(f"{DATA}/annotated_body_ids.npy")
sup = dict(zip(ann.bodyId.values, ann.superclass.fillna("").astype(str).values))
cls = dict(zip(ann.bodyId.values, ann["class"].fillna("").astype(str).values))
sens_all = np.array([i for i, b in enumerate(body_ids)
                     if "sensory" in sup.get(b, "") or "sensory" in cls.get(b, "")])
indeg = A_raw.getnnz(axis=0)
active = np.where(indeg >= 1)[0]
rng = np.random.default_rng(0)
feat = np.sort(rng.choice(active, 2048, replace=False))
T, rec = 4000, 2500
stream = np.tile(ids[300000:305000], 2)[:T]

C = [
    ("sensall_a2_g1.5_l0.7",  sens_all, 2.0, 1.5, 0.7),
    ("sensall_a4_g2_l0.7",    sens_all, 4.0, 2.0, 0.7),
    ("sensall_a6_g3_l0.5",    sens_all, 6.0, 3.0, 0.5),
    ("sensall_a10_g5_l0.3",   sens_all, 10.0, 5.0, 0.3),
    ("sens1k_a4_g2_l0.7",     rng.choice(sens_all, 1024, replace=False), 4.0, 2.0, 0.7),
    ("rand30k_a4_g2_l0.7",    rng.choice(active, 30000, replace=False), 4.0, 2.0, 0.7),
    ("sensall_a15_g8_l0.3",   sens_all, 15.0, 8.0, 0.3),
    ("sensall_a6_g2_l0.9",    sens_all, 6.0, 2.0, 0.9),
]
B = len(C)
gains = torch.tensor([c[3] for c in C], dtype=torch.float32).unsqueeze(0)
leaks = torch.tensor([c[4] for c in C], dtype=torch.float32).unsqueeze(0)
in_map = {}
for k, (name, pool, ing, g, lk) in enumerate(C):
    in_idx = np.sort(pool)
    B_enc = (np.random.default_rng(7 + k).standard_normal((V, len(in_idx)))
             * ing).astype(np.float32)
    in_map[k] = (torch.from_numpy(in_idx.astype(np.int64)), torch.from_numpy(B_enc))

X = torch.zeros(N, B)
states = torch.zeros(T - rec, 2048, B)
with torch.no_grad():
    for t in range(T):
        Z = torch.sparse.mm(At, X) * gains
        tok = torch.tensor([stream[t]])
        for k, (in_t, Bt) in in_map.items():
            Z[in_t, k] += Bt[tok].squeeze(0)
        H = torch.tanh(Z)
        X = (1 - leaks) * X + leaks * H
        if t >= rec:
            states[t - rec] = X[feat, :].clone()  # (2048 features, B configs)

print("collected", tuple(states.shape), flush=True)
np.save(f"{_R}/data/malecns/grid_states.npy", states.numpy())
sl_all = states[300:].numpy()
t0s = rec + 300
prev = stream[t0s - 1: t0s - 1 + sl_all.shape[0]]
cur = stream[t0s: t0s + sl_all.shape[0]]
p2 = stream[t0s - 2: t0s - 2 + sl_all.shape[0]]

def decode(sl, targets):
    Y = np.zeros((len(targets), V)); Y[np.arange(len(targets)), targets] = 1.0
    F = sl.astype(np.float64)
    W = np.linalg.solve(F.T @ F + 1e-1 * np.eye(F.shape[1]), F.T @ Y)
    return float((F @ W).argmax(1).__eq__(targets).mean())

for k, (name, *_ ) in enumerate(C):
    sl = sl_all[:, :, k]
    std = float(sl.std(0).mean())
    frozen = float((sl.std(0) < 1e-3).mean())
    print(f"{name:24s} std={std:.3f} frozen={frozen:.2f} "
          f"cur={decode(sl, cur):.3f} prev={decode(sl, prev):.3f} 2back={decode(sl, p2):.3f}",
          flush=True)
