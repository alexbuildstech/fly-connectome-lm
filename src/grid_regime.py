"""Grid search for a LIVE reservoir regime: strong temporal modulation + decodability."""
import numpy as np, scipy.sparse as sp, torch, sys, os, itertools
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
stream = np.tile(ids[300000:305000], 2)

def run(in_pool, ingain, gain, leak, T=5000, rec=3000):
    in_idx = np.sort(in_pool)
    B_enc = (rng.standard_normal((V, len(in_idx))) / np.sqrt(len(in_idx)) * ingain).astype(np.float32)
    X = torch.zeros(N, 1)
    in_t = torch.from_numpy(in_idx.astype(np.int64)); Bt = torch.from_numpy(B_enc)
    states = np.zeros((T - rec, 2048), dtype=np.float32)
    with torch.no_grad():
        for t in range(T):
            Z = torch.sparse.mm(At, X)
            Z.index_add_(0, in_t, Bt[torch.tensor([stream[t]])].T)
            X.mul_(1 - leak).add_(torch.tanh(Z * gain), alpha=leak)
            if t >= rec:
                states[t - rec] = X[feat, 0].numpy()
    sl = states[300:]
    t0s = rec + 300
    prev = stream[t0s - 1: t0s - 1 + len(sl)]
    cur = stream[t0s: t0s + len(sl)]
    p2 = stream[t0s - 2: t0s - 2 + len(sl)]
    def decode(targets):
        Y = np.zeros((len(targets), V)); Y[np.arange(len(targets)), targets] = 1.0
        F = sl.astype(np.float64)
        W = np.linalg.solve(F.T @ F + 1e-1 * np.eye(2048), F.T @ Y)
        return float((F @ W).argmax(1).__eq__(targets).mean())
    std = float(sl.std(0).mean())
    frozen = float((sl.std(0) < 1e-3).mean())
    print(f"pool={len(in_pool):6d} ingain={ingain} gain={gain} leak={leak}: "
          f"std={std:.3f} frozen={frozen:.2f} cur={decode(cur):.3f} prev={decode(prev):.3f} 2back={decode(p2):.3f}",
          flush=True)

sens1024 = rng.choice(sens_all, 1024, replace=False)
for pool, ing, g, lk in [
    (sens_all, 4.0, 1.5, 0.7),
    (sens_all, 4.0, 2.5, 0.7),
    (sens_all, 8.0, 2.5, 0.7),
    (sens_all, 8.0, 4.0, 0.5),
    (sens1024, 8.0, 4.0, 0.5),
    (sens_all, 16.0, 4.0, 0.5),
]:
    run(pool, ing, g, lk)
