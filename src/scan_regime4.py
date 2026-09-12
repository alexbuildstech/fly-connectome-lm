"""Regime scan v4: TRUE row-stochastic normalization (divide by row weight-sum).
Scan gain x in-gain; metric = ridge val NLL for next-char, plus prev-char decodability."""
import numpy as np, scipy.sparse as sp, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reservoir_lib import load_corpus, load_adjacency

torch.set_num_threads(2)
text, ids, _, _ = load_corpus(); V = len(set(text))
A_raw = load_adjacency()
coo = A_raw.tocoo(); keep = coo.data >= 2
A_raw = sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=A_raw.shape).tocsr()
rowsum = np.asarray(A_raw.sum(axis=1)).ravel()
A = (sp.diags(1.0 / np.maximum(rowsum, 1e-6)) @ A_raw).tocsr()
print("max row sum after norm:", float(np.asarray(A.sum(axis=1)).max()), flush=True)
At = torch.sparse_csr_tensor(torch.from_numpy(A.indptr.astype(np.int64)),
                             torch.from_numpy(A.indices.astype(np.int64)),
                             torch.from_numpy(A.data.astype(np.float32)), size=A.shape)
N = A.shape[0]
indeg = A_raw.getnnz(axis=0)
active = np.where(indeg >= 1)[0]
rng = np.random.default_rng(0)
feat = np.sort(rng.choice(active, 2048, replace=False))
in_idx = np.sort(rng.choice(active, 30000, replace=False))
in_t = torch.from_numpy(in_idx.astype(np.int64))
B = 12
st = torch.arange(B)
leak = 0.7
train_ids = ids[:120000]
L = len(train_ids) // B
TR = torch.from_numpy(np.stack([train_ids[k * L:(k + 1) * L] for k in range(B)]))
T, burn = 3000, 300

configs = [(ig, g) for ig in [1.0, 2.0, 4.0] for g in [0.8, 1.0, 1.3, 1.6]]
gains = torch.tensor([g for _, g in configs], dtype=torch.float32).unsqueeze(0)
encs = [torch.from_numpy((np.random.default_rng(200 + k).standard_normal((V, 30000)) * ig).astype(np.float32))
        for k, (ig, g) in enumerate(configs)]

X = torch.zeros(N, B)
F_list, y_list = [], []
pos = 0
for t in range(T):
    tok = TR[st, pos]
    nxt = TR[st, pos + 1]
    Z = torch.sparse.mm(At, X) * gains
    src = torch.stack([encs[k][tok[k]] for k in range(B)], dim=1)
    Z.index_add_(0, in_t, src)
    X = (1 - leak) * X + leak * torch.tanh(Z)
    if t >= burn and (t - burn) % 2 == 0:
        F_list.append(X[feat].T.clone())
        y_list.append(nxt.clone())
    pos += 1

S = torch.stack(F_list, dim=0).double().numpy()
Ys = torch.stack(y_list, dim=0).numpy()
n = S.shape[0]; ntr = int(n * 0.7)
lam = 0.1
print(f"{'ingain':>6} {'gain':>5} {'std':>7} {'mean|x|':>8} {'valNLL':>8} {'valACC':>7}")
best = (None, 1e9)
for k, (ig, g) in enumerate(configs):
    Fk = S[:, k, :]
    yk = Ys[:, k]
    Ftr, Fva = Fk[:ntr], Fk[ntr:]
    ytr, yva = yk[:ntr], yk[ntr:]
    Ytr = np.eye(V)[ytr]; Yva = np.eye(V)[yva]
    Gm = Ftr.T @ Ftr + lam * np.eye(Fk.shape[1])
    W = np.linalg.solve(Gm, Ftr.T @ Ytr)
    logits = Fva @ W
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
    nll = float(-np.log(p[np.arange(len(yva)), yva] + 1e-12).mean())
    acc = float((logits.argmax(1) == yva).mean())
    print(f"{ig:6.1f} {g:5.1f} {Fk.std(0).mean():7.4f} {np.abs(Fk).mean():8.4f} {nll:8.4f} {acc:7.3f}", flush=True)
    if nll < best[1]:
        best = ((ig, g), nll)
print("BEST:", best)
