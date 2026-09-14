"""Shared utilities: corpus, connectome variants, dynamics for the full-brain reservoir.

Design notes (2-core / 4.1GB CPU box):
- Full-brain reservoir = scipy CSR (26M nnz). One forward step = sparse @ dense(N x B):
  B parallel text streams amortize the spmv cost.
- Readout trained by STREAMING ridge accumulation (X^T X, X^T y) — states never stored.
"""
import numpy as np
import scipy.sparse as sp

from repo_paths import CORPUS, PROCESSED

DATA = PROCESSED


# ---------------------------------------------------------------- corpus
def load_corpus():
    with open(CORPUS, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    return text, ids, stoi, itos


# ---------------------------------------------------------------- connectome variants
def load_adjacency():
    return sp.load_npz(f"{DATA}/adjacency.npz").tocsr().astype(np.float32)


def spectral_radius(A, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[1]).astype(np.float32)
    v /= np.linalg.norm(v)
    rho = 1.0
    for _ in range(iters):
        w = A @ v
        n = np.linalg.norm(w)
        if n < 1e-12:
            return 0.0
        rho = n
        v = w / n
    return float(rho)


def normalize_connectome(A, norm="global", seed=0):
    """Return A scaled to a live regime. Variants: global / deg / log / row (row-stochastic)."""
    A = A.copy()
    if norm == "log":
        A.data = np.log1p(A.data).astype(np.float32)
    elif norm == "deg":
        outd = np.sqrt(np.maximum(A.getnnz(axis=1), 1)).astype(np.float32)
        ind = np.sqrt(np.maximum(A.getnnz(axis=0), 1)).astype(np.float32)
        A = (sp.diags(1.0 / outd) @ A @ sp.diags(1.0 / ind)).tocsr()
    elif norm == "row":
        outd = np.maximum(A.getnnz(axis=1), 1).astype(np.float32)
        A = (sp.diags(1.0 / outd) @ A).tocsr()
        return A  # edge-count normalized; residual gain ~ mean synapse weight
    elif norm == "rowsum":
        rowsum = np.asarray(A.sum(axis=1)).ravel()
        A = (sp.diags(1.0 / np.maximum(rowsum, 1e-6)) @ A).tocsr()
        return A  # TRUE row-stochastic: row weight-sums = 1; gain fully controlled by --gain
    rho = spectral_radius(A, seed=seed)
    if rho > 0:
        A = (A * (1.0 / rho)).tocsr()
    return A


def random_graph_like(A, seed=0):
    """Random wiring ablation: same per-row edge counts, weights resampled i.i.d. from the
    fly's synapse-count distribution, targets uniform random (duplicate targets merged by
    CSR summing — standard config-model approximation)."""
    rng = np.random.default_rng(seed)
    N = A.shape[0]
    counts = A.getnnz(axis=1)
    rows = np.repeat(np.arange(N, dtype=np.int64), counts)
    cols = rng.integers(0, N, size=rows.size, dtype=np.int64)
    w = rng.choice(A.data, size=rows.size, replace=True).astype(np.float32)
    R = sp.coo_matrix((w, (rows, cols)), shape=(N, N), dtype=np.float32).tocsr()
    R.sum_duplicates()
    return R


def shuffled_weights(A, seed=0):
    """Same wiring, synapse counts permuted across connections."""
    rng = np.random.default_rng(seed)
    B = A.copy()
    B.data = rng.permutation(A.data).astype(np.float32)
    return B
