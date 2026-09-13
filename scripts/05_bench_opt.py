"""Benchmark spmv optimizations: int32 indices + RCM reordering."""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import torch

torch.set_num_threads(2)

A = sp.load_npz("/home/z/my-project/data/malecns/processed/adjacency.npz").tocsr().astype(np.float32)
rowsum = np.asarray(A.sum(axis=1)).ravel()
A = (sp.diags(1.0 / np.maximum(rowsum, 1e-6)) @ A).tocsr()
N = A.shape[0]
print(f"N={N} nnz={A.nnz/1e6:.1f}M")


def bench(At, N, B, tag, R=5):
    X = torch.randn(N, B)
    for _ in range(2):
        Y = torch.sparse.mm(At, X)
    t0 = time.time()
    for _ in range(R):
        Y = torch.sparse.mm(At, X)
    dt = (time.time() - t0) / R
    print(f"{tag} B={B}: {dt*1000:.1f} ms/step")
    return dt


# baseline int64
At64 = torch.sparse_csr_tensor(
    torch.from_numpy(A.indptr.astype(np.int64)),
    torch.from_numpy(A.indices.astype(np.int64)),
    torch.from_numpy(A.data), size=A.shape)
bench(At64, N, 64, "int64 baseline")

# int32 indices
At32 = torch.sparse_csr_tensor(
    torch.from_numpy(A.indptr.astype(np.int32)),
    torch.from_numpy(A.indices.astype(np.int32)),
    torch.from_numpy(A.data), size=A.shape)
bench(At32, N, 64, "int32")

# RCM reorder (symmetrized view for ordering)
t0 = time.time()
Asym = A + A.T
perm = csg.reverse_cuthill_mckee(Asym, symmetric_mode=True)
print(f"RCM computed in {time.time()-t0:.0f}s, bandwidth={np.abs(np.diff(np.sort(perm[:1000]))).mean():.0f}")
P = sp.eye(N, format="csr")[perm, :]  # row perm
A2 = (P @ A @ P.T).tocsr().astype(np.float32)
At32r = torch.sparse_csr_tensor(
    torch.from_numpy(A2.indptr.astype(np.int32)),
    torch.from_numpy(A2.indices.astype(np.int32)),
    torch.from_numpy(A2.data), size=A2.shape)
bench(At32r, N, 64, "int32+RCM")

np.save("/home/z/my-project/data/malecns/processed/rcm_perm.npy", perm.astype(np.int64))
print("perm saved")
