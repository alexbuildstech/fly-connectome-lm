"""Benchmark: sparse spmv (full 26M-nnz connectome) + full-state readout gemm on this box."""
import time
import numpy as np
import scipy.sparse as sp
import torch

torch.set_num_threads(2)

A = sp.load_npz("/home/z/my-project/data/malecns/processed/adjacency.npz").tocsr().astype(np.float32)
rowsum = np.asarray(A.sum(axis=1)).ravel()
A = (sp.diags(1.0 / np.maximum(rowsum, 1e-6)) @ A).tocsr()
N = A.shape[0]
nnz = A.nnz
print(f"N={N} nnz={nnz/1e6:.1f}M")

At = torch.sparse_csr_tensor(
    torch.from_numpy(A.indptr.astype(np.int64)),
    torch.from_numpy(A.indices.astype(np.int64)),
    torch.from_numpy(A.data),
    size=A.shape)

for B in [32, 64, 128]:
    X = torch.randn(N, B)
    # warmup
    for _ in range(2):
        Y = torch.sparse.mm(At, X)
    t0 = time.time()
    R = 5
    for _ in range(R):
        Y = torch.sparse.mm(At, X)
    dt = (time.time() - t0) / R
    print(f"torch sparse.mm N x {B}: {dt*1000:.1f} ms/step  -> {nnz*B*2/dt/1e9:.1f} GFLOPS-equiv")

    Xg = X.numpy()
    Ag = A
    Yg = np.zeros_like(Xg)
    Ag._mulv(Xg[:0]) if False else None
    t0 = time.time()
    for _ in range(R):
        Yg = Ag @ Xg
    dt2 = (time.time() - t0) / R
    print(f"scipy CSR @ dense N x {B}: {dt2*1000:.1f} ms/step")

# full-state readout gemm benchmark: logits = X.T @ W  (B x N @ N x 65)
D, V = N, 65
for B in [32, 64]:
    W = torch.randn(N, V)
    X = torch.randn(B, N)
    for _ in range(2):
        L = X @ W
    t0 = time.time()
    R = 10
    for _ in range(R):
        L = X @ W
    dt = (time.time() - t0) / R
    print(f"readout fwd B={B}: {dt*1000:.1f} ms  ({B*N*V*2/dt/1e9:.1f} GFLOPS)")
    # backward dW = X.T @ dL
    dL = torch.randn(B, V)
    t0 = time.time()
    for _ in range(R):
        dW = X.T @ dL
    dt = (time.time() - t0) / R
    print(f"readout bwd dW B={B}: {dt*1000:.1f} ms  ({N*V*B*2/dt/1e9:.1f} GFLOPS)")
