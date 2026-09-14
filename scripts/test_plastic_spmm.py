"""Gradcheck for PlasticSpMM against dense autograd ground truth (CPU, tiny)."""
import os, sys
os.environ["FLYLM_PROCESSED"] = "/home/z/my-project/repo/data/malecns/processed"
os.environ["FLYLM_OUT"] = "/tmp/smoke_plastic"
sys.path.insert(0, "/home/z/my-project/repo/kaggle")

import numpy as np
import scipy.sparse as sp
import torch

# import without running main
import importlib.util
spec = importlib.util.spec_from_file_location("kp", "/home/z/my-project/repo/kaggle/kaggle_run_plastic.py")
kp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kp)

torch.manual_seed(0)
N, B, density = 60, 3, 0.2
A = (torch.rand(N, N) < density).float() * torch.randn(N, N).abs().sign() * torch.rand(N, N)
A = A.to_sparse().to_sparse_csr()
crow, col = A.crow_indices().int(), A.col_indices().int()
vals = A.values().clone().requires_grad_(True)

# transpose structure
Am = sp.csr_matrix((np.arange(vals.numel(), dtype=np.float64), col.numpy(), crow.numpy()), shape=(N, N))
At = Am.tocsc()
permT = torch.from_numpy(np.asarray(At.data, dtype=np.int64))
crowT = torch.from_numpy(At.indptr.astype(np.int32))
colTt = torch.from_numpy(At.indices.astype(np.int32))

tr = torch.tensor([0, 3, 5, 11, 20, 45, 80, 111], dtype=torch.int64)
e_row = np.repeat(np.arange(N), np.diff(crow.numpy()))
rowT = torch.from_numpy(e_row[tr.numpy()])
colTr = col[tr].long()

x0 = torch.randn(N, B)

# ---- custom backward
v1 = vals.detach().clone().requires_grad_(True)
x1 = x0.clone().requires_grad_(True)
out = kp.PlasticSpMM.apply(x1, v1, crow, col, crowT, colTt, permT, tr, rowT, colTr)
g = torch.randn_like(out)
out.backward(g)

# ---- dense ground truth
Wd = torch.zeros(N, N)
Wd[np.repeat(np.arange(N), np.diff(crow.numpy())), col.numpy()] = vals.detach()
v2 = Wd.clone().requires_grad_(True)
x2 = x0.clone().requires_grad_(True)
out2 = v2 @ x2
out2.backward(g)

# compare
gx_err = (x1.grad - x2.grad).abs().max().item()
gv_true = torch.zeros_like(v1.grad)
gv_true[tr] = v2.grad[rowT.numpy(), colTr.numpy()]
gv_tr_err = (v1.grad[tr] - gv_true[tr]).abs().max().item()   # trainable: fp32 tolerance
frozen_mask = torch.ones_like(v1.grad, dtype=torch.bool)
frozen_mask[tr] = False
frozen_max = v1.grad[frozen_mask].abs().max().item()          # frozen: must be exactly 0

print(f"forward allclose: {torch.allclose(out, out2, atol=1e-5)}")
print(f"grad_x max err: {gx_err:.2e}")
print(f"grad_vals trainable max err: {gv_tr_err:.2e}")
print(f"grad_vals frozen max abs (must be 0): {frozen_max:.2e}")
assert torch.allclose(out, out2, atol=1e-5)
assert gx_err < 1e-5 and gv_tr_err < 1e-5 and frozen_max == 0.0
print("PLASTICSPMM GRADCHECK PASSED")
