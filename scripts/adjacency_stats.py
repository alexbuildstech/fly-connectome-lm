"""Compute meta stats for adjacency.npz (memory-lean, separate process)."""
import numpy as np
import scipy.sparse as sp
import json

OUT = "/home/z/my-project/data/malecns/processed"
A = sp.load_npz(f"{OUT}/adjacency.npz").tocsr()
N = A.shape[0]
in_w = np.asarray(A.sum(axis=0)).ravel()
out_w = np.asarray(A.sum(axis=1)).ravel()
deg_in = A.getnnz(axis=0)
deg_out = A.getnnz(axis=1)

# largest strongly connected component size (on binarized graph)
B = A.copy()
B.data = np.ones_like(B.data)
ncomp, labels = sp.csgraph.connected_components(B, directed=True, connection="strong")
sizes = np.bincount(labels)
lcc = int(sizes.max())
lcc_frac = lcc / N

# weakly connected
ncomp_w, labels_w = sp.csgraph.connected_components(B, directed=False)
sizes_w = np.bincount(labels_w)
lwcc = int(sizes_w.max())

meta = {
    "n_bodies": int(N),
    "nnz_unique_connections": int(A.nnz),
    "total_synapses": float(A.sum()),
    "density": float(A.nnz) / (N * N),
    "mean_out_weight": float(out_w.mean()),
    "max_out_weight": float(out_w.max()),
    "mean_in_weight": float(in_w.mean()),
    "max_in_weight": float(in_w.max()),
    "n_bodies_with_edges": int(((deg_in > 0) | (deg_out > 0)).sum()),
    "n_isolated_bodies": int(((deg_in == 0) & (deg_out == 0)).sum()),
    "n_strong_components": int(ncomp),
    "largest_strong_component": lcc,
    "largest_strong_frac": float(lcc_frac),
    "n_weak_components": int(ncomp_w),
    "largest_weak_component": lwcc,
    "reciprocal_connection_pairs": int(B.multiply(B.T).nnz // 2),
}
with open(f"{OUT}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2))
