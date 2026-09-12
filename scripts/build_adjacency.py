"""MaleCNS v1.0 -> NEURON-LEVEL adjacency (restricted to annotated bodies).

One sequential pass over the 1.05GB feather (fragment ids beyond annotations are
skipped — they are not part of the 166,700-neuron brain map). Pairs (pre, post)
with both endpoints annotated are mapped to indices, streamed to an int32 memmap,
then chunk-converted to CSR with duplicate summation.
"""
import numpy as np
import pyarrow.ipc as ipc
import scipy.sparse as sp
import json, os, time, hashlib

SRC = "/home/z/my-project/data/malecns/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
BASE = "/home/z/my-project/data/malecns"
OUT = f"{BASE}/processed"
DUMP = f"{BASE}/neuron_pairs_int32.npy"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

lut_ids = np.load(f"{OUT}/annotated_body_ids.npy")  # sorted int64
N = len(lut_ids)
print(f"LUT: {N} annotated bodies", flush=True)

CAP = 60_000_000  # kept pairs are unique (pre,post) rows; 26M expected + margin
if os.path.exists(DUMP):
    os.remove(DUMP)
store = np.lib.format.open_memmap(DUMP, mode="w+", dtype=np.int32, shape=(CAP, 3))
n_kept = 0
n_rows = 0
wmin = wmax = None

with ipc.open_file(SRC) as f:
    nb = f.num_record_batches
    for i in range(nb):
        b = f.get_batch(i)
        pre = b.column("body_pre").to_numpy(zero_copy_only=True)
        post = b.column("body_post").to_numpy(zero_copy_only=True)
        w = b.column("weight").to_numpy(zero_copy_only=True)
        r = np.searchsorted(lut_ids, pre)
        c = np.searchsorted(lut_ids, post)
        r_valid = (r < N) & (lut_ids[np.minimum(r, N - 1)] == pre)
        c_valid = (c < N) & (lut_ids[np.minimum(c, N - 1)] == post)
        keep = r_valid & c_valid
        k = int(keep.sum())
        if k:
            store[n_kept:n_kept + k, 0] = r[keep].astype(np.int32)
            store[n_kept:n_kept + k, 1] = c[keep].astype(np.int32)
            store[n_kept:n_kept + k, 2] = w[keep].astype(np.int32)
            n_kept += k
        n_rows += len(pre)
        wmin = int(w.min()) if wmin is None else min(wmin, int(w.min()))
        wmax = int(w.max()) if wmax is None else max(wmax, int(w.max()))
        if i % 200 == 0:
            print(f"pass {i}/{nb} rows={n_rows} kept={n_kept} t={time.time()-t0:.0f}s", flush=True)

store.flush()
print(f"PASS done: rows={n_rows} kept_neuron_synapse_rows={n_kept} w=[{wmin},{wmax}] t={time.time()-t0:.0f}s", flush=True)

A_total = None
CHUNK = 40_000_000
for s in range(0, n_kept, CHUNK):
    e = min(s + CHUNK, n_kept)
    blk = np.asarray(store[s:e])
    r = blk[:, 0]; c = blk[:, 1]; w = blk[:, 2].astype(np.float32)
    del blk
    Ac = sp.coo_matrix((w, (r, c)), shape=(N, N), dtype=np.float32).tocsr()
    Ac.sum_duplicates()
    A_total = Ac if A_total is None else (A_total + Ac)
    print(f"csr chunk [{s},{e}) nnz={A_total.nnz} t={time.time()-t0:.0f}s", flush=True)
    del Ac, r, c, w

store._mmap.close()
os.remove(DUMP)

A = A_total.tocsr().astype(np.float32)
A.sum_duplicates()
print(f"CSR final nnz={A.nnz} t={time.time()-t0:.0f}s", flush=True)
sp.save_npz(f"{OUT}/adjacency.npz", A)

in_w = np.asarray(A.sum(axis=0)).ravel()
out_w = np.asarray(A.sum(axis=1)).ravel()
deg_in = A.getnnz(axis=0)
deg_out = A.getnnz(axis=1)
meta = {
    "source_url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "restriction": "pairs restricted to annotated bodies (body-annotations file, 211,577 ids)",
    "source_head_sha256_8": hashlib.sha256(open(SRC, "rb").read(1 << 20)).hexdigest()[:8],
    "n_annotated_bodies": int(N),
    "raw_rows_total": int(n_rows),
    "raw_rows_kept": int(n_kept),
    "raw_weight_range": [wmin, wmax],
    "nnz_unique_neuron_connections": int(A.nnz),
    "total_synapses": float(A.sum()),
    "density": float(A.nnz) / (N * N),
    "mean_out_weight": float(out_w.mean()),
    "max_out_weight": float(out_w.max()),
    "mean_in_weight": float(in_w.mean()),
    "max_in_weight": float(in_w.max()),
    "n_isolated_bodies": int(((deg_in == 0) & (deg_out == 0)).sum()),
}
with open(f"{OUT}/meta.json", "w") as fj:
    json.dump(meta, fj, indent=2)
print(json.dumps(meta, indent=2), flush=True)
print(f"ALL DONE t={time.time()-t0:.0f}s", flush=True)
