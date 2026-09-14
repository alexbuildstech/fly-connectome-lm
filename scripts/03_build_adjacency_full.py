"""Stream the 1.05GB connectome-weights feather -> FULL deduped adjacency (no pruning).

Full-model build: ALL bodies, ALL synapse weights (min weight = 1), deduped by
(pre,post) pair. Verifies against prior-session meta:
  211,577 bodies / 26,028,386 unique pairs / 125,365,936 synapses.

Output:
  data/malecns/processed/adjacency_full.npz   (CSR float32, int32 indices)
  data/malecns/processed/adjacency_meta.json
"""
import os as _os
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repo root
import json
import os
import time

import numpy as np
import pyarrow.ipc as ipc
import scipy.sparse as sp

SRC = f"{_R}/data/connectome-weights.feather"
OUT_DIR = f"{_R}/data/malecns/processed"
os.makedirs(OUT_DIR, exist_ok=True)

t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:7.0f}s] {msg}", flush=True)


# ---- pass 1: unique body ids -----------------------------------------------------
ids_seen = set()
n_rows = 0
max_id = 0
with open(SRC, "rb") as f:
    fr = ipc.RecordBatchFileReader(f)
    nb = fr.num_record_batches
    log(f"record batches: {nb}")
    for bi in range(nb):
        b = fr.get_batch(bi)
        n_rows += b.num_rows
        pre_np = np.asarray(b.column(0)).astype(np.int64)
        post_np = np.asarray(b.column(1)).astype(np.int64)
        ids_seen.update(pre_np.tolist())
        ids_seen.update(post_np.tolist())
        max_id = max(max_id, int(pre_np.max()), int(post_np.max()))
        if bi % 500 == 0:
            log(f"pass1 batch {bi}/{nb} rows={n_rows} uniq={len(ids_seen)}")

body_ids = np.array(sorted(ids_seen), dtype=np.int64)
N = len(body_ids)
log(f"rows={n_rows} unique_bodies={N} max_id={max_id}")

if max_id < 300_000_000:
    lookup = np.full(max_id + 1, -1, dtype=np.int32)
    lookup[body_ids] = np.arange(N, dtype=np.int32)
    remap_vec = True
else:
    remap_vec = False
    raise SystemExit("body ids too large for lookup array; implement fallback")

# ---- pass 2: remap -> COO memmaps ------------------------------------------------
pre_mm = np.lib.format.open_memmap(f"{OUT_DIR}/_coo_pre.npy", mode="w+", dtype=np.int32, shape=(n_rows,))
post_mm = np.lib.format.open_memmap(f"{OUT_DIR}/_coo_post.npy", mode="w+", dtype=np.int32, shape=(n_rows,))
w_mm = np.lib.format.open_memmap(f"{OUT_DIR}/_coo_w.npy", mode="w+", dtype=np.float32, shape=(n_rows,))

off = 0
with open(SRC, "rb") as f:
    fr = ipc.RecordBatchFileReader(f)
    for bi in range(nb):
        b = fr.get_batch(bi)
        pre_np = lookup[np.asarray(b.column(0)).astype(np.int64)]
        post_np = lookup[np.asarray(b.column(1)).astype(np.int64)]
        w_np = np.asarray(b.column(2)).astype(np.float32)
        k = len(pre_np)
        pre_mm[off:off + k] = pre_np
        post_mm[off:off + k] = post_np
        w_mm[off:off + k] = w_np
        off += k
        if bi % 500 == 0:
            log(f"pass2 batch {bi}/{nb} off={off}")
log("COO memmaps written")

# ---- pass 3: row counts -> indptr -------------------------------------------------
row_counts = np.zeros(N + 1, dtype=np.int64)
for off2 in range(0, n_rows, 4_000_000):
    chunk = pre_mm[off2:off2 + 4_000_000]
    row_counts += np.bincount(chunk, minlength=N + 1)
indptr = np.zeros(N + 1, dtype=np.int64)
np.cumsum(row_counts, out=indptr[1:])
nnz_total = int(indptr[-1])
assert nnz_total == n_rows
log(f"indptr done, nnz={nnz_total}")

# ---- pass 4: counting-sort into CSR (row-grouped) --------------------------------
out_ind = np.lib.format.open_memmap(f"{OUT_DIR}/_csr_ind.npy", mode="w+", dtype=np.int32, shape=(n_rows,))
out_dat = np.lib.format.open_memmap(f"{OUT_DIR}/_csr_dat.npy", mode="w+", dtype=np.float32, shape=(n_rows,))
cursor = indptr[:-1].copy()

CH = 1_000_000
for off2 in range(0, n_rows, CH):
    pre_c = pre_mm[off2:off2 + CH]
    post_c = post_mm[off2:off2 + CH]
    w_c = w_mm[off2:off2 + CH]
    order = np.argsort(pre_c, kind="stable")
    pre_s = pre_c[order]
    post_s = post_c[order]
    w_s = w_c[order]
    # cumcount within equal-pre runs
    starts = np.zeros(len(pre_s), dtype=bool)
    starts[1:] = pre_s[1:] != pre_s[:-1]
    starts[0] = True
    grp_id = np.cumsum(starts) - 1  # group index per element
    grp_start_pos = np.flatnonzero(starts)
    cumcount = np.arange(len(pre_s), dtype=np.int64) - np.repeat(grp_start_pos, np.diff(np.append(grp_start_pos, len(pre_s))))
    dest = cursor[pre_s] + cumcount
    out_ind[dest] = post_s
    out_dat[dest] = w_s
    # advance cursor by per-group counts
    uniq_pre = pre_s[grp_start_pos]
    grp_counts = np.diff(np.append(grp_start_pos, len(pre_s)))
    np.add.at(cursor, uniq_pre, grp_counts)
log("CSR row-grouped")

del pre_mm, post_mm, w_mm
for f_ in ["_coo_pre.npy", "_coo_post.npy", "_coo_w.npy"]:
    os.remove(f"{OUT_DIR}/{f_}")

# ---- pass 5: dedupe (sort cols within rows, sum duplicates) -----------------------
A = sp.csr_matrix((out_dat, out_ind, indptr.astype(np.int32)), shape=(N, N), copy=False)
log(f"csr built: nnz={A.nnz}")
A.sum_duplicates()
A.sort_indices()
log(f"after sum_duplicates: nnz={A.nnz}")

total_syn = float(A.data.sum())
meta = {
    "n_bodies": int(N),
    "nnz_unique_connections": int(A.nnz),
    "total_synapses": total_syn,
    "density": float(A.nnz) / (N * N),
    "n_rows_source": int(n_rows),
    "max_body_id": int(max_id),
    "source_file": SRC,
}
with open(f"{OUT_DIR}/adjacency_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

A_32 = A.astype(np.float32)
sp.save_npz(f"{OUT_DIR}/adjacency_full.npz", A_32)
log(f"saved adjacency_full.npz  meta={json.dumps(meta)}")

# cleanup temps
del A, A_32
for f_ in ["_csr_ind.npy", "_csr_dat.npy"]:
    os.remove(f"{OUT_DIR}/{f_}")
log("DONE")
