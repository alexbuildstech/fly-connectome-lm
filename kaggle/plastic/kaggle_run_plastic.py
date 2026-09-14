"""FlyLM v3 PLASTIC — GPU kernel (Kaggle T4x2).

Port of src/flylm_v3_plastic.py: BACKPROP-THROUGH-TIME ON TRAINABLE SYNAPSES OF
THE FULL CONNECTOME. All 211,577 neurons in the dynamics; synapse weights
trainable by BPTT on the frozen wiring pattern (custom PlasticSpMM: forward =
native CSR spmm, grad_x = native spmm with A^T, grad_w = chunked gather over
the trainable edge subset).

New in this kernel vs the repo version:
  * mode flysigned_fly : E/I-signed wiring (nt-based signs) with trainable
    synapse magnitudes (signs fixed, |values| scaled to spectral radius 1.2)
    — combines the two biggest upgrades from the critique (#1 x #2)
  * exact frozen-edge refreeze after every optimizer step (train-frac < 1
    modes keep their frozen edges exactly frozen despite Adam momentum)
Modes:
  fly_fly      fly wiring, synapse init = real synapse counts (scaled)
  flysigned_fly fly wiring + E/I signs, init = signed counts (scaled)
  fly_frozen   fly wiring, real counts, synapses frozen (isolates synapse training)
  fly_rand     fly wiring, random synapse init
  rand_rand    config-model random wiring (same out-degree sequence), random init
"""
import json
import math
import os
import time

import numpy as np
import scipy.sparse as sp
import torch

def _find_proc():
    """Locate the mounted dataset dir regardless of mount layout."""
    if os.path.exists("/kaggle/input/fly-connectome-v3/corpus_ids.npy"):
        return "/kaggle/input/fly-connectome-v3"
    import glob
    hits = glob.glob("/kaggle/input/**/corpus_ids.npy", recursive=True)
    if hits:
        return os.path.dirname(sorted(hits)[0])
    return None


_proc_found = _find_proc()
if _proc_found:
    PROC = _proc_found
    OUT = "/kaggle/working"
else:
    PROC = os.environ.get("FLYLM_PROCESSED", "../data/malecns/processed")
    OUT = os.environ.get("FLYLM_OUT", "../results/kaggle_smoke")
os.makedirs(OUT, exist_ok=True)
SMOKE = os.environ.get("FLYLM_SMOKE", "0") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[flylm-v3-plastic] device={DEVICE} proc={PROC} out={OUT}", flush=True)

V = 65
GAIN = 1.2
torch.backends.cuda.matmul.allow_tf32 = True


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- graph builders
def spectral_scale_values(indptr, indices, values, target=1.2, iters=40, seed=0):
    """Power iteration on the matrix with the given values; returns scale so the
    spectral radius hits `target`. For signed graphs pass |values|."""
    rng = np.random.default_rng(seed)
    N = len(indptr) - 1
    v = rng.standard_normal(N).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    A = sp.csr_matrix((values, indices, indptr), shape=(N, N))
    rho = 1.0
    for _ in range(iters):
        w = A @ v
        n = float(np.linalg.norm(w))
        if n < 1e-12:
            return 1.0
        rho = n
        v = (w / n).astype(np.float32)
    return target / max(rho, 1e-12)


def build_mode(mode, seed=0):
    if mode in ("flysigned_fly",):
        Araw = sp.load_npz(f"{PROC}/adjacency_flysigned_s{seed}.npz").tocsr().astype(np.float32)
        N = Araw.shape[0]
        indptr, indices = Araw.indptr, Araw.indices
        mag = np.abs(Araw.data)
        scale = spectral_scale_values(indptr, indices, mag, GAIN, seed=seed)
        values = np.sign(Araw.data) * mag * scale
        del Araw
        return (indptr.astype(np.int64), indices.astype(np.int64),
                values.astype(np.float32), N)
    A = sp.load_npz(f"{PROC}/adjacency.npz").tocsr().astype(np.float32)
    N = A.shape[0]
    if mode in ("fly_fly", "fly_frozen"):
        indptr, indices, values = A.indptr, A.indices, A.data.copy()
        values *= spectral_scale_values(indptr, indices, values, GAIN, seed=seed)
    elif mode == "fly_rand":
        rng = np.random.default_rng(seed + 10)
        indptr, indices = A.indptr, A.indices
        values = rng.standard_normal(A.nnz).astype(np.float32) * 0.05
        values *= spectral_scale_values(indptr, indices, values, GAIN, seed=seed)
    elif mode == "rand_rand":
        rng = np.random.default_rng(seed + 20)
        counts = A.getnnz(axis=1)
        rows = np.repeat(np.arange(N, dtype=np.int64), counts)
        cols = rng.integers(0, N, size=rows.size)
        C = sp.coo_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)),
                          shape=(N, N)).tocsr()
        C.sum_duplicates()
        indptr, indices = C.indptr, C.indices
        values = rng.standard_normal(C.nnz).astype(np.float32) * 0.05
        values *= spectral_scale_values(indptr, indices, values, GAIN, seed=seed)
        del C
    else:
        raise ValueError(mode)
    Nn = A.shape[0]
    del A
    return (indptr.astype(np.int64), indices.astype(np.int64),
            values.astype(np.float32), Nn)


def transpose_structure(indptr, indices, nnz):
    """CSR structure of A^T + permutation taking values from CSR order to CSC order."""
    N = len(indptr) - 1
    marker = sp.csr_matrix((np.arange(nnz, dtype=np.float64), indices, indptr),
                           shape=(N, N))
    At = marker.tocsc()
    permT = np.asarray(At.data, dtype=np.int64)
    return (At.indptr.astype(np.int32), At.indices.astype(np.int32), permT)


def sensory_indices_original():
    import pandas as pd
    slim = pd.read_parquet(f"{PROC}/annotations_slim.parquet")
    body_ids = np.load(f"{PROC}/annotated_body_ids.npy")
    id2orig = {int(b): i for i, b in enumerate(body_ids)}
    KEY = ("olfactory", "visual", "auditory", "mechanosensory", "gustatory",
           "thermosensory", "hygrosensory", "chemosensory", "proprioceptive",
           "nociceptive", "sensory")
    sup = slim["superclass"].fillna("").astype(str).values
    cls = slim["class"].fillna("").astype(str).values
    keep = np.zeros(len(slim), dtype=bool)
    for i in range(len(slim)):
        c = cls[i].lower()
        if "sensory" in sup[i].lower() or any(k in c for k in KEY):
            keep[i] = True
    bodies = slim["bodyId"].values[keep]
    return np.array(sorted(id2orig[int(b)] for b in bodies if int(b) in id2orig),
                    dtype=np.int64)


# ---------------------------------------------------------------- custom spmm
class PlasticSpMM(torch.autograd.Function):
    """out = W @ x with trainable W values on a frozen pattern.

    forward  : native CSR spmm
    backward : grad_x = W^T @ g (native spmm via precomputed A^T CSR structure
               with CSC-ordered values), grad_w[e] = sum_b x[col_e,b]*g[row_e,b]
               over the trainable edge subset (chunked).
    """

    @staticmethod
    def forward(ctx, x, vals, crow, col, crowT, colTt, permT, tr_pos, rowT, colTr):
        W = torch.sparse_csr_tensor(crow, col, vals, size=(crow.numel() - 1,) * 2)
        out = torch.sparse.mm(W, x)
        ctx.save_for_backward(x, vals)
        ctx.crowT, ctx.colTt, ctx.permT = crowT, colTt, permT
        ctx.tr_pos, ctx.rowT, ctx.colTr = tr_pos, rowT, colTr
        ctx.shape = (crow.numel() - 1,)
        return out

    @staticmethod
    def backward(ctx, g):
        x, vals = ctx.saved_tensors
        N = ctx.shape[0]
        WT = torch.sparse_csr_tensor(ctx.crowT, ctx.colTt, vals[ctx.permT], size=(N, N))
        grad_x = torch.sparse.mm(WT, g)
        grad_vals = torch.zeros_like(vals)
        if ctx.tr_pos is not None and ctx.tr_pos.numel():
            CH = 2_000_000
            xt, gt = x.t().contiguous(), g.t().contiguous()
            for s in range(0, ctx.tr_pos.numel(), CH):
                e = min(s + CH, ctx.tr_pos.numel())
                pos = ctx.tr_pos[s:e]
                r, c = ctx.rowT[s:e], ctx.colTr[s:e]
                grad_vals[pos] = (xt[:, c] * gt[:, r]).sum(dim=0)
        return grad_x, grad_vals, None, None, None, None, None, None, None, None


# ---------------------------------------------------------------- one mode run
def run_mode(cfg):
    mode = cfg["mode"]
    train_frac = cfg.get("train_frac", 0.2)
    steps = cfg.get("steps", 2000 if not SMOKE else 4)
    batch = cfg.get("batch", 8)
    T = cfg.get("T", 24)
    lr_w = cfg.get("lr_w", 5e-4)
    lr_io = cfg.get("lr_io", 1e-3)
    leak = cfg.get("leak", 0.7)
    seed = cfg.get("seed", 0)
    train_chars = 4000 if SMOKE else 800_000
    val_chars = 800 if SMOKE else 24_000
    t0 = time.time()
    torch.manual_seed(seed)

    tag = f"{cfg.get('tag', 'v3plastic')}_{mode}_f{train_frac}_s{seed}"
    freeze_syn = mode == "fly_frozen"
    train_syn = (not freeze_syn) and train_frac > 0

    ids = np.load(f"{PROC}/corpus_ids.npy")
    train_ids = ids[:train_chars]
    val_ids = ids[train_chars:train_chars + val_chars]
    B = batch

    indptr, indices, values, N = build_mode(mode, seed)
    crow = torch.from_numpy(indptr.astype(np.int32)).to(DEVICE)
    col = torch.from_numpy(indices.astype(np.int32)).to(DEVICE)
    Wv = torch.tensor(values, device=DEVICE)
    del values
    nnz = Wv.numel()
    log(f"[{tag}] building transpose structure (nnz={nnz})...")
    crowT_n, colT_n, permT_n = transpose_structure(indptr, indices, nnz)
    crowT = torch.from_numpy(crowT_n).to(DEVICE)
    colT = torch.from_numpy(colT_n).to(DEVICE)
    permT = torch.from_numpy(permT_n).to(DEVICE)
    del crowT_n, colT_n, permT_n, indptr, indices

    rng = np.random.default_rng(seed + 40)
    if train_syn:
        if train_frac >= 1.0:
            tr_pos_np = np.arange(nnz, dtype=np.int64)
        else:
            n_tr = int(round(nnz * train_frac))
            tr_pos_np = np.sort(rng.choice(nnz, n_tr, replace=False)).astype(np.int64)
        e_row = np.repeat(np.arange(N, dtype=np.int64), np.diff(crow.cpu().numpy()))
        rowT = torch.from_numpy(e_row[tr_pos_np]).to(DEVICE)
        colT_tr = col[torch.from_numpy(tr_pos_np).to(DEVICE)].long()
        tr_pos = torch.from_numpy(tr_pos_np).to(DEVICE)
        train_mask_np = np.zeros(nnz, dtype=bool)
        train_mask_np[tr_pos_np] = True
        del e_row
    else:
        tr_pos = rowT = colT_tr = torch.zeros(0, dtype=torch.int64, device=DEVICE)
        train_mask_np = None

    sens_idx = sensory_indices_original()
    n_sens = len(sens_idx)
    in_idx = torch.from_numpy(sens_idx).to(DEVICE)
    B_enc = (torch.randn(V, n_sens, generator=torch.Generator().manual_seed(seed + 3))
             * 0.05).to(DEVICE)
    b_in = torch.zeros(N, device=DEVICE)
    R = (torch.randn(N, V, generator=torch.Generator().manual_seed(seed + 4))
         * 0.02).to(DEVICE)

    Wv0 = Wv.detach().clone() if train_syn else None
    train_mask = torch.from_numpy(train_mask_np).to(DEVICE) if train_syn else None

    Wv.requires_grad_(train_syn)
    B_enc.requires_grad_(True); b_in.requires_grad_(True); R.requires_grad_(True)
    params = ([Wv] if train_syn else []) + [B_enc, b_in, R]
    opt = torch.optim.AdamW(
        ([{"params": [Wv], "lr": lr_w}] if train_syn else []) +
        [{"params": [B_enc, b_in, R], "lr": lr_io}], weight_decay=0.0)
    n_train = sum(p.numel() for p in params if p.requires_grad)
    log(f"[{tag}] fresh init N={N} nnz={nnz} trainable_syn="
        f"{tr_pos.numel() if train_syn else 0} params={n_train} steps={steps}")

    def run_batch(tok):
        x = torch.zeros(N, B, device=DEVICE)
        logits_all = []
        for t in range(T):
            if train_syn or freeze_syn:
                x = PlasticSpMM.apply(x, Wv, crow, col, crowT, colT, permT,
                                      tr_pos if train_syn else None, rowT, colT_tr)
            else:
                W = torch.sparse_csr_tensor(crow, col, Wv, size=(N, N))
                x = torch.sparse.mm(W, x)
            x = x * GAIN
            x = x.index_add(0, in_idx, B_enc[tok[:, t]].T)
            x = x + b_in.unsqueeze(1)
            x = (1 - leak) * x + leak * torch.tanh(x)
            logits_all.append((x / (x.norm(dim=0, keepdim=True) + 1e-6)).T @ R)
        return torch.stack(logits_all, dim=1)

    rngb = np.random.default_rng(seed)

    def get_batch():
        ix = rngb.integers(0, len(train_ids) - T - 1, B)
        x = np.stack([train_ids[i:i + T] for i in ix])
        y = np.stack([train_ids[i + 1:i + T + 1] for i in ix])
        return (torch.from_numpy(x).to(DEVICE), torch.from_numpy(y).to(DEVICE))

    hist = []
    for step in range(steps):
        xb, yb = get_batch()
        logits = run_batch(xb)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.get("clip", 1.0))
        opt.step()
        if train_syn:  # exact refreeze of non-trainable edges
            with torch.no_grad():
                Wv.data[~train_mask] = Wv0[~train_mask]
        if step % max(1, steps // 10) == 0:
            hist.append({"step": step, "loss": float(loss)})
            el = time.time() - t0
            eta = el / max(step, 1) * (steps - step) if step else 0
            log(f"[{tag}] step {step}/{steps} loss {float(loss):.4f} ({el:.0f}s eta {eta/60:.1f}m)")
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    @torch.no_grad()
    def evaluate():
        nll, cnt, correct = 0.0, 0, 0
        x = torch.zeros(N, 1, device=DEVICE)
        W = torch.sparse_csr_tensor(crow, col, Wv, size=(N, N))
        pos = 0
        while pos < len(val_ids) - 1:
            t_end = min(T, len(val_ids) - 1 - pos)
            for t in range(t_end):
                x = torch.sparse.mm(W, x) * GAIN
                x = x.index_add(0, in_idx, B_enc[torch.tensor([val_ids[pos + t]],
                                                              device=DEVICE)].T)
                x = x + b_in.unsqueeze(1)
                x = (1 - leak) * x + leak * torch.tanh(x)
                xn = x / (x.norm(dim=0, keepdim=True) + 1e-6)
                logit = (xn.T @ R)[0]
                tgt = torch.tensor([val_ids[pos + t + 1]], device=DEVICE)
                nll += float(torch.nn.functional.cross_entropy(
                    logit.unsqueeze(0), tgt, reduction="sum"))
                correct += int(logit.argmax() == val_ids[pos + t + 1])
                cnt += 1
            pos += t_end
        return nll / max(cnt, 1), correct / max(cnt, 1)

    nll, acc = evaluate()
    out = {"model": f"plastic_v3_{mode}", "mode": mode, "train_frac": train_frac,
           "N": int(N), "nnz": int(nnz),
           "trainable_synapses": int(tr_pos.numel()) if train_syn else 0,
           "trainable_params": int(n_train), "steps_done": int(steps),
           "batch": B, "T": T, "lr_w": lr_w, "lr_io": lr_io, "leak": leak,
           "gain": GAIN, "train_chars": int(len(train_ids)),
           "val_chars": int(len(val_ids)), "seed": seed,
           "val_bits_per_char": nll / math.log(2), "val_acc": acc,
           "loss_curve": hist, "wall_seconds": time.time() - t0, "device": DEVICE,
           "signed": mode == "flysigned_fly"}
    with open(f"{OUT}/plastic_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    log(f"[{tag}] RESULT bpc={out['val_bits_per_char']:.4f} acc={out['val_acc']:.4f} "
        f"wall={out['wall_seconds']:.0f}s")
    del crow, col, crowT, colT, permT, Wv, Wv0, train_mask, R, b_in, B_enc
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- battery
def main():
    t0 = time.time()
    if SMOKE:
        run_mode({"mode": "flysigned_fly", "train_frac": 0.05, "steps": 3,
                  "tag": "smoke"})
        run_mode({"mode": "fly_frozen", "train_frac": 0.0, "steps": 3, "tag": "smoke"})
        log(f"SMOKE DONE in {time.time()-t0:.0f}s")
        return
    ST = 3000
    run_mode({"mode": "flysigned_fly", "train_frac": 1.0, "steps": ST})   # E/I x plasticity
    run_mode({"mode": "fly_fly", "train_frac": 1.0, "steps": ST})         # plasticity, unsigned
    run_mode({"mode": "fly_frozen", "train_frac": 0.0, "steps": ST})      # control
    run_mode({"mode": "fly_rand", "train_frac": 0.2, "steps": ST})        # control
    run_mode({"mode": "rand_rand", "train_frac": 0.2, "steps": ST})       # control
    with open(f"{OUT}/plastic_battery_summary.json", "w") as f:
        json.dump({"total_wall_seconds": time.time() - t0, "device": DEVICE}, f, indent=2)
    log(f"ALL PLASTIC RUNS DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
