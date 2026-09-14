"""FlyLM v3 FROZEN battery — GPU kernel (Kaggle T4x2).

Port of src/flylm_v3_frozen.py to a single self-contained script:
  * runs the full one-variable-at-a-time battery on the FULL corpus with the
    FULL 211,577-neuron state readout (B=64 streams)
  * E/I signed graphs (nt-based signs), multi-timescale leaks, synaptic delays
  * stores 4096-d random projections of the state during validation and fits a
    NONLINEAR readout (MLP) vs a linear readout on the same features (critique #6)
    plus per-lag memory probes (critique #4)

Protocol is byte-compatible with the committed v2/v3 protocol so numbers are
comparable to the repo's published results.
"""
import json
import math
import os
import time

import numpy as np
import scipy.sparse as sp
import torch

# ---------------------------------------------------------------- paths / device
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
else:  # local smoke test
    PROC = os.environ.get("FLYLM_PROCESSED", "../data/malecns/processed")
    OUT = os.environ.get("FLYLM_OUT", "../results/kaggle_smoke")
os.makedirs(OUT, exist_ok=True)
SMOKE = os.environ.get("FLYLM_SMOKE", "0") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[flylm-v3-frozen] device={DEVICE} proc={PROC} out={OUT}", flush=True)

V = 65
SIGNED = ("flysigned", "shuffledsigned", "randomsigned")
torch.backends.cuda.matmul.allow_tf32 = True


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- graph loading
def load_adjacency_v3(variant, seed=0):
    if variant in SIGNED:
        Araw = sp.load_npz(f"{PROC}/adjacency_{variant}_s{seed}.npz").tocsr().astype(np.float32)
        absr = np.abs(Araw)
        rowsum = np.asarray(absr.sum(axis=1)).ravel()
        A = (sp.diags(1.0 / np.maximum(rowsum, 1e-6)) @ Araw).tocsr()
        del Araw, absr
    else:
        if variant != "fly":
            Araw = sp.load_npz(f"{PROC}/adjacency_{variant}_s{seed}.npz").tocsr().astype(np.float32)
        else:
            Araw = sp.load_npz(f"{PROC}/adjacency.npz").tocsr().astype(np.float32)
        rowsum = np.asarray(Araw.sum(axis=1)).ravel()
        A = (sp.diags(1.0 / np.maximum(rowsum, 1e-6)) @ Araw).tocsr()
        del Araw
    perm = np.load(f"{PROC}/rcm_perm.npy")
    P = sp.eye(A.shape[0], format="csr")[perm, :]
    A2 = (P @ A @ P.T).tocsr().astype(np.float32)
    return A2, perm


def to_torch_csr(A, dev):
    return torch.sparse_csr_tensor(
        torch.from_numpy(A.indptr.astype(np.int32)),
        torch.from_numpy(A.indices.astype(np.int32)),
        torch.from_numpy(A.data), size=A.shape, device=dev)


def parse_leak(s, N):
    if s.startswith("multi:"):
        vals = [float(x) for x in s[6:].split(",")]
        seg = N // len(vals)
        lv = np.zeros(N, dtype=np.float32)
        for k, v in enumerate(vals):
            lo = k * seg
            hi = N if k == len(vals) - 1 else (k + 1) * seg
            lv[lo:hi] = v
        return None, torch.from_numpy(lv).unsqueeze(1).to(DEVICE), "+".join(f"{v:g}" for v in vals)
    return float(s), None, f"{float(s):g}"


def split_delays(A, frac, seed=0):
    coo = A.tocoo()
    rng = np.random.default_rng(seed)
    m = rng.random(coo.nnz) >= frac
    N = A.shape[0]
    Af = sp.coo_matrix((coo.data[m], (coo.row[m], coo.col[m])), shape=(N, N)).tocsr()
    Ad = sp.coo_matrix((coo.data[~m], (coo.row[~m], coo.col[~m])), shape=(N, N)).tocsr()
    return Af, Ad


def sensory_indices_in_perm_space(perm):
    """Sensory body indices (original space) mapped into perm space."""
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
    orig = np.array(sorted(id2orig[int(b)] for b in bodies if int(b) in id2orig),
                    dtype=np.int64)
    inv = np.argsort(perm)  # orig position -> perm position
    return inv[orig]


# ---------------------------------------------------------------- one run
def run_one(cfg):
    tag = cfg["tag"]
    variant = cfg["variant"]
    leak_s = cfg["leak"]
    gain = cfg["gain"]
    delay_frac = cfg["delay_frac"]
    seed = cfg.get("seed", 0)
    B = cfg.get("streams", 64)
    S = 1
    t0 = time.time()

    train_chars = 4000 if SMOKE else 1_051_394
    val_chars = 800 if SMOKE else 64_000
    burn_in = 20 if SMOKE else 200

    ids = np.load(f"{PROC}/corpus_ids.npy")
    train_ids = ids[:train_chars]
    val_ids = ids[train_chars:train_chars + val_chars]

    def make_chunks(id_arr):
        L = len(id_arr) // B
        return torch.from_numpy(np.stack([id_arr[k * L:(k + 1) * L] for k in range(B)]))

    TR = make_chunks(train_ids).to(DEVICE)
    VA = make_chunks(val_ids).to(DEVICE)
    Ltr, Lva = TR.shape[1], VA.shape[1]

    A2, perm = load_adjacency_v3(variant, seed)
    Nn = A2.shape[0]
    leak_scalar, leak_vec, leak_name = parse_leak(leak_s, Nn)
    sens_idx = sensory_indices_in_perm_space(perm)
    n_sens = len(sens_idx)
    in_idx = torch.from_numpy(sens_idx.astype(np.int64)).to(DEVICE)
    B_enc = (np.random.default_rng(seed + 1).standard_normal((V, n_sens))
             * cfg.get("in_gain", 2.0)).astype(np.float32)
    B_enc_t = torch.from_numpy(B_enc).to(DEVICE)

    if delay_frac > 0:
        Af, Ad = split_delays(A2, delay_frac, seed=seed)
        At_fast, At_del = to_torch_csr(Af, DEVICE), to_torch_csr(Ad, DEVICE)
        nnz = int(Af.nnz + Ad.nnz)
        del Af, Ad
    else:
        At_fast, At_del = to_torch_csr(A2, DEVICE), None
        nnz = int(A2.nnz)
    del A2

    W = (torch.randn(Nn, S * V, generator=torch.Generator().manual_seed(100)) * 0.05
         ).to(DEVICE).requires_grad_(True)
    b0 = torch.zeros(S * V, device=DEVICE, requires_grad=True)
    X = torch.zeros(Nn, B, device=DEVICE)
    Xprev = torch.zeros(Nn, B, device=DEVICE)

    opt = torch.optim.AdamW([W, b0], lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("wd", 1e-2))
    accum = cfg.get("accum", 4)
    clip = cfg.get("clip", 5.0)
    steps_total = Ltr - 1
    wu = 120

    def lr_factor(step):
        upd = step // accum
        total_upd = steps_total // accum
        if upd < wu:
            return (upd + 1) / wu
        p = (upd - wu) / max(1, total_upd - wu)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))

    X_old = [None]

    def step_batch(tok):
        if At_del is not None:
            Zt = (torch.sparse.mm(At_fast, X) + torch.sparse.mm(At_del, Xprev)) * gain
        else:
            Zt = torch.sparse.mm(At_fast, X) * gain
        Zt.index_add_(0, in_idx, B_enc_t[tok].T)
        torch.tanh(Zt, out=Zt)
        if leak_vec is not None:
            Zt.mul_(leak_vec)
            X.mul_(1 - leak_vec)
            X.add_(Zt)
        else:
            X.mul_(1 - leak_scalar).add_(Zt, alpha=leak_scalar)
        if At_del is not None:
            Xprev.copy_(X_old[0])

    def step_batch_v(tok):
        X_old[0] = X.clone()
        step_batch(tok)

    def readout_logits():
        Xn = X / (X.norm(dim=0, keepdim=True) + 1e-6)
        return (Xn.T @ W + b0).view(B, S, V), Xn

    hist = []
    st = torch.arange(B, device=DEVICE)
    opt.zero_grad()
    log(f"[{tag}] train start: variant={variant} leak={leak_name} gain={gain} "
        f"delay={delay_frac} N={Nn} nnz={nnz} n_sens={n_sens} steps={steps_total}")
    for pos in range(steps_total):
        tok = TR[st, pos]
        nxt = TR[st, pos + 1]
        step_batch_v(tok)
        logits, _ = readout_logits()
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(B * S, V), nxt.repeat(S))
        (loss / accum).backward()
        if (pos + 1) % accum == 0 or pos == steps_total - 1:
            for g in opt.param_groups:
                g["lr"] = cfg.get("lr", 1e-3) * lr_factor(pos)
            torch.nn.utils.clip_grad_norm_([W, b0], clip)
            opt.step()
            opt.zero_grad()
        if pos % max(1, steps_total // 12) == 0:
            hist.append({"pos": pos, "loss": float(loss.detach())})
            el = time.time() - t0
            eta = el / max(pos, 1) * (steps_total - pos) if pos else 0
            log(f"[{tag}] train {pos}/{steps_total} loss={float(loss.detach()):.4f} "
                f"({el:.0f}s eta {eta/60:.1f}m)")
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    train_done = time.time()

    # ---------------- validation + projection store ----------------
    K = 32
    rngP = np.random.default_rng(7)
    rows = np.repeat(np.arange(Nn, dtype=np.int64), K)
    cols = rngP.integers(0, 4096, size=rows.size, dtype=np.int64)
    vals = (rngP.choice([-1.0, 1.0], size=rows.size) / math.sqrt(K)).astype(np.float32)
    projT = sp.coo_matrix((vals, (cols, rows)), shape=(4096, Nn), dtype=np.float32).tocsr()
    Pt = to_torch_csr(projT, DEVICE)
    del projT

    nll = torch.zeros(S, device=DEVICE)
    correct = torch.zeros(S, device=DEVICE)
    cnt = 0
    curve = []
    F_list, Y_list = [], []
    store_proj = not SMOKE
    proj_every = cfg.get("proj_every", 4)
    for vpos in range(Lva - 1):
        tok = VA[st, vpos]
        nxt = VA[st, vpos + 1]
        step_batch_v(tok)
        if vpos >= burn_in:
            with torch.no_grad():
                logits, _ = readout_logits()
                ce_none = torch.nn.functional.cross_entropy(
                    logits.reshape(B * S, V), nxt.repeat(S), reduction="none").view(B, S).sum(0)
                nll += ce_none
                correct += (logits.argmax(-1) == nxt.unsqueeze(1)).sum(0)
            cnt += B
            curve.append(float(nll[-1]) / max(cnt, 1))
            if store_proj and (vpos % proj_every == 0):
                F_list.append(torch.sparse.mm(Pt, X).T.to(torch.float16).cpu().numpy())
                Y_list.append(nxt.cpu().numpy())
        if vpos % max(1, (Lva - 1) // 4) == 0:
            log(f"[{tag}] val {vpos}/{Lva-1}")
    n_val = max(cnt, 1)
    bpc = (nll / n_val / math.log(2)).cpu().numpy()
    acc = (correct / n_val).cpu().numpy()

    probes_path = None
    if F_list:
        F = np.concatenate(F_list)
        Y = np.stack(Y_list)
        LAGS = [k for k in [0, 1, 2, 4, 8, 16, 32, 64] if k * proj_every < len(Y) - 16]
        Yl = np.stack([np.roll(Y, k * proj_every, axis=0) for k in LAGS], axis=-1)
        Yl[:16, :, 0] = -1
        probes_path = f"{OUT}/probes_{tag}_{variant}_{leak_name}_dl{delay_frac}.npz"
        np.savez_compressed(probes_path, F=F, Yl=Yl, lags=np.array(LAGS),
                            proj_every=np.array(proj_every))
        del F_list, Y_list, F, Y, Yl

    wall = time.time() - t0
    fname = (f"flylm_v3_{tag}_{variant}_leak{leak_name.replace(':', '-').replace(',', '_')}"
             f"_dl{delay_frac}_gain{gain}_s{seed}.json")
    out = {"model": f"flylm-v3frozen-{variant}", "tag": tag, "variant": variant,
           "bits_per_char": [float(x) for x in bpc], "acc": [float(x) for x in acc],
           "val_positions": int(n_val), "burn_in": burn_in,
           "N_neurons": int(Nn), "nnz": nnz, "n_sensory": int(n_sens),
           "leak": leak_name, "gain": gain, "delay_frac": delay_frac,
           "signed": variant in SIGNED, "accum": accum,
           "trainable_params": int(Nn * S * V + S * V), "frozen_params": int(nnz),
           "wall_seconds": float(wall), "train_wall_seconds": float(train_done - t0),
           "train_chars": int(len(train_ids)), "val_chars": int(len(val_ids)),
           "device": DEVICE, "loss_curve": hist,
           "probes_path": os.path.basename(probes_path) if probes_path else None}
    with open(f"{OUT}/{fname}", "w") as f:
        json.dump(out, f, indent=2)
    log(f"[{tag}] RESULT {fname} bpc={float(bpc[0]):.4f} acc={float(acc[0]):.4f} wall={wall:.0f}s")
    # free GPU memory before next run
    del At_fast, W, b0, X, Xprev, opt, TR, VA, in_idx, B_enc_t, Pt
    if At_del is not None:
        del At_del
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- nonlinear readout arm
def nonlinear_arm(probes_path, tag):
    """Fit linear vs MLP readouts on the SAME stored random projections.
    Targets = next char at several lags -> tests whether a nonlinear readout
    unlocks extra memory from the slow-leak third (critiques #4 and #6).

    F rows are val positions (projection of X over all B streams); targets are
    taken from stream 0 so the probe is a clean single-stream prediction task."""
    z = np.load(probes_path)
    F = z["F"].astype(np.float32)          # (P, 4096)
    Yl = z["Yl"]                            # (P, B, n_lags)
    lags = z["lags"]
    proj_every = int(z["proj_every"])
    P = F.shape[0]
    results = {}
    for li, lag in enumerate(lags):
        # validity: sentinel rows (-1) + rows contaminated by the roll wrap-around
        y_all = Yl[:, 0, li].astype(np.int64)  # stream-0 target at this lag
        y_all[:lag * proj_every] = -1
        valid = y_all >= 0
        n_valid = int(valid.sum())
        ntr = int(n_valid * 0.8)
        Fv = F[valid]
        yv = y_all[valid]
        Ftr = torch.from_numpy(Fv[:ntr]).to(DEVICE)
        Fva = torch.from_numpy(Fv[ntr:]).to(DEVICE)
        ytr = torch.from_numpy(yv[:ntr]).to(DEVICE)
        yva = torch.from_numpy(yv[ntr:]).to(DEVICE)
        # linear readout
        Wl = torch.zeros(4096, V, device=DEVICE, requires_grad=True)
        bl = torch.zeros(V, device=DEVICE, requires_grad=True)
        optl = torch.optim.AdamW([Wl, bl], lr=3e-3, weight_decay=1e-4)
        eps = 2 if SMOKE else 6
        for ep in range(eps):
            perm = torch.randperm(ntr, device=DEVICE)
            for i in range(0, ntr, 2048):
                idx = perm[i:i + 2048]
                loss = torch.nn.functional.cross_entropy(Ftr[idx] @ Wl + bl, ytr[idx])
                optl.zero_grad(); loss.backward(); optl.step()
        # MLP readout
        torch.manual_seed(0)
        mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(4096), torch.nn.Linear(4096, 1024), torch.nn.GELU(),
            torch.nn.Linear(1024, V)).to(DEVICE)
        optm = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(eps):
            perm = torch.randperm(ntr, device=DEVICE)
            for i in range(0, ntr, 2048):
                idx = perm[i:i + 2048]
                loss = torch.nn.functional.cross_entropy(mlp(Ftr[idx]), ytr[idx])
                optm.zero_grad(); loss.backward(); optm.step()
        with torch.no_grad():
            nva = max(Fva.shape[0], 1)
            lva = float(torch.nn.functional.cross_entropy(Fva @ Wl + bl, yva, reduction="sum"))
            mva = float(torch.nn.functional.cross_entropy(mlp(Fva), yva, reduction="sum"))
            lacc = float(((Fva @ Wl + bl).argmax(-1) == yva).float().mean())
            macc = float((mlp(Fva).argmax(-1) == yva).float().mean())
        results[f"lag_{lag}"] = {"lag_chars": int(lag * proj_every),
                                 "linear_bpc": lva / nva / math.log(2),
                                 "mlp_bpc": mva / nva / math.log(2),
                                 "linear_acc": lacc, "mlp_acc": macc,
                                 "n_train": ntr, "n_val": Fva.shape[0]}
        log(f"[{tag}] lag {lag*proj_every}: linear {results[f'lag_{lag}']['linear_bpc']:.4f} "
            f"mlp {results[f'lag_{lag}']['mlp_bpc']:.4f} bpc")
        del Wl, bl, optl, mlp, optm
    with open(f"{OUT}/nonlinear_readout_{tag}.json", "w") as f:
        json.dump({"tag": tag, "probes": os.path.basename(probes_path),
                   "lags": results}, f, indent=2)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------- battery
def main():
    t0 = time.time()
    ids = np.load(f"{PROC}/corpus_ids.npy")
    log(f"corpus {len(ids)} ids, range [{ids.min()}, {ids.max()}]")

    if SMOKE:
        battery = [
            {"tag": "smoke", "variant": "flysigned", "leak": "0.7", "gain": 1.6, "delay_frac": 0.0},
            {"tag": "smoke", "variant": "fly", "leak": "multi:0.3,0.7,0.99", "gain": 1.6, "delay_frac": 0.3},
        ]
        for cfg in battery:
            run_one(cfg)
            if cfg["variant"] == "flysigned":
                import glob
                pp = glob.glob(f"{OUT}/probes_smoke_flysigned_*.npz")
                if pp:
                    nonlinear_arm(pp[0], "smoke_flysigned")
        log(f"SMOKE DONE in {time.time()-t0:.0f}s")
        return

    # signed gain escalation: run flysigned first; escalate if degenerate
    first = run_one({"tag": "v3", "variant": "flysigned", "leak": "0.7", "gain": 1.6, "delay_frac": 0.0})
    best_flysigned = first
    g = 1.6
    while best_flysigned["bits_per_char"][0] > 4.5 and g < 3.5:
        g += 0.8
        log(f"flysigned degenerate (bpc={best_flysigned['bits_per_char'][0]:.3f}) -> retry gain {g}")
        best_flysigned = run_one({"tag": "v3", "variant": "flysigned", "leak": "0.7",
                                  "gain": g, "delay_frac": 0.0})
    run_one({"tag": "v3", "variant": "shuffledsigned", "leak": "0.7", "gain": 1.6, "delay_frac": 0.0})
    run_one({"tag": "v3", "variant": "randomsigned", "leak": "0.7", "gain": 1.6, "delay_frac": 0.0})
    run_one({"tag": "v3", "variant": "fly", "leak": "0.9", "gain": 1.6, "delay_frac": 0.0})
    run_one({"tag": "v3", "variant": "fly", "leak": "0.99", "gain": 1.6, "delay_frac": 0.0})
    run_one({"tag": "v3", "variant": "fly", "leak": "multi:0.3,0.7,0.99", "gain": 1.6, "delay_frac": 0.0})
    run_one({"tag": "v3", "variant": "fly", "leak": "0.7", "gain": 1.6, "delay_frac": 0.3})
    run_one({"tag": "v3", "variant": "fly", "leak": "0.99", "gain": 1.6, "delay_frac": 0.3})

    # nonlinear readout arms on stored projections
    import glob
    for pat, tg in [("probes_v3_flysigned_*.npz", "flysigned"),
                    ("probes_v3_fly_leakmulti*.npz", "fly_multi"),
                    ("probes_v3_fly_leak0.99_dl0.0*.npz", "fly_leak099")]:
        pp = sorted(glob.glob(f"{OUT}/{pat}"))
        if pp:
            nonlinear_arm(pp[-1], tg)

    summary = {"total_wall_seconds": time.time() - t0, "device": DEVICE}
    with open(f"{OUT}/frozen_battery_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"ALL FROZEN RUNS DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
