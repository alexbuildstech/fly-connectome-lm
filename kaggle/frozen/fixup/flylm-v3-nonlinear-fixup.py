"""FIXED nonlinear readout arm for FlyLM v3 frozen-battery probes.

Bug postmortem, fix description, and validation gates are documented in
repo/kaggle/frozen/fixup/flylm-v3-nonlinear-fixup.py (identical code; that
copy is the Kaggle kernel artifact of this file).
"""
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch

V = 65
SMOKE = os.environ.get("FLYLM_SMOKE", "0") == "1"
DEVICE = "cuda" if torch.cuda.is_available() and not SMOKE else "cpu"
MLP_HIDDEN = int(os.environ.get("FLYLM_MLP_HIDDEN", "1024"))
EPOCHS = 2 if SMOKE else int(os.environ.get("FLYLM_EPOCHS", "30"))
LR_EPOCHS = 2 if SMOKE else int(os.environ.get("FLYLM_LR_EPOCHS", "60"))
BATCH = int(os.environ.get("FLYLM_BATCH", "8192"))
torch.set_num_threads(int(os.environ.get("FLYLM_THREADS", "4")))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def selftest():
    """Positive control: planted linear signal at lag 2 must be recovered
    (~0 bpc); signal-free lags 0/4 must stay at chance."""
    rng = np.random.default_rng(0)
    P, B, D, Vt = 60, 4, 32, 8
    Y = rng.integers(0, Vt, size=(P, B)).astype(np.int64)
    F = rng.standard_normal((P, B, D)).astype(np.float32) * 0.1
    F[:, :, :Vt] = 0.0
    for p in range(2, P):
        for s in range(B):
            F[p, s, Y[p - 2, s]] += 3.0
    path = "/tmp/selftest_probes.npz"
    lags = np.array([0, 2, 4])
    Yl = np.stack([np.roll(Y, k, axis=0) for k in lags], axis=-1)
    Yl[:3, :, 0] = -1
    np.savez_compressed(path, F=F.reshape(P * B, D), Yl=Yl, lags=lags,
                        proj_every=np.array(1))
    res = nonlinear_arm(path, "selftest", out_dir="/tmp", V=Vt)
    chance = math.log2(Vt)
    ok2 = res["lag_2"]["linear_bpc"] < 0.6
    ok0 = abs(res["lag_0"]["linear_bpc"] - chance) < 0.35
    ok4 = abs(res["lag_4"]["linear_bpc"] - chance) < 0.35
    ok = ok0 and ok2 and ok4
    log(f"selftest: lag0 {res['lag_0']['linear_bpc']:.3f} / lag2 "
        f"{res['lag_2']['linear_bpc']:.3f} / lag4 "
        f"{res['lag_4']['linear_bpc']:.3f}  [chance={chance:.2f}]  "
        f"mlp lag2 {res['lag_2']['mlp_bpc']:.3f}")
    log("SELFTEST PASS — alignment verified" if ok else "SELFTEST FAIL")
    return ok


def _train_early(model, params, Ftr, ytr, lr, wd, eps, patience=None):
    n = Ftr.shape[0]
    n_in = max(int(n * 0.8), 1)
    F_in, y_in = Ftr[:n_in], ytr[:n_in]
    F_se, y_se = Ftr[n_in:], ytr[n_in:]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if patience is None:
        patience = max(8, eps // 8)
    best_ce, best_state, best_ep, since = float("inf"), None, 0, 0
    loss = torch.tensor(0.0)
    for ep in range(eps):
        perm = torch.randperm(F_in.shape[0], device=DEVICE)
        for i in range(0, F_in.shape[0], BATCH):
            idx = perm[i:i + BATCH]
            loss = torch.nn.functional.cross_entropy(model(F_in[idx]), y_in[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if F_se.shape[0] >= 16:
            with torch.no_grad():
                ce = float(torch.nn.functional.cross_entropy(
                    model(F_se), y_se, reduction="sum")) / F_se.shape[0]
            if ce < best_ce - 1e-4:
                best_ce, best_ep, since = ce, ep, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                since += 1
                if since >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_ce, best_ep, float(loss.detach())


def _fit_linear(Ftr, ytr, Fva, yva, V=V):
    D = Ftr.shape[1]
    lin = torch.nn.Linear(D, V).to(DEVICE)
    with torch.no_grad():
        lin.weight.zero_(); lin.bias.zero_()
    sel = _train_early(lin, lin.parameters(), Ftr, ytr,
                       lr=float(os.environ.get("FLYLM_LR", "0.005")), wd=1e-4,
                       eps=LR_EPOCHS)
    with torch.no_grad():
        nva = max(Fva.shape[0], 1)
        lva = float(torch.nn.functional.cross_entropy(
            lin(Fva), yva, reduction="sum"))
        lacc = float((lin(Fva).argmax(-1) == yva).float().mean())
    return lva / nva / math.log(2), lacc, sel


def _fit_mlp(Ftr, ytr, Fva, yva, V=V):
    torch.manual_seed(0)
    mlp = torch.nn.Sequential(
        torch.nn.LayerNorm(Ftr.shape[1]),
        torch.nn.Linear(Ftr.shape[1], MLP_HIDDEN), torch.nn.GELU(),
        torch.nn.Linear(MLP_HIDDEN, V)).to(DEVICE)
    sel = _train_early(mlp, mlp.parameters(), Ftr, ytr, lr=1e-3, wd=1e-4,
                       eps=EPOCHS)
    with torch.no_grad():
        nva = max(Fva.shape[0], 1)
        mva = float(torch.nn.functional.cross_entropy(
            mlp(Fva), yva, reduction="sum"))
        macc = float((mlp(Fva).argmax(-1) == yva).float().mean())
    return mva / nva / math.log(2), macc, sel


def nonlinear_arm(probes_path, tag, out_dir="/kaggle/working", V=V):
    z = np.load(probes_path)
    Fr = z["F"]
    B = z["Yl"].shape[1]
    P = Fr.shape[0] // B
    assert P * B == Fr.shape[0], f"unexpected F rows {Fr.shape}"
    F = Fr.reshape(P, B, Fr.shape[1])
    Y = z["Yl"][:, :, 0].astype(np.int64)
    lags = [int(k) for k in z["lags"]]
    proj_every = int(z["proj_every"])
    results = {}
    out_path = os.path.join(out_dir, f"nonlinear_readout_{tag}.json")
    for k in lags:
        try:
            Fp = F[max(k, 0):]
            yp = Y[:P - k] if k > 0 else Y
            if yp.shape[0] != Fp.shape[0]:
                yp = Y[max(k, 0):P]
            Ff = torch.from_numpy(
                Fp.reshape(-1, Fp.shape[2]).astype(np.float32))
            yf = torch.from_numpy(yp.reshape(-1))
            m = yf >= 0
            Ff, yf = Ff[m], yf[m]
            n = Ff.shape[0]
            ntr = int(n * 0.8)
            Ftr, Fva = Ff[:ntr], Ff[ntr:]
            mu = Ftr.mean(0, keepdim=True)
            sd = Ftr.std(0, keepdim=True) + 1e-6
            Ftr = (Ftr - mu) / sd
            Fva = (Fva - mu) / sd
            Ftr, Fva = Ftr.to(DEVICE), Fva.to(DEVICE)
            ytr, yva = yf[:ntr].to(DEVICE), yf[ntr:].to(DEVICE)
            lbpc, lacc, lsel = _fit_linear(Ftr, ytr, Fva, yva, V=V)
            mbpc, macc, msel = _fit_mlp(Ftr, ytr, Fva, yva, V=V)
            results[f"lag_{k}"] = {
                "lag_rows": k, "lag_chars": k * proj_every,
                "linear_bpc": lbpc, "mlp_bpc": mbpc,
                "linear_best_inner_ce": lsel[0], "linear_best_epoch": lsel[1],
                "mlp_best_inner_ce": msel[0], "mlp_best_epoch": msel[1],
                "linear_acc": lacc, "mlp_acc": macc,
                "n_train": ntr, "n_val": n - ntr}
            log(f"[{tag}] lag {k * proj_every:>3d}ch: linear {lbpc:.4f}  "
                f"mlp {mbpc:.4f}  bpc  (n={n})")
        except Exception:
            log(f"[{tag}] lag {k} FAILED:\n{traceback.format_exc()}")
        with open(out_path, "w") as f:
            json.dump({"tag": tag, "probes": os.path.basename(probes_path),
                       "lag_note": "lag_chars = lag_rows * proj_every; "
                                   "state predicts char lag_chars in the past",
                       "lags": results}, f, indent=2)
        del Ff, Ftr, Fva
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


def find_probe_files():
    import glob
    return sorted(glob.glob("/kaggle/input/**/probes_*.npz", recursive=True))


def main():
    t0 = time.time()
    log(f"device={DEVICE} hidden={MLP_HIDDEN} epochs={EPOCHS} "
        f"lr_epochs={LR_EPOCHS} batch={BATCH}")
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        ok = selftest()
        sys.exit(0 if ok else 1)
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/kaggle/working"
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    os.makedirs(out_dir, exist_ok=True)
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        import glob
        files = sorted(glob.glob(os.path.join(sys.argv[1], "probes_*.npz")))
    else:
        files = find_probe_files()
    if SMOKE:
        files = files[:1]
    log(f"{len(files)} probe file(s) found")
    for fp in files:
        base = os.path.basename(fp)
        if only and not any(o in base for o in only):
            continue
        tg = base[len("probes_"):-len(".npz")]
        if not SMOKE and os.path.exists(
                os.path.join(out_dir, f"nonlinear_readout_{tg}.json")):
            log(f"[{tg}] already done, skip")
            continue
        log(f"[{tg}] starting {base}")
        t1 = time.time()
        nonlinear_arm(fp, tg, out_dir=out_dir)
        log(f"[{tg}] done in {time.time() - t1:.0f}s")
    import glob as g
    roll = {}
    for f in sorted(g.glob(os.path.join(out_dir, "nonlinear_readout_*.json"))):
        d = json.load(open(f))
        roll[d["tag"]] = {k: {"linear_bpc": v["linear_bpc"],
                              "mlp_bpc": v["mlp_bpc"]}
                          for k, v in d["lags"].items()}
    with open(os.path.join(out_dir, "nonlinear_rollup.json"), "w") as f:
        json.dump(roll, f, indent=2)
    log(f"ALL DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
