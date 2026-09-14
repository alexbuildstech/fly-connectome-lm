"""FlyLM v3 TASK — closed-loop odor-gradient navigation on the fly connectome
(Kaggle T4x2). Addresses critique #5: "the only task is next-character
prediction, which no fly brain ever faced".

Task. 2-D arena (10x10). An odor source emits a Gaussian field
C(p) = exp(-d(p, source)^2 / (2*1.5^2)). The agent moves at constant speed and
steers with a turn rate produced by the brain. Two antennae sample the field at
p +- 0.35 perpendicular to heading -> (C_L, C_R) in [0,1].

Fly-acceptable input: (C_L, C_R) amplitude-drive two fixed random halves of the
17,937 sensory neurons (the same sensory population the LM uses).

Brain: frozen E/I-signed connectome (row-|.| normalized), leaky tanh units,
batched over 64 independent agents.

Policy readout: fixed random projection 211k -> 2048 (normalized state) ++
delayed (C_L, C_R, C_L - C_R) -> ridge regression to the expert turn
(behavior cloning + 2 DAgger rounds, data accumulated). The IDENTICAL machinery
is used for every graph, so differences come from the wiring only.

Per-step order (identical in training collection and eval):
  sense -> delay buffer -> brain.step(drive) -> features -> policy -> env step

Conditions:
  graphs    flysigned | shuffledsigned | randomsigned
  delays    0 | 5 | 10 steps of sensory delay
  baselines sensory-only reflex (ridge on delayed sensor features, no brain)
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
print(f"[flylm-v3-task] device={DEVICE} proc={PROC} out={OUT}", flush=True)

torch.backends.cuda.matmul.allow_tf32 = True


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- graph + sensory
def load_signed_graph(variant, seed=0):
    Araw = sp.load_npz(f"{PROC}/adjacency_{variant}_s{seed}.npz").tocsr().astype(np.float32)
    absr = np.abs(Araw)
    rowsum = np.asarray(absr.sum(axis=1)).ravel()
    A = (sp.diags(1.0 / np.maximum(rowsum, 1e-6)) @ Araw).tocsr()
    nnz = int(A.nnz)
    del Araw, absr
    At = torch.sparse_csr_tensor(
        torch.from_numpy(A.indptr.astype(np.int32)),
        torch.from_numpy(A.indices.astype(np.int32)),
        torch.from_numpy(A.data), size=A.shape, device=DEVICE)
    return At, A.shape[0], nnz


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


# ---------------------------------------------------------------- environment
ARENA = 10.0
SOURCE = np.array([7.0, 7.0])
SIGMA = 1.5
SPEED = 0.25
ANT = 0.35
MAX_STEPS = 60 if SMOKE else 200
GOAL_R = 0.5
TURN_SCALE = 3.0
TURN_MAX = 1.5


def odor(p):
    d = np.linalg.norm(p - SOURCE, axis=-1)
    return np.exp(-d ** 2 / (2 * SIGMA ** 2))


def reset_env(n, rng):
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = rng.uniform(2.5, 5.0, n)
    p = np.stack([SOURCE[0] + rad * np.cos(ang), SOURCE[1] + rad * np.sin(ang)], axis=1)
    p = np.clip(p, 0.2, ARENA - 0.2)
    theta = rng.uniform(0, 2 * np.pi, n)
    return p.astype(np.float32), theta.astype(np.float32)


def sensor(p, theta):
    perp = np.stack([-np.sin(theta), np.cos(theta)], axis=1)
    cl = odor(p + ANT * perp)
    cr = odor(p - ANT * perp)
    return cl.astype(np.float32), cr.astype(np.float32)


def step_env(p, theta, omega, rng):
    del rng  # deterministic env; kept for signature symmetry
    theta = theta + np.clip(omega, -TURN_MAX, TURN_MAX) * 0.5
    d = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    p = p + SPEED * d
    for ax in range(2):
        lo = p[:, ax] < 0.05
        hi = p[:, ax] > ARENA - 0.05
        p[lo, ax] = 0.05
        p[hi, ax] = ARENA - 0.05
    return p.astype(np.float32), theta.astype(np.float32)


def oracle_turn(p, theta):
    ang_to_src = np.arctan2(SOURCE[1] - p[:, 1], SOURCE[0] - p[:, 0])
    wrap = np.arctan2(np.sin(ang_to_src - theta), np.cos(ang_to_src - theta))
    return np.clip(TURN_SCALE * wrap, -TURN_MAX, TURN_MAX).astype(np.float32)


# ---------------------------------------------------------------- brain
class FlyBrain:
    def __init__(self, At, N, in_idx, leak, in_gain=4.0):
        self.At = At
        self.N = N
        self.in_idx = in_idx
        if isinstance(leak, str) and leak.startswith("multi:"):
            vals = [float(x) for x in leak[6:].split(",")]
            seg = N // len(vals)
            lv = np.zeros(N, dtype=np.float32)
            for k, v in enumerate(vals):
                lo, hi = k * seg, (N if k == len(vals) - 1 else (k + 1) * seg)
                lv[lo:hi] = v
            self.lvec = torch.from_numpy(lv).unsqueeze(1).to(DEVICE)
            self.lsc = None
        else:
            self.lvec = None
            self.lsc = float(leak)
        self.in_gain = in_gain
        self.X = None

    def reset(self, B):
        self.X = torch.zeros(self.N, B, device=DEVICE)

    @torch.no_grad()
    def step(self, drive):
        """drive: (B, n_sens) amplitude-coded sensory drive -> scattered to neurons."""
        Z = torch.sparse.mm(self.At, self.X) * 1.6
        Z.index_add_(0, self.in_idx, drive.T * self.in_gain)
        torch.tanh(Z, out=Z)
        if self.lvec is not None:
            Z.mul_(self.lvec)
            self.X.mul_(1 - self.lvec)
            self.X.add_(Z)
        else:
            self.X.mul_(1 - self.lsc).add_(Z, alpha=self.lsc)
        return self.X


# ---------------------------------------------------------------- policy readout
PROJ_DIM = 2048


def make_projection(N, seed=11):
    rng = np.random.default_rng(seed)
    K = 16
    rows = np.repeat(np.arange(N, dtype=np.int64), K)
    cols = rng.integers(0, PROJ_DIM, size=rows.size, dtype=np.int64)
    vals = (rng.choice([-1.0, 1.0], size=rows.size) / math.sqrt(K)).astype(np.float32)
    P = sp.coo_matrix((vals, (cols, rows)), shape=(PROJ_DIM, N), dtype=np.float32).tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(P.indptr.astype(np.int32)),
        torch.from_numpy(P.indices.astype(np.int32)),
        torch.from_numpy(P.data), size=P.shape, device=DEVICE)


def features(brain, Pt, cl_d, cr_d):
    """brain.X: (N,B) current state; cl_d/cr_d: (B,) numpy delayed sensor values."""
    Xn = brain.X / (brain.X.norm(dim=0, keepdim=True) + 1e-6)
    Fb = torch.sparse.mm(Pt, Xn).T
    clt = torch.from_numpy(cl_d).to(DEVICE)
    crt = torch.from_numpy(cr_d).to(DEVICE)
    sens = torch.stack([clt, crt, clt - crt], dim=1)
    return torch.cat([Fb, sens], dim=1)


def fit_ridge(F, y, lam=1.0):
    A = F.T @ F + lam * torch.eye(F.shape[1], device=DEVICE)
    return torch.linalg.solve(A, F.T @ y)


# ---------------------------------------------------------------- rollout helper
def rollout(brain, Pt, policy, rng, delay, vL, vR, mode, noise=0.0, collect_data=False):
    """mode: 'expert' | 'policy'. Returns (F_list, y_list, episode stats)."""
    B = 64
    p, th = reset_env(B, rng)
    start_dist = np.linalg.norm(p - SOURCE, axis=1)
    if brain is not None:
        brain.reset(B)
    cl, cr = sensor(p, th)
    buf = [(cl, cr)]
    Fs, ys = [], []
    done = np.zeros(B, dtype=bool)
    steps_taken = np.zeros(B, dtype=np.int64)
    pathlen = np.zeros(B, dtype=np.float64)
    for t in range(MAX_STEPS):
        cl_d, cr_d = buf[-1 - min(delay, len(buf) - 1)]
        if brain is not None:
            drive = (torch.from_numpy(cl_d).to(DEVICE).unsqueeze(1) * vL.unsqueeze(0)
                     + torch.from_numpy(cr_d).to(DEVICE).unsqueeze(1) * vR.unsqueeze(0))
            brain.step(drive)
        if brain is not None:
            feat = features(brain, Pt, cl_d, cr_d)
        else:
            clt = torch.from_numpy(cl_d).to(DEVICE)
            crt = torch.from_numpy(cr_d).to(DEVICE)
            feat = torch.stack([clt, crt, clt - crt], dim=1)
        if collect_data:
            Fs.append(feat.detach())
            ys.append(torch.from_numpy(oracle_turn(p, th)))
        if mode == "expert":
            omega = oracle_turn(p, th)
        else:
            with torch.no_grad():
                omega = (feat @ policy).cpu().numpy()
            if noise > 0:
                omega = omega + rng.normal(0, noise, size=omega.shape).astype(np.float32)
        p_prev = p.copy()
        p, th = step_env(p, th, omega, rng)
        pathlen += np.linalg.norm(p - p_prev, axis=1)
        cl, cr = sensor(p, th)
        buf.append((cl, cr))
        if len(buf) > 32:
            buf.pop(0)
        d = np.linalg.norm(p - SOURCE, axis=1)
        newly = (~done) & (d < GOAL_R)
        steps_taken[newly] = t + 1
        done |= newly
    d = np.linalg.norm(p - SOURCE, axis=1)
    done |= d < GOAL_R
    steps_taken[done & (steps_taken == 0)] = 1
    stats = {"success_rate": float(done.mean()),
             "median_steps": float(np.median(steps_taken[done])) if done.any() else None,
             "mean_path_efficiency": float(np.mean(start_dist[done] / np.maximum(pathlen[done], 1e-6))) if done.any() else None,
             "steps_list": steps_taken[done].tolist(),
             "eff_list": (start_dist[done] / np.maximum(pathlen[done], 1e-6)).tolist()}
    if collect_data:
        return Fs, ys, stats
    return stats


# ---------------------------------------------------------------- condition runner
def run_condition(variant, leak, delay, seed=0):
    t0 = time.time()
    At, N, nnz = load_signed_graph(variant, seed)
    rng = np.random.default_rng(1000 + seed)

    sens = sensory_indices_original()
    n_sens = len(sens)
    rngS = np.random.default_rng(555)
    half = rngS.permutation(n_sens)
    hL, hR = half[:n_sens // 2], half[n_sens // 2:]
    eL = (rngS.standard_normal(n_sens // 2) * 0.03).astype(np.float32)
    eR = (rngS.standard_normal(n_sens - n_sens // 2) * 0.03).astype(np.float32)
    vL = torch.zeros(n_sens, device=DEVICE)
    vR = torch.zeros(n_sens, device=DEVICE)
    vL[torch.from_numpy(hL).to(DEVICE)] = torch.from_numpy(eL).to(DEVICE)
    vR[torch.from_numpy(hR).to(DEVICE)] = torch.from_numpy(eR).to(DEVICE)
    in_idx = torch.from_numpy(sens).to(DEVICE)

    brain = FlyBrain(At, N, in_idx, leak, in_gain=4.0)
    Pt = make_projection(N, seed=11)

    # behavior cloning (expert) + 2 DAgger rounds, data accumulated
    policy = torch.zeros(PROJ_DIM + 3, device=DEVICE)
    Fs, ys = [], []
    for rnd, (mode, noise) in enumerate([("expert", 0.0), ("policy", 0.4), ("policy", 0.2)]):
        for _ in range(4 if not SMOKE else 1):
            Fb, yb, _ = rollout(brain, Pt, policy, rng, delay, vL, vR,
                                mode=mode, noise=noise, collect_data=True)
            Fs += Fb
            ys += yb
        F = torch.cat(Fs).to(DEVICE)
        y = torch.cat(ys).to(DEVICE)
        policy = fit_ridge(F, y, lam=1.0)
        del F, y

    ev_rng = np.random.default_rng(9000 + seed)
    all_steps, all_eff, succ_acc, n_ep = [], [], 0.0, 0
    for _ in range(8 if not SMOKE else 1):
        s2 = rollout(brain, Pt, policy, ev_rng, delay, vL, vR, mode="policy")
        succ_acc += s2["success_rate"] * 64
        n_ep += 64
        all_steps += s2["steps_list"]
        all_eff += s2["eff_list"]
    stats = {"success_rate": succ_acc / max(n_ep, 1),
             "median_steps": float(np.median(all_steps)) if all_steps else None,
             "mean_path_efficiency": float(np.mean(all_eff)) if all_eff else None}
    res = {"variant": variant, "leak": leak, "delay_steps": delay,
           "N": int(N), "nnz": int(nnz), "n_sensory": int(n_sens),
           "wall_seconds": time.time() - t0, "device": DEVICE, **stats}
    log(f"[task] {variant} leak={leak} delay={delay}: succ={res['success_rate']:.3f} "
        f"med={res['median_steps']} eff={res['mean_path_efficiency']} ({res['wall_seconds']:.0f}s)")
    del At, brain, Pt
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return res


def run_sensory_only(leak, delay, seed=0):
    t0 = time.time()
    rng = np.random.default_rng(1000 + seed)
    policy = torch.zeros(3, device=DEVICE)
    Fs, ys = [], []
    for rnd, (mode, noise) in enumerate([("expert", 0.0), ("policy", 0.4), ("policy", 0.2)]):
        for _ in range(4 if not SMOKE else 1):
            Fb, yb, _ = rollout(None, None, policy, rng, delay, None, None,
                                mode=mode, noise=noise, collect_data=True)
            Fs += Fb
            ys += yb
        F = torch.cat(Fs).to(DEVICE)
        y = torch.cat(ys).to(DEVICE)
        policy = fit_ridge(F, y, lam=1.0)
        del F, y
    ev_rng = np.random.default_rng(9000 + seed)
    all_steps, all_eff, succ_acc, n_ep = [], [], 0.0, 0
    for _ in range(8 if not SMOKE else 1):
        s2 = rollout(None, None, policy, ev_rng, delay, None, None, mode="policy")
        succ_acc += s2["success_rate"] * 64
        n_ep += 64
        all_steps += s2["steps_list"]
        all_eff += s2["eff_list"]
    stats = {"success_rate": succ_acc / max(n_ep, 1),
             "median_steps": float(np.median(all_steps)) if all_steps else None,
             "mean_path_efficiency": float(np.mean(all_eff)) if all_eff else None}
    res = {"variant": "sensory_only", "leak": leak, "delay_steps": delay,
           "wall_seconds": time.time() - t0, "device": DEVICE, **stats}
    log(f"[task] sensory_only leak={leak} delay={delay}: succ={res['success_rate']:.3f}")
    return res


def main():
    t0 = time.time()
    leaks = ["0.7"] if SMOKE else ["0.7", "multi:0.3,0.7,0.99"]
    delays = [0] if SMOKE else [0, 5, 10]
    variants = ["flysigned"] if SMOKE else ["flysigned", "shuffledsigned", "randomsigned"]
    results = []
    for leak in leaks:
        for delay in delays:
            for v in variants:
                results.append(run_condition(v, leak, delay))
            results.append(run_sensory_only(leak, delay))
        if SMOKE:
            break
    with open(f"{OUT}/flytask_results.json", "w") as f:
        json.dump({"results": results, "total_wall_seconds": time.time() - t0,
                   "task": "odor_gradient_navigation", "arena": ARENA,
                   "sigma": SIGMA, "speed": SPEED, "goal_r": GOAL_R,
                   "max_steps": MAX_STEPS}, f, indent=2)
    log(f"TASK BATTERY DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
