"""FlyLM v2 — full-brain Echo-State LM on the REAL MaleCNS v1.0 connectome.

Dynamics: frozen fly wiring (true row-stochastic normalization, gain-controlled), tokens
drive 30k neurons directly; states collected to disk; readout = softmax logistic layer
trained with Adam (proper calibrated probabilities, like a transformer's unembedding).

CHECKPOINTED phase machine (train -> valcollect -> fit -> final_eval -> done).
Repeat the same command until it prints RESULT / DONE.

Usage:
  python flylm_esn2.py --variant fly --norm rowsum --leak 0.7 --gain 1.6 --in-gain 2.0 \
      --min-weight 2 --seed 0 --train-chars 300000 --tag main
"""
import os as _os
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repo root
import argparse, json, os, sys, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reservoir_lib import (load_corpus, load_adjacency, normalize_connectome,
                           random_graph_like, shuffled_weights, DATA)

RESULTS = f"{_R}/fly-connectome-lm/results"
CKPT = f"{_R}/data/malecns/ckpts"
torch.set_num_threads(2)


def to_torch_csr(A):
    return torch.sparse_csr_tensor(
        torch.from_numpy(A.indptr.astype(np.int64)),
        torch.from_numpy(A.indices.astype(np.int64)),
        torch.from_numpy(A.data.astype(np.float32)),
        size=A.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="fly", choices=["fly", "random", "shuffled"])
    ap.add_argument("--norm", default="rowsum", choices=["global", "deg", "log", "row", "rowsum"])
    ap.add_argument("--pool", default="random_active", choices=["random_active", "sensory"])
    ap.add_argument("--pool-size", type=int, default=30000)
    ap.add_argument("--leak", type=float, default=0.7)
    ap.add_argument("--gain", type=float, default=1.6)
    ap.add_argument("--in-gain", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-chars", type=int, default=300_000)
    ap.add_argument("--val-chars", type=int, default=40_000)
    ap.add_argument("--burn-in", type=int, default=200)
    ap.add_argument("--streams", type=int, default=64)
    ap.add_argument("--in-neurons", type=int, default=30000)
    ap.add_argument("--readout", type=int, default=2048)
    ap.add_argument("--subsample", type=int, default=2)
    ap.add_argument("--min-weight", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-seconds", type=int, default=460)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    t_start = time.time()

    os.makedirs(CKPT, exist_ok=True)
    tag_id = (f"{args.tag}_{args.variant}_{args.norm}_leak{args.leak}_gain{args.gain}"
              f"_ing{args.in_gain}_{args.pool}{args.pool_size}_s{args.seed}_mw{args.min_weight}_tc{args.train_chars}")
    ck = f"{CKPT}/flylm2_{tag_id}.npz"
    fF = f"{CKPT}/flylm2_{tag_id}_Ftr.f16"
    fy = f"{CKPT}/flylm2_{tag_id}_ytr.npy"
    fFv = f"{CKPT}/flylm2_{tag_id}_Fva.f16"
    fyv = f"{CKPT}/flylm2_{tag_id}_yva.npy"

    text, ids, _, _ = load_corpus()
    V = len(set(text))
    train_ids = ids[:args.train_chars]
    val_ids = ids[args.train_chars:args.train_chars + args.val_chars]
    B = args.streams
    D = args.readout
    st_t = torch.arange(B)

    def make_chunks(id_arr):
        L = len(id_arr) // B
        return np.stack([id_arr[k * L:(k + 1) * L] for k in range(B)])

    TR = torch.from_numpy(make_chunks(train_ids))
    VA = torch.from_numpy(make_chunks(val_ids))
    Ltr, Lva = TR.shape[1], VA.shape[1]

    # ---------- checkpoint / fresh init ----------
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=True)
        phase = str(z["phase"]); pos = int(z["pos"]); vpos = int(z["vpos"]) if "vpos" in z else 0
        in_idx_np = z["in_idx"]; B_enc = z["B_enc"]; feat_np = z["feat_idx"]
        Nn = int(z["N"])
        fit_state = dict(z["fit_state"].item()) if "fit_state" in z else {}
        print(f"[{args.tag}] resumed phase={phase} pos={pos}", flush=True)
    else:
        A_probe = load_adjacency()
        Nn = A_probe.shape[0]
        indeg = A_probe.getnnz(axis=0)
        del A_probe
        active = np.where(indeg >= 1)[0]
        body_ids = np.load(f"{DATA}/annotated_body_ids.npy")
        if args.pool == "sensory":
            import pandas as pd
            ann = pd.read_parquet(f"{DATA}/annotations_slim.parquet")[["bodyId", "superclass", "class"]]
            sup = dict(zip(ann.bodyId.values, ann.superclass.fillna("").astype(str).values))
            cls = dict(zip(ann.bodyId.values, ann["class"].fillna("").astype(str).values))
            pool = np.array([i for i, b in enumerate(body_ids)
                             if "sensory" in sup.get(b, "") or "sensory" in cls.get(b, "")])
        else:
            pool = active
        rng = np.random.default_rng(args.seed)
        in_idx_np = np.sort(rng.choice(pool, min(args.pool_size, len(pool)), replace=False))
        n_in = len(in_idx_np)
        B_enc = (np.random.default_rng(args.seed + 1).standard_normal((V, n_in)) * args.in_gain).astype(np.float32)
        feat_np = np.sort(np.random.default_rng(args.seed + 2).choice(active, D, replace=False))
        phase = "train"; pos = 0; vpos = 0; fit_state = {}
        np.savez(ck, phase=phase, pos=pos, vpos=vpos, in_idx=in_idx_np, B_enc=B_enc,
                 feat_idx=feat_np, N=Nn, fit_state=np.array(fit_state, dtype=object))
        print(f"[{args.tag}] fresh init (n_in={n_in})", flush=True)

    # rebuild sparse matrix each call
    A_raw = load_adjacency()
    if args.min_weight > 1:
        coo = A_raw.tocoo(); keep = coo.data >= args.min_weight
        A_raw = sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])),
                              shape=A_raw.shape).tocsr()
        del coo
    if args.variant == "fly":
        A = normalize_connectome(A_raw, args.norm, seed=args.seed)
    elif args.variant == "random":
        A = normalize_connectome(random_graph_like(A_raw, seed=args.seed), args.norm, seed=args.seed)
    else:
        A = normalize_connectome(shuffled_weights(A_raw, seed=args.seed), args.norm, seed=args.seed)
    del A_raw
    At = to_torch_csr(A)
    del A

    in_idx = torch.from_numpy(in_idx_np.astype(np.int64))
    B_enc_t = torch.from_numpy(B_enc)
    feat = torch.from_numpy(feat_np.astype(np.int64))

    X = torch.zeros(Nn, B)
    pos_state = {"p": pos}

    def step_batch(tok_ids):
        Z = torch.sparse.mm(At, X) * args.gain
        Z.index_add_(0, in_idx, B_enc_t[tok_ids].T)
        X.mul_(1 - args.leak).add_(torch.tanh(Z), alpha=args.leak)
        pos_state["p"] += 1

    def reset_streams(chunks):
        pos_state["p"] = 0
        X.zero_()

    done = False

    # ---------------- phase: train (collect train features) ----------------
    if phase == "train":
        if pos_state["p"] == 0:
            X.zero_()
        t0 = time.time()
        while pos_state["p"] < Ltr - 1:
            tok = TR[st_t, pos_state["p"]]
            nxt = TR[st_t, pos_state["p"] + 1]
            step_batch(tok)
            if pos_state["p"] >= args.burn_in and (pos_state["p"] - args.burn_in) % args.subsample == 0:
                with open(fF, "ab") if not os.path.exists(fF) else open(fF, "ab") as fh:
                    pass
                # append float16 features + int16 labels
                with open(fF, "ab") as fh:
                    fh.write(X[feat].T.to(torch.float16).numpy().tobytes())
                with open(fy, "ab") as fh:
                    fh.write(nxt.numpy().astype(np.int16).tobytes())
            if pos_state["p"] % 2000 == 0:
                print(f"[{args.tag}] train {pos_state['p']}/{Ltr-1} (+{time.time()-t0:.0f}s)", flush=True)
            if time.time() - t_start > args.max_seconds:
                break
        if pos_state["p"] >= Ltr - 1:
            phase = "valcollect"
            pos_state["p"] = 0
            np.savez(ck, phase=phase, pos=0, vpos=0, in_idx=in_idx_np, B_enc=B_enc,
                     feat_idx=feat_np, N=Nn, fit_state=np.array(fit_state, dtype=object))
        else:
            np.savez(ck, phase="train", pos=pos_state["p"], vpos=0, in_idx=in_idx_np,
                     B_enc=B_enc, feat_idx=feat_np, N=Nn,
                     fit_state=np.array(fit_state, dtype=object))
            done = True

    # ---------------- phase: valcollect ----------------
    if phase == "valcollect" and not done:
        X.zero_()
        t0 = time.time()
        while pos_state["p"] < Lva - 1:
            tok = VA[st_t, pos_state["p"]]
            nxt = VA[st_t, pos_state["p"] + 1]
            step_batch(tok)
            if pos_state["p"] >= args.burn_in and (pos_state["p"] - args.burn_in) % args.subsample == 0:
                with open(fFv, "ab") as fh:
                    fh.write(X[feat].T.to(torch.float16).numpy().tobytes())
                with open(fyv, "ab") as fh:
                    fh.write(nxt.numpy().astype(np.int16).tobytes())
            if time.time() - t_start > args.max_seconds:
                break
        if pos_state["p"] >= Lva - 1:
            phase = "fit"
            pos_state["p"] = 0
            np.savez(ck, phase=phase, pos=0, vpos=0, in_idx=in_idx_np, B_enc=B_enc,
                     feat_idx=feat_np, N=Nn, fit_state=np.array(fit_state, dtype=object))
        else:
            np.savez(ck, phase="valcollect", pos=pos_state["p"], vpos=0, in_idx=in_idx_np,
                     B_enc=B_enc, feat_idx=feat_np, N=Nn,
                     fit_state=np.array(fit_state, dtype=object))
            done = True

    # ---------------- phase: fit (softmax readout via Adam) ----------------
    if phase == "fit" and not done:
        Df = D
        W = torch.zeros(Df, V, requires_grad=True)
        b = torch.zeros(V, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=args.lr, weight_decay=1e-4)
        start_epoch = fit_state.get("epoch", 0)
        best = fit_state.get("best", (1e9, -1))
        # memmap the features
        n_tr = os.path.getsize(fF) // (Df * 2)
        Fm = np.memmap(fF, dtype=np.float16, mode="r", shape=(n_tr, Df))
        ym = np.fromfile(fy, dtype=np.int16)
        n_va = os.path.getsize(fFv) // (Df * 2)
        Fmv = np.memmap(fFv, dtype=np.float16, mode="r", shape=(n_va, Df))
        ymv = np.fromfile(fyv, dtype=np.int16)
        bs = 4096
        t0 = time.time()
        stop = False
        for epoch in range(start_epoch, args.epochs):
            idx = np.random.default_rng(args.seed + epoch).permutation(n_tr)
            tot = 0.0
            for s in range(0, n_tr, bs):
                ii = idx[s:s + bs]
                Fb = torch.from_numpy(Fm[ii].astype(np.float32))
                yb = torch.from_numpy(ym[ii].astype(np.int64))
                logits = Fb @ W + b
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * len(ii)
                if time.time() - t_start > args.max_seconds:
                    stop = True
                    break
            # val check
            with torch.no_grad():
                vll, vc = 0.0, 0
                vacc = 0
                for s in range(0, n_va, bs):
                    Fb = torch.from_numpy(Fmv[s:s + bs].astype(np.float32))
                    yb = torch.from_numpy(ymv[s:s + bs].astype(np.int64))
                    logits = Fb @ W + b
                    ll = F.cross_entropy(logits, yb, reduction="sum")
                    vll += float(ll); vc += len(ii) and len(yb)
                    vacc += int((logits.argmax(1) == yb).sum())
                vnll = vll / vc
            print(f"[{args.tag}] epoch {epoch}: train {tot/n_tr:.4f} val {vnll:.4f} "
                  f"({vnll/np.log(2):.3f} bpc) acc {vacc/vc:.3f} (+{time.time()-t0:.0f}s)", flush=True)
            fit_state["epoch"] = epoch + 1
            fit_state["last_val"] = vnll
            if vnll < best[0]:
                best = (vnll, epoch)
                fit_state["best"] = best
                torch.save({"W": W.detach(), "b": b.detach()}, f"{CKPT}/flylm2_{tag_id}_best.pt")
            np.savez(ck, phase="fit", pos=0, vpos=0, in_idx=in_idx_np, B_enc=B_enc,
                     feat_idx=feat_np, N=Nn, fit_state=np.array(fit_state, dtype=object))
            if stop or time.time() - t_start > args.max_seconds * 0.9:
                done = True
                break
        if fit_state.get("epoch", 0) >= args.epochs:
            phase = "final_eval"
            np.savez(ck, phase=phase, pos=0, vpos=0, in_idx=in_idx_np, B_enc=B_enc,
                     feat_idx=feat_np, N=Nn, fit_state=np.array(fit_state, dtype=object))
        else:
            done = True

    # ---------------- phase: final_eval ----------------
    if phase == "final_eval" and not done:
        bestpt = torch.load(f"{CKPT}/flylm2_{tag_id}_best.pt", weights_only=False)
        W = bestpt["W"].float(); b = bestpt["b"].float()
        n_va = os.path.getsize(fFv) // (D * 2)
        Fmv = np.memmap(fFv, dtype=np.float16, mode="r", shape=(n_va, D))
        ymv = np.fromfile(fyv, dtype=np.int16)
        nlls, accs = [], []
        with torch.no_grad():
            for s in range(0, n_va, 8192):
                Fb = torch.from_numpy(Fmv[s:s + 8192].astype(np.float32))
                yb = torch.from_numpy(ymv[s:s + 8192].astype(np.int64))
                logits = Fb @ W + b
                nlls.append(F.cross_entropy(logits, yb, reduction="sum"))
                accs.append((logits.argmax(1) == yb).sum())
        nll = float(torch.stack([torch.as_tensor(x) for x in nlls]).sum()) / n_va
        acc = float(sum(a.item() if torch.is_tensor(a) else a for a in accs)) / n_va
        res = {
            "variant": args.variant, "norm": args.norm, "leak": args.leak, "gain": args.gain,
            "in_gain": args.in_gain, "seed": args.seed, "train_chars": args.train_chars,
            "val_chars": args.val_chars, "streams": B, "in_neurons": int(len(in_idx_np)),
            "readout": D, "subsample": args.subsample, "min_weight": args.min_weight,
            "ridge_samples": n_va,
            "val_nll_nats_per_char": nll,
            "val_bits_per_char": nll / np.log(2),
            "val_acc": acc,
            "readout_type": "softmax (Adam) on frozen reservoir features",
            "input_neuron_source": f"{args.pool} (n={len(in_idx_np)})",
            "wall_seconds": time.time() - t_start,
        }
        os.makedirs(RESULTS, exist_ok=True)
        out = f"{RESULTS}/flylm_{args.tag}_{args.variant}_{args.norm}_leak{args.leak}_gain{args.gain}_{args.pool}{args.pool_size}_s{args.seed}.json"
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        np.savez(f"{RESULTS}/flylm2_state_{args.tag}_{args.variant}_{args.norm}_leak{args.leak}_gain{args.gain}_{args.pool}{args.pool_size}_s{args.seed}.npz",
                 W=W.numpy(), b=b.numpy(), feat_idx=feat_np, in_idx=in_idx_np, B_enc=B_enc,
                 leak=args.leak, gain=args.gain)
        print("RESULT " + json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}))
        print(f"[{args.tag}] saved {out}", flush=True)
        np.savez(ck, phase="done", pos=0, vpos=0, in_idx=in_idx_np, B_enc=B_enc,
                 feat_idx=feat_np, N=Nn, fit_state=np.array(fit_state, dtype=object))
        done = True

    if phase == "done":
        print(f"[{args.tag}] DONE", flush=True)


if __name__ == "__main__":
    main()
