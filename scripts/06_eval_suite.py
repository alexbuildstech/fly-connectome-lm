"""Evaluation suite: bigram-full, memory-depth probes, induction task, generation.

Parts (run separately to bound call time):
  python3 06_eval_suite.py bigram
  python3 06_eval_suite.py probes
  python3 06_eval_suite.py induction
  python3 06_eval_suite.py generate
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/z/my-project/src")
DATA = "/home/z/my-project/data/malecns/processed"
RESULTS = "/home/z/my-project/results"
CORPUS = "/home/z/my-project/data_provenance/tinyshakespeare_input.txt"
torch.set_num_threads(2)
V = 65

ids = np.load(f"{DATA}/corpus_ids.npy")
TRAIN = ids[:1_051_394]
VAL = ids[1_051_394:]


def part_bigram():
    # bigram counts on FULL train, eval on FULL val (same split as all arms)
    C = np.zeros((V, V), dtype=np.int64)
    np.add.at(C, (TRAIN[:-1], TRAIN[1:]), 1)
    row = np.maximum(C.sum(axis=1, keepdims=True), 1)
    P = (C + 0.1) / (row + 0.1 * V)  # smoothed
    nll, correct, n = 0.0, 0, 0
    for s in range(0, len(VAL) - 1, 1_000_000):
        x = VAL[s:-1] if s == 0 else VAL[s:-1]
        y = VAL[s + 1:]
        m = len(y)
        nll += -np.log(P[x, y]).sum()
        correct += int((P[x].argmax(1) == y).sum())
        n += m
    out = {"model": "bigram-full", "bits_per_char": nll / n / math.log(2),
           "acc": correct / n, "n_val": n}
    json.dump(out, open(f"{RESULTS}/bigram_full.json", "w"), indent=2)
    print("RESULT", json.dumps(out))


def part_probes():
    z = np.load(f"{RESULTS}/probes_full2_fly.npz", allow_pickle=True)
    F = z["F"].astype(np.float32)   # (n, 4096)
    Yl = z["Yl"]                    # (n, B, n_lags)
    lags = z["lags"]
    n, Bdim, L = Yl.shape
    F2 = F.reshape(n * Bdim, -1)
    res = {}
    import sklearn.linear_model as lm
    for li, lag in enumerate(lags):
        Y = Yl[:, :, li].reshape(n * Bdim)
        ok = Y >= 0
        X, y = F2[ok], Y[ok]
        ntr = int(0.7 * len(y))
        clf = lm.LogisticRegression(max_iter=60, C=0.5, n_jobs=2)
        clf.fit(X[:ntr], y[:ntr])
        acc = float(clf.score(X[ntr:], y[ntr:]))
        prior = float(np.bincount(y[:ntr], minlength=V).max() / ntr)
        res[int(lag)] = {"acc": acc, "prior": prior}
        print(f"lag {lag}: acc {acc:.3f} (prior {prior:.3f})", flush=True)
    json.dump(res, open(f"{RESULTS}/memory_probes_fly.json", "w"), indent=2)
    print("PROBES DONE")


def _load_fly_model():
    import flylm_full2 as fl
    ck = torch.load("/home/z/my-project/data/malecns/ckpts/flylm_full2_fly_leak0.7_gain1.6_ing2.0_s0.pt",
                    weights_only=False)
    At = fl.load_fly_csr_cached()
    B_enc = torch.from_numpy(ck["B_enc"])
    in_idx = ck["in_idx"]
    W = ck["W"].detach().float()
    b0 = ck["b"].detach().float()
    return At, B_enc, in_idx, W, b0


def _fly_forward_stream(at, b_enc, in_idx, W, b0, tok_seq, gain=1.6, leak=0.7):
    """Run a single stream; return per-position logits (len, V)."""
    N = at.shape[0]
    X = torch.zeros(N, 1)
    logits = []
    for t in range(len(tok_seq)):
        Zt = torch.sparse.mm(at, X) * gain
        Zt.index_add_(0, in_idx, b_enc[torch.tensor([int(tok_seq[t])])].T)
        torch.tanh(Zt, out=Zt)
        X.mul_(1 - leak).add_(Zt, alpha=leak)
        Xn = X / (X.norm(dim=0, keepdim=True) + 1e-6)
        logits.append((Xn.T @ W + b0).squeeze(0))
    return torch.stack(logits)


def part_induction():
    """Zero-shot in-context induction: sequence S (L=16) presented twice.
    Two conditions: uniform random chars, and real text fragments from val.
    Induction gain = acc(pred | 2nd occurrence) - acc(pred | 1st occurrence)."""
    rng = np.random.default_rng(123)
    L = 16
    N_REP = 40
    # ---- fly ----
    at, b_enc, in_idx, W, b0 = _load_fly_model()
    gain, leak = 1.6, 0.7
    fly_first, fly_second = [], []
    for r in range(N_REP):
        S = rng.integers(0, V, L)
        seq = np.concatenate([S, S])
        with torch.no_grad():
            lg = _fly_forward_stream(at, b_enc, in_idx, W, b0, seq, gain, leak)
        pred = lg.argmax(-1).numpy()
        tgt = np.roll(S, -1)
        fly_first.append(float((pred[:L] == tgt).mean()))
        fly_second.append(float((pred[L:] == tgt).mean()))
    # ---- transformer ----
    src = open("/home/z/my-project/src/transformer_lm.py").read().replace(
        'if __name__ == "__main__":\n    main()', "")
    tl = {"__file__": "/home/z/my-project/src/transformer_lm.py"}
    exec(src, tl)
    model = tl["TinyGPT"](V, 320, 5, 4, 1280, 128)
    z = torch.load("/home/z/my-project/data/malecns/ckpts/transformer_full_L_s0.pt", weights_only=False)
    model.load_state_dict(z["model"])
    model.eval()
    tf_first, tf_second = [], []
    with torch.no_grad():
        for r in range(N_REP):
            S = rng.integers(0, V, L)
            seq = np.concatenate([S, S])
            x = torch.from_numpy(seq[:-1]).unsqueeze(0)
            logits = model(x)[0]
            pred = logits.argmax(-1).numpy()
            tgt = np.roll(S, -1)
            tf_first.append(float((pred[:L - 1] == tgt[:L - 1]).mean()))
            tf_second.append(float((pred[L - 1:] == tgt[L - 1:]).mean()))
    out = {
        "random_tokens": {
            "fly": {"acc_first": float(np.mean(fly_first)), "acc_second": float(np.mean(fly_second)),
                    "induction_gain": float(np.mean(fly_second) - np.mean(fly_first))},
            "transformer": {"acc_first": float(np.mean(tf_first)), "acc_second": float(np.mean(tf_second)),
                            "induction_gain": float(np.mean(tf_second) - np.mean(tf_first))}},
            "L": L, "n_rep": N_REP, "protocol": "zero-shot, both models trained on Shakespeare only",
    }
    json.dump(out, open(f"{RESULTS}/induction.json", "w"), indent=2)
    print("RESULT", json.dumps(out))


def _ngram_stats(text_ids, n=3):
    from collections import Counter
    g = Counter(tuple(text_ids[i:i + n]) for i in range(len(text_ids) - n + 1))
    return g


def part_generate():
    import flylm_full2 as fl
    # ---- fly generation (temperature 1.0, greedy-free sampling) ----
    at, b_enc, in_idx, W, b0 = _load_fly_model()
    N = at.shape[0]
    X = torch.zeros(N, 1)
    prompt = list(ids[:64])
    toks = list(prompt)
    torch.manual_seed(0)
    t0 = time.time()
    for t in range(1200):
        tok = torch.tensor([toks[-1]])
        Zt = torch.sparse.mm(at, X) * 1.6
        Zt.index_add_(0, in_idx, b_enc[tok].T)
        torch.tanh(Zt, out=Zt)
        X.mul_(0.3).add_(Zt, alpha=0.7)
        Xn = X / (X.norm(dim=0, keepdim=True) + 1e-6)
        lg = (Xn.T @ W + b0).squeeze(0)
        p = torch.softmax(lg / 0.9, dim=-1)
        nxt = int(torch.multinomial(p, 1))
        toks.append(nxt)
        if t % 300 == 0:
            print(f"fly gen {t}/1200 ({time.time()-t0:.0f}s)", flush=True)
    chars = sorted(set(open(CORPUS, encoding="utf-8").read()))
    fly_text = "".join(chars[t] for t in toks)
    open(f"{RESULTS}/sample_fly_full.txt", "w").write(fly_text)
    # ---- transformer generation ----
    src = open("/home/z/my-project/src/transformer_lm.py").read().replace(
        'if __name__ == "__main__":\n    main()', "")
    tl = {"__file__": "/home/z/my-project/src/transformer_lm.py"}
    exec(src, tl)
    model = tl["TinyGPT"](V, 320, 5, 4, 1280, 128)
    z = torch.load("/home/z/my-project/data/malecns/ckpts/transformer_full_L_s0.pt", weights_only=False)
    model.load_state_dict(z["model"])
    model.eval()
    ctx_ids = list(ids[:64])
    torch.manual_seed(0)
    for _ in range(1200):
        x = torch.tensor(ctx_ids[-128:]).unsqueeze(0)
        with torch.no_grad():
            lg = model(x)[0, -1]
        p = torch.softmax(lg / 0.9, dim=-1)
        ctx_ids.append(int(torch.multinomial(p, 1)))
    tf_text = "".join(chars[t] for t in ctx_ids[64:])
    open(f"{RESULTS}/sample_transformer_full.txt", "w").write(tf_text)
    # ---- stats ----
    val_ids = VAL
    g_val = _ngram_stats(val_ids)
    fly_ids = np.array([chars.index(c) for c in fly_text[64:]])
    tf_ids = np.array([chars.index(c) for c in tf_text])
    g_fly = _ngram_stats(fly_ids)
    g_tf = _ngram_stats(tf_ids)
    def overlap(ga, gb):
        inter = sum((ga & gb).values())
        tot = sum(ga.values())
        return inter / max(tot, 1)
    out = {"trigram_overlap_fly_vs_val": overlap(g_fly, g_val),
           "trigram_overlap_tf_vs_val": overlap(g_tf, g_val),
           "distinct_trigram_fly": len(g_fly) / max(sum(g_fly.values()), 1),
           "distinct_trigram_tf": len(g_tf) / max(sum(g_tf.values()), 1),
           "n_gen": 1200}
    json.dump(out, open(f"{RESULTS}/generation_stats.json", "w"), indent=2)
    print("RESULT", json.dumps(out))


if __name__ == "__main__":
    part = sys.argv[1]
    {"bigram": part_bigram, "probes": part_probes,
     "induction": part_induction, "generate": part_generate}[part]()
