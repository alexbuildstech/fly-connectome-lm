"""Bigram (order-1 Markov) baseline on tinyshakespeare — the no-brain reference floor."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reservoir_lib import load_corpus

import repo_paths
RESULTS = repo_paths.RESULTS

text, ids, stoi, itos = load_corpus()
V = len(stoi)
n_train = 300_000
val_ids = ids[n_train:n_train + 40_000]

# count bigrams on train
C = np.zeros((V, V), dtype=np.float64)
np.add.at(C, (ids[1:n_train], ids[:n_train - 1]), 1.0)  # C[next, cur]
P = C / np.maximum(C.sum(axis=0, keepdims=True), 1.0)   # P(next|cur)

# val NLL
cur = val_ids[:-1]
nxt = val_ids[1:]
nll = -np.log(P[nxt, cur] + 1e-12)
# unigram floor
uni = np.bincount(ids[:n_train], minlength=V) / n_train
nll_uni = -np.log(uni[nxt] + 1e-12)

res = {
    "model": "bigram",
    "train_chars": n_train,
    "val_chars": len(nxt),
    "val_nll_nats_per_char": float(nll.mean()),
    "val_bits_per_char": float(nll.mean() / np.log(2)),
    "val_acc": float((P.argmax(axis=0)[cur] == nxt).mean()),
    "unigram_bits_per_char": float(nll_uni.mean() / np.log(2)),
    "uniform_bits_per_char": float(np.log2(V)),
}
os.makedirs(RESULTS, exist_ok=True)
with open(f"{RESULTS}/bigram_baseline.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
