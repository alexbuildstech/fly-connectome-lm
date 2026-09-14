"""Induction with REAL text fragments: repeat a 16-char val excerpt, measure
2nd-occurrence prediction gain. Zero-shot for both models."""
import json
import sys

import numpy as np
import torch

import repo_paths
sys.path.insert(0, repo_paths.SRC)
DATA = repo_paths.PROCESSED
RESULTS = repo_paths.RESULTS
torch.set_num_threads(2)
V = 65

ids = np.load(f"{DATA}/corpus_ids.npy")
VAL = ids[1_051_394:]

sys.path.insert(0, repo_paths.SCRIPTS)
from importlib import util as _util
spec = _util.spec_from_file_location("ev", f"{repo_paths.SCRIPTS}/06_eval_suite.py")
# avoid running main: load source and exec only defs
src = open(f"{repo_paths.SCRIPTS}/06_eval_suite.py").read().replace(
    'if __name__ == "__main__":\n    part = sys.argv[1]\n    {"bigram": part_bigram, "probes": part_probes,\n     "induction": part_induction, "generate": part_generate}[part]()', "")
ev = {}
exec(src, ev)

rng = np.random.default_rng(777)
L = 16
N_REP = 40
frags = []
starts = rng.integers(1000, len(VAL) - L - 1, N_REP)
for st in starts:
    frags.append(VAL[st:st + L].copy())

# fly
at, b_enc, in_idx, W, b0 = ev["_load_fly_model"]()
fly_first, fly_second = [], []
for f in frags:
    seq = np.concatenate([f, f])
    with torch.no_grad():
        lg = ev["_fly_forward_stream"](at, b_enc, in_idx, W, b0, seq)
    pred = lg.argmax(-1).numpy()
    tgt = np.roll(f, -1)
    fly_first.append(float((pred[:L] == tgt).mean()))
    fly_second.append(float((pred[L:] == tgt).mean()))

# transformer
tl = {"__file__": f"{repo_paths.SRC}/transformer_lm.py"}
tsrc = open(f"{repo_paths.SRC}/transformer_lm.py").read().replace(
    'if __name__ == "__main__":\n    main()', "")
exec(tsrc, tl)
model = tl["TinyGPT"](V, 320, 5, 4, 1280, 128)
z = torch.load(f"{repo_paths.CKPT}/transformer_full_L_s0.pt",
               weights_only=False)
model.load_state_dict(z["model"])
model.eval()
tf_first, tf_second = [], []
with torch.no_grad():
    for f in frags:
        seq = np.concatenate([f, f])
        x = torch.from_numpy(seq[:-1]).unsqueeze(0)
        pred = model(x)[0].argmax(-1).numpy()
        tgt = np.roll(f, -1)
        tf_first.append(float((pred[:L - 1] == tgt[:L - 1]).mean()))
        tf_second.append(float((pred[L - 1:] == tgt[L - 1:]).mean()))

out = {"text_fragments": {
    "fly": {"acc_first": float(np.mean(fly_first)), "acc_second": float(np.mean(fly_second)),
            "induction_gain": float(np.mean(fly_second) - np.mean(fly_first))},
    "transformer": {"acc_first": float(np.mean(tf_first)), "acc_second": float(np.mean(tf_second)),
                    "induction_gain": float(np.mean(tf_second) - np.mean(tf_first))},
    "L": L, "n_rep": N_REP,
    "note": "fragments drawn from held-out val text; char statistics familiar to both models"}}
json.dump(out, open(f"{RESULTS}/induction_fragments.json", "w"), indent=2)
print("RESULT", json.dumps(out))
