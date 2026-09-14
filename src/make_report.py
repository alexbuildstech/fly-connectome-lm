"""Aggregate all result JSONs into the results summary table + generate plots.

Outputs:
  results/summary.json      — merged key metrics for every run
  results/fig_learning.png  — training curves (transformer, plastic runs)
  results/fig_main.png      — bar chart: bits/char by model
"""
import os as _os
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repo root
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS = f"{_R}/fly-connectome-lm/results"

rows = []
for fp in sorted(glob.glob(f"{RESULTS}/*.json")):
    name = os.path.basename(fp)
    if name in ("summary.json",) or name.endswith("_state.npz"):
        continue
    try:
        d = json.load(open(fp))
    except Exception:
        continue
    key = None
    if "val_bits_per_char" in d:
        rows.append({
            "file": name,
            "model": d.get("model", d.get("variant", "?")),
            "tag": d.get("tag", ""),
            "bits_per_char": d["val_bits_per_char"],
            "nats": d.get("val_nll_nats_per_char"),
            "acc": d.get("val_acc"),
            "params": d.get("params") or d.get("trainable_params"),
            "seed": d.get("seed"),
            "train_chars": d.get("train_chars"),
        })
with open(f"{RESULTS}/summary.json", "w") as f:
    json.dump(rows, f, indent=2)
print(json.dumps(rows, indent=2))

# ---------------- plots ----------------
main_rows = [r for r in rows if r["tag"] in ("", "main") and r["train_chars"] and r["train_chars"] >= 150000]
if main_rows:
    labels = [f"{r['model']}\n{r['file'].split('.')[0][:28]}" for r in main_rows]
    vals = [r["bits_per_char"] for r in main_rows]
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    colors = ["#2e7d32" if "fly" in r["model"] and "rand" not in r["model"] and "shuf" not in r["model"]
              else "#757575" for r in main_rows]
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
    ax.axhline(3.850, color="crimson", ls="--", lw=1, label="bigram (3.85 bpc)")
    ax.axhline(4.773, color="orange", ls="--", lw=1, label="unigram (4.77 bpc)")
    ax.set_ylabel("validation bits/char (lower = better)")
    ax.set_title("FlyLM vs baselines — next-char prediction on tinyshakespeare")
    ax.legend(fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
    fig.savefig(f"{RESULTS}/fig_main.png", dpi=160)
    print("saved fig_main.png")

# learning curves
fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
for fp in sorted(glob.glob(f"{RESULTS}/transformer_main_*.json")) + sorted(glob.glob(f"{RESULTS}/plastic_main_*.json")):
    d = json.load(open(fp))
    if "loss_curve" in d and d["loss_curve"]:
        xs = [h["step"] for h in d["loss_curve"]]
        ys = [h["train_loss"] for h in d["loss_curve"]]
        ax.plot(xs, ys, label=os.path.basename(fp).replace(".json", ""))
ax.set_xlabel("training step")
ax.set_ylabel("train cross-entropy (nats)")
ax.set_title("Training curves (CPU-trained)")
ax.legend(fontsize=7)
fig.savefig(f"{RESULTS}/fig_learning.png", dpi=160)
print("saved fig_learning.png")
