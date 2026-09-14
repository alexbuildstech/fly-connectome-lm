"""Make final figures + compute ledger."""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import repo_paths
RESULTS = repo_paths.RESULTS

# ---------- gather ----------
rows = []
tf = json.load(open(f"{RESULTS}/transformer_full_L_s0.json"))
rows.append(("Transformer-L\n(6.25M trainable)", tf["val_bits_per_char"], tf["val_acc"], "#2a9d8f"))
rnd = json.load(open(f"{RESULTS}/flylm_full2_random_leak0.7_gain1.6_ing2.0_s0.json"))
rows.append(("Random graph\n(frozen, 26M syn.)", rnd["bits_per_char"][0], rnd["acc"][0], "#b08968"))
shf = json.load(open(f"{RESULTS}/flylm_full2_shuffled_leak0.7_gain1.6_ing2.0_s0.json"))
rows.append(("Shuffled connectome\n(frozen, 26M syn.)", shf["bits_per_char"][0], shf["acc"][0], "#c1121f"))
fly = json.load(open(f"{RESULTS}/flylm_full2_fly_leak0.7_gain1.6_ing2.0_s0.json"))
rows.append(("Real fly connectome\n(MaleCNS v1.0, frozen)", fly["bits_per_char"][0], fly["acc"][0], "#7f5539"))
bg = json.load(open(f"{RESULTS}/bigram_full.json"))
rows.append(("Bigram baseline", bg["bits_per_char"], bg["acc"], "#adb5bd"))
v1 = json.load(open(f"{RESULTS}/flylmfull_full_fly_leak0.7_gain1.6_ing2.0_s0.json"))
rows.append(("Fly (v1, mis-fit\nreadout - appendix)", v1["bits_per_char"][0], v1["acc"][0], "#e5e5e5"))

# ---------- fig 1: main bar chart ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
names = [r[0] for r in rows]
bpcs = [r[1] for r in rows]
accs = [r[2] for r in rows]
cols = [r[3] for r in rows]
y = np.arange(len(rows))[::-1]
ax1.barh(y, bpcs, color=cols, edgecolor="#333", linewidth=0.6)
for yi, v in zip(y, bpcs):
    ax1.text(v + 0.03, yi, f"{v:.3f}", va="center", fontsize=10)
ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=9)
ax1.axvline(math.log2(65), color="#888", ls=":", lw=1)
ax1.text(math.log2(65) + 0.03, len(rows) - 0.8, "uniform (6.03)", fontsize=8, color="#666")
ax1.set_xlabel("validation bits per char (lower = better)")
ax1.set_title("Next-char prediction on held-out Shakespeare\n(same corpus, same val split for all arms)", fontsize=10)
ax2.barh(y, accs, color=cols, edgecolor="#333", linewidth=0.6)
for yi, v in zip(y, accs):
    ax2.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=10)
ax2.set_yticks(y); ax2.set_yticklabels([])
ax2.set_xlabel("next-char top-1 accuracy")
ax2.set_title("Accuracy", fontsize=10)
fig.savefig(f"{RESULTS}/fig_main.png", dpi=150)
plt.close(fig)

# ---------- fig 2: training curves ----------
ck = torch.load(f"{repo_paths.CKPT}/flylm_full2_fly_leak0.7_gain1.6_ing2.0_s0.pt",
                weights_only=False)
h = ck["hist"]
p = [x["pos"] for x in h]; l = [x["loss"] for x in h]
fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
ax.plot(np.array(p) * 64 / 1000, l, lw=0.8, color="#7f5539", label="fly: online readout CE (nats)")
th = tf["loss_curve"]
ax.plot([x["step"] * 32 * 128 / 1000 for x in th], [x["train_loss"] for x in th],
        lw=0.8, color="#2a9d8f", label="transformer: train CE (nats)")
ax.set_xlabel("characters seen (thousands)")
ax.set_ylabel("cross-entropy (nats/char)")
ax.set_title("Training curves", fontsize=10)
ax.legend(fontsize=8)
fig.savefig(f"{RESULTS}/fig_curves.png", dpi=150)
plt.close(fig)

# ---------- fig 3: memory probes ----------
pr = json.load(open(f"{RESULTS}/memory_probes_fly.json"))
lags = sorted(int(k) for k in pr)
acc = [pr[str(k)]["acc"] for k in lags]
pri = [pr[str(k)]["prior"] for k in lags]
fig, ax = plt.subplots(figsize=(6.2, 3.4), constrained_layout=True)
ax.plot(lags, acc, "o-", color="#7f5539", label="linear probe acc")
ax.plot(lags, pri, "--", color="#999", label="majority-class prior")
ax.axhline(1 / 65, color="#bbb", ls=":", label="chance (1/65)")
ax.set_xlabel("lag (chars back)")
ax.set_ylabel("token decodable from brain state")
ax.set_title("Fly reservoir functional memory depth\n(probe on held-out states, 4096-d random projection)", fontsize=10)
ax.legend(fontsize=8)
fig.savefig(f"{RESULTS}/fig_probes.png", dpi=150)
plt.close(fig)

# ---------- compute ledger ----------
fly_wall = fly["wall_seconds"]
import glob, os
def sweep_wall(variant):
    ck2 = torch.load(f"{repo_paths.CKPT}/flylm_full2_{variant}_leak0.7_gain1.6_ing2.0_s0.pt",
                     weights_only=False)
    return None
ledger = {
    "transformer": {
        "trainable_params": tf["params"],
        "frozen_params": 0,
        "steps": 4000, "batch": 32, "ctx": 128,
        "tokens_seen": 4000 * 32 * 128,
        "wall_hours": round(4000 * 1.09 / 3600, 2),
        "train_flops_est": int(6 * tf["params"] * 4000 * 32 * 128),
    },
    "fly_full": {
        "trainable_params": fly["trainable_params"],
        "frozen_params": fly["frozen_params"],
        "neurons": fly["N_neurons"], "synapses": 125365936,
        "connections": fly["nnz"], "sensory_inputs": fly["n_sensory"],
        "positions": 16427, "streams": 64,
        "tokens_seen": 16427 * 64,
        "wall_hours": round((fly["wall_seconds"] + 11 * 545) / 3600, 2),
        "train_flops_est": fly["train_flops_est"],
    },
}
json.dump(ledger, open(f"{RESULTS}/compute_ledger.json", "w"), indent=2)
print("ledger:", json.dumps(ledger, indent=1))
print("FIGURES DONE")
