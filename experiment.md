# experiment.md — FlyLM: Turning the MaleCNS v1.0 Fly Connectome into a Token Machine

> **Single source of truth for this project.** Everything discovered, decided, built, and
> measured lives here, updated continuously, so that context is never lost between sessions.

---

## 0. TL;DR (living section — update as results land)

**Question.** Google/Janelia released the complete connectome of an adult male fruit fly
(`MaleCNS v1.0`, 166,700+ neurons, ~125M synapses). People have already wired it to play Doom
and Mario 64. Could this brain, modified enough, be turned into a **token-in / token-out
language machine** — i.e., replace the transformer's learnable weight stacks with the fly's
wiring — and how would it perform **against a real transformer** trained on the same task with
the same compute?

**Status:** data acquired and verified (125.4M synapses ✓ matches publication). Experiment
code in progress. Results will be filled in below as they land.

---

## 1. What exactly was released (verified context)

- **What:** `MaleCNS v1.0` — the first complete connectome of an **adult male fruit fly
  central nervous system** (brain + ventral nerve cord + neck connective), published by the
  Janelia FlyEM team, Google Research (DeepMind neural mapping), Cambridge/MRC LMB and
  collaborators.
- **When:** dataset version v1.0 released **2026-06-08** (Janelia release notes); the public
  Google Research blog announcement "**A connectomics milestone: Mapping the complete male
  fruit fly brain**" went out **September 3, 2026** — which is the release that went viral and
  triggered the Doom/Mario/Bitcoin creative wave in the following days.
- **Paper:** *"Distributed control circuits across a brain-and-cord connectome"*, Nature
  656:957–970 (2026), doi:10.1038/s41586-026-10735-w.
- **Scale:** 166,700 neurons, ~125M chemical synapses, joined to the earlier female whole-brain
  (FlyWire/FIB-SEM) map for cross-sex comparison (male-specific & dimorphic cell classes are
  annotated, e.g. courtship circuits).
- **License:** CC-BY-4.0. Public data, no auth needed for the flat-connectome files.
- **Prior work that made this possible:** hemibrain (2020), FANC + MANC (VNC, 2023),
  FlyWire female adult brain (2023/2024, Nature 2024).

### 1.1 Primary sources (URLs)

| Resource | URL |
|---|---|
| Google Research blog (Sep 3, 2026) | https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain |
| Google blog visuals post | https://blog.google/innovation-and-ai/technology/research/male-fruit-fly-brain-map |
| HHMI/Janelia news | https://www.hhmi.org/news/scientists-complete-full-map-fruit-fly-brain-connectome |
| Nature paper | https://www.nature.com/articles/s41586-026-10735-w |
| MaleCNS portal (Janelia FlyEM) | https://www.janelia.org/project-team/flyem/male-cns-connectome |
| Official site + download page | https://janelia-flyem.github.io/male-cns/ … `/download/` |
| neuPrint (dataset `male-cns:v1.0`) | https://neuprint.janelia.org |
| Flat-connectome data (GCS, public) | https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/ |
| Cell consortium page | https://www.cell.com/consortium/male-fly-connectome |

### 1.2 The "fly brain plays games" wave (verified)

- **DOOMFLY** — Alex Wormuth (GitHub `nftechie`, ~Sept 6-8, 2026): simulated the MaleCNS v1.0
  connectome, fed each **Doom frame into sensory neurons**, read **motor-neuron spikes as key
  presses**. Covered by Tom's Hardware (Sep 8), PC Gamer (Sep 10), Gizmodo, TheGamer, ResetEra.
  Repo described as "Fly-connectome simulation controlling a live Doom arena, with experimental
  neural plasticity, spectator UI, and scientific validation reports."
- Others wired the same connectome to **Super Mario 64** (AI-powered 3D model, same week), and
  a **fly-brain Bitcoin trader** ("I gave the fly brain $100 to trade bitcoin", Yahoo Tech).
- **Takeaway (and the premise of this experiment):** the community has already established
  that the connectome can take arbitrary **inputs** (pixels → sensory neurons) and produce
  arbitrary **outputs** (spikes → key presses). Nobody (yet) has answered whether it can be
  bent into a **language model** and how that compares to a transformer. That is the gap this
  experiment fills.

### 1.3 Raw data files (flat-connectome, minconf 0.5)

| File | Size | Content |
|---|---|---|
| `connectome-weights-...feather` | 1.05 GB | **The edge list**: (body_pre, body_post, weight=synapse count). 151.9M rows, 2318 arrow record batches |
| `body-annotations-...feather` | 14 MB | 211,577 annotated bodies: type/class/superclass/somaSide/status/dimorphism/… |
| `body-neurotransmitters-...feather` | 43 MB | per-body neurotransmitter predictions (acetylcholine/gaba/glutamate/…) |
| `body-stats-...feather` | 778 MB | per-body stats (not needed yet) |
| `syn-points`, `syn-partners` | 0.9 / 6.8 GB | raw per-synapse tables (too big for this box; `connectome-weights` is the aggregated form) |

## 2. What we downloaded and verified (done ✓)

- Machine: 2 vCPU, 4.1 GB RAM, 9.3 GB disk (everything sized for this).
- Downloaded `connectome-weights` (1.05 GB), `body-annotations` (14 MB),
  `body-neurotransmitters` (43 MB) from the official GCS bucket. sha256-head recorded.
- **Verified global figure:** our processed neuron-level matrix sums to
  **125,365,936 synapses ≈ the published ~125M** — data is genuine and complete.

## 3. Preprocessing pipeline (done ✓)

`scripts/build_adjacency.py` (single cold-tolerant streaming pass, ~21 s):

1. LUT = sorted annotated bodyIds (211,577) from `body-annotations`.
2. Stream the 151.9M-row edge list once; keep rows where **both** pre & post are annotated
   bodies → 26,028,386 unique directed neuron→neuron connections (rows are already unique
   per pair; `weight` = synapse count, max 2591).
3. Chunked COO→CSR accumulation (memory-bounded) → `adjacency.npz` (211,577 × 211,577,
   26.03M nnz, float32 synapse weights, 81.5 MB on disk).

### 3.1 Connectome vital statistics (from `scripts/adjacency_stats.py`)

| Statistic | Value |
|---|---|
| Annotated bodies (LUT size) | 211,577 |
| Bodies with ≥1 kept edge | 188,778 |
| Isolated bodies | 22,799 |
| Unique directed connections (nnz) | 26,028,386 |
| **Total synapses** | **125,365,936 ✓ (= published ~125M)** |
| Density | 5.81e-4 |
| Mean / max out-weight | 592.5 / 140,897 synapses |
| Strongly connected components | 30,047 (largest = 181,273 bodies = **85.7%**) |
| Weakly connected components | 22,944 (largest = 188,383) |
| Reciprocal connection pairs | 3,876,436 (≈15% of edges — rich recurrence) |

Neurotransmitter table join: acetylcholine / gaba / glutamate / … per body (parquet saved).

## 4. Experimental design (the actual science)

**Framing.** A transformer LM is: tokens → (learnable embedding) → stack of learned attention
weights → (learnable unembedding) → next-token logits. Our FlyLM replaces the *middle* with the
fly brain:

```
tokens ──(learnable encoder B)──▶ INPUT neurons ──▶ [FROZEN fly wiring: recurrent state
         propagation through real connectome A] ──▶ states x_t ──(learnable readout R)──▶
         next-token logits
```

- The fly's wiring `A` stays **fixed** (it's a brain, not a parameter dump): normalized
  synapse weights, iterated as a recurrent dynamical system with tanh (rate-based) dynamics.
  This is the Echo-State/Reservoir paradigm — the honest minimal modification needed to make a
  brain "spit out tokens".
- **Trainable parts:** input encoder `B` (token→neurons), readout `R` (neurons→vocab). That is
  the direct analogue of the transformer's learnable token weights, as the user framed it.
- **Baselines for the same trainable budget & data:**
  1. A real char-level **transformer** (torch, CPU-sized).
  2. A **random-graph reservoir** with matched size/sparsity (is the fly's *specific* wiring
     doing anything, or would any sparse net do?).
  3. A **weight-shuffled fly** reservoir (same degree distribution, randomized synapse counts).
  4. (optional) n-gram / bigram floor.

**Task:** next-character prediction on tinyshakespeare (1,115,394 chars, 65-vocab) — the
canonical tiny LM benchmark, small enough for this machine, hard enough to separate models.

**Metrics:** val loss (nats/char) + accuracy, learning curves, sample generations, ablation
deltas with seeds. All numbers land in `results/` + Section 6.

## 5. Decisions log

- Use the **annotated-body** neuron set (166.7k neurons + fragments with annotations) — the
  fragment-only ids in the raw file (millions) are not part of the published brain map.
- Rate-based (tanh) dynamics, not spiking: spiking sims (like DOOMFLY's) need precise timing
  machinery; rate dynamics are the standard reservoir-computing bridge from connectome to ML,
  and they backprop-friendly if we later unfreeze parts of A.
- Char-level LM, CPU-only training; transformer sized to ~1M params.
- Every number reported with the exact script that produced it (commit hash in results files).

## 6. Results (to be filled)

_(running table — updated as each run finishes)_

## 7. Answers to the user's core questions (to be filled at the end)

- **Q1: Could the Sept-2026 Google fly brain be modified into a token-spitting LM?**
- **Q2: How does it perform against a transformer (at our scale)?**
- **Q3: Is the *real* fly wiring doing real work (vs random graphs)?**
- **Q4: What surprising-but-genuine findings emerged?**

## 8. Reproduction

Environment: Python 3.12.14, torch 2.14.0+cpu, numpy 2.1.3, scipy 1.14.1, pandas 2.2.3,
pyarrow 25.0.1. 2 vCPU / 4.1 GB RAM box.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu numpy scipy pandas pyarrow matplotlib
python scripts/build_adjacency.py       # needs the 1.05GB feather (URL in §1.3)
python scripts/adjacency_stats.py
python src/train_flylm.py               # + train_transformer.py, ablations
```

## 9. Provenance & integrity

- Repo: `github.com/alexbuildstech/fly-connectome-lm` (private).
- Data: official Janelia GCS bucket, CC-BY-4.0; sha256-head of the 1.05GB edge list recorded
  in `data_provenance/`.
- No credentials, tokens, or secrets are committed anywhere in this repo.
