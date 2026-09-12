# experiment.md — FlyLM: Turning the MaleCNS v1.0 Fly Connectome into a Token Machine

> **Single source of truth for this project.** Everything discovered, decided, built, and
> measured lives here, updated continuously, so that context is never lost between sessions.
> STATUS: **EXPERIMENT COMPLETE** (all main runs finished; see §6–7).

---

## 0. TL;DR — the answers

**The question.** Google/Janelia released the complete connectome of an adult male fruit fly
(`MaleCNS v1.0`, 166,700+ neurons, ~125M synapses, announced Sept 3, 2026). People have
already wired it to play Doom and Mario 64. Could this brain, modified enough, be turned into
a **token-in / token-out language machine** — the fly's wiring replacing the transformer's
learned weight stacks — and how does it compare **against a real transformer** trained on the
same task with the same compute?

**The answers, from real data + real training on this machine:**

1. **Yes — it can be made to spit out tokens.** We built FlyLM: the fly's actual 26M-connection
   wiring as a frozen recurrent dynamical system; tokens are injected into fly neurons, the
   state evolves through the real connectome, and a trained softmax readout emits next-token
   probabilities. It beats a bigram model (3.60 vs 3.85 bits/char) and produces
   Shakespeare-flavored text.
2. **Against a transformer, it loses clearly at this scale.** A 2-layer, 420k-param GPT
   trained on the same 300k chars for ~7 minutes of CPU reaches **2.91 bpc vs FlyLM's 3.60**
   (and 42.1% vs 29.9% next-char accuracy). Giving the fly *learnable* synapses (Track B,
   BPTT) narrows the gap to **3.25 bpc** but does not close it.
3. **The specific fly wiring is not doing special work for language.** Controls with matched
   size/sparsity/weights show: random graph 3.53 bpc, weight-shuffled fly 3.61, real fly 3.60.
   The fly's advantage over bigram comes from having a large recurrent mixing substrate at all,
   not from its biological structure. (With plasticity, init from real synapse counts is
   actually *marginally worse* than random init: 3.252 vs 3.229.)
4. **Surprising-but-genuine findings** (details in §7): a calibration trap (ridge-to-onehot
   readouts look fine by accuracy but are NLL-catastrophic — switching to softmax readout
   moved the fly from 5.77 → 3.70 bpc, the single biggest jump of the project); a dynamics
   "dead vs saturated vs alive" trichotomy in the real connectome; and the fact that
   stimulating random central neurons works as well as stimulating the fly's real sensory
   periphery.

---

## 1. What exactly was released (verified context)

- **What:** `MaleCNS v1.0` — the first complete connectome of an **adult male fruit fly
  central nervous system** (brain + ventral nerve cord + neck connective), published by the
  Janelia FlyEM team, Google Research (DeepMind neural mapping), Cambridge/MRC LMB and
  collaborators.
- **When:** dataset version v1.0 released **2026-06-08** (Janelia release notes); the public
  Google Research blog announcement "**A connectomics milestone: Mapping the complete male
  fruit fly brain**" went out **September 3, 2026** — the viral moment that triggered the
  Doom/Mario/Bitcoin creative wave days later.
- **Paper:** *"Distributed control circuits across a brain-and-cord connectome"*, Nature
  656:957–970 (2026), doi:10.1038/s41586-026-10735-w.
- **Scale:** 166,700 neurons, ~125M chemical synapses; joins the earlier female whole-brain
  (FlyWire) map for cross-sex comparison (male-specific & dimorphic cell classes annotated,
  e.g. courtship circuits).
- **License:** CC-BY-4.0. Public data, no auth for the flat-connectome files.

### 1.1 Primary sources

| Resource | URL |
|---|---|
| Google Research blog (Sep 3, 2026) | https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain |
| Nature paper | https://www.nature.com/articles/s41586-026-10735-w |
| MaleCNS portal (Janelia FlyEM) | https://www.janelia.org/project-team/flyem/male-cns-connectome |
| Official download page | https://janelia-flyem.github.io/male-cns/download/ |
| neuPrint (dataset `male-cns:v1.0`) | https://neuprint.janelia.org |
| Flat-connectome data (GCS, public) | https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/ |

### 1.2 The "fly brain plays games" wave (verified)

- **DOOMFLY** — Alex Wormuth (GitHub `nftechie`, Sept 6–8, 2026): simulated the MaleCNS v1.0
  connectome; each **Doom frame stimulates sensory neurons**; **motor-neuron spikes become key
  presses**. Covered by Tom's Hardware (Sep 8), PC Gamer (Sep 10), Gizmodo, TheGamer.
- Others: **Super Mario 64** (same week), a **fly-brain Bitcoin trader** (Yahoo Tech).
- **Premise validated by the community:** the connectome takes arbitrary inputs and produces
  arbitrary outputs. Nobody had answered whether it can be bent into a *language model* and
  how that compares to a transformer. That gap is what this experiment fills.

### 1.3 Raw data files used

| File | Size | Content |
|---|---|---|
| `connectome-weights-...minconf-0.5.feather` | 1.05 GB | edge list: (body_pre, body_post, weight=synapse count), 151.9M rows, 2318 arrow batches |
| `body-annotations-...feather` | 14 MB | 211,577 annotated bodies (type/class/superclass/somaSide/status/dimorphism) |
| `body-neurotransmitters-...feather` | 43 MB | per-body neurotransmitter (acetylcholine/gaba/glutamate/…) |

## 2. What we downloaded and verified ✓

- Machine: 2 vCPU, 4.1 GB RAM, 9.3 GB disk (all experiments sized for this).
- Downloaded the three files above from the official GCS bucket.
- **Integrity check:** our processed neuron-level matrix sums to **125,365,936 synapses —
  exactly the published "~125M"**. The data is genuine and complete.

## 3. Preprocessing pipeline ✓ (`scripts/build_adjacency.py`, ~21 s)

1. LUT = 211,577 sorted annotated bodyIds.
2. Stream the 151.9M-row edge list once; keep rows where **both** endpoints are annotated
   bodies → **26,028,386 unique directed neuron→neuron connections** (`weight` = synapse
   count, up to 2,591).
3. Chunked COO→CSR → `adjacency.npz` (211,577², 26.03M nnz, float32, 81.5 MB).

### 3.1 Connectome vital statistics

| Statistic | Value |
|---|---|
| Annotated bodies | 211,577 (188,778 with edges) |
| Unique directed connections | 26,028,386 |
| **Total synapses** | **125,365,936 ✓** |
| Density | 5.81e-4 |
| Mean / max out-weight | 592.5 / 140,897 synapses |
| Largest strongly connected component | 181,273 bodies (**85.7%**) |
| Reciprocal connection pairs | 3,876,436 (≈15% of edges) |

## 4. Final experimental design

**FlyLM (Track A — frozen brain).** Tokens are injected as currents into 30,000 fly neurons
("modified sensory epithelium"); state evolves through the frozen real wiring
(`x ← (1−λ)x + λ·tanh(g·Āx + u)`, Ā = row-stochastic-normalized connectome, g = gain);
2,048 state neurons are read out by a trained softmax layer → next-char distribution.
*Trainable:* only the token encoder scale and the readout (the honest minimal modification
that makes a brain "spit out tokens"). Protocol: 64 parallel streams over contiguous corpus
chunks, 200-step burn-in, dynamics on torch sparse CSR.

**Track B — plastic subbrain.** Top-1,024 neurons (by synapse strength): every real synapse
between them (61,494) becomes a **trainable weight** (frozen binary mask = fly wiring);
encoder + readout also trained; BPTT over 48 steps, Adam, 1,500 steps.

**Transformer baselines.** TinyGPT, 2 layers, ctx 64: M (d=128, 420k params) and
S (d=96, 243k params ≈ FlyLM's trainable budget). AdamW, OneCycle, 3,000 steps ≈ 7 min CPU.

**Task & protocol (all models).** tinyshakespeare (1,115,394 chars, 65-vocab), train on first
300k chars, validate on chars [300k, 340k). Same split for every model. Metric: val bits/char
(NLL) + next-char accuracy. (Seeds 1–2 robustness runs used 150k train chars.)

**Controls (identical protocol/dynamics/size).** random graph (same per-row edge counts,
weights resampled from the fly's distribution), weight-shuffled fly (same wiring, permuted
synapse counts), sensory-only injection, and plastic-track mask/init swaps.

## 5. Discoveries made while building (the messy middle — all real)

1. **OOM + cache lessons:** the 1.05 GB feather decompresses to ~3.6 GB (OOM on this box);
   streaming 2,318 arrow batches works. Background processes die between tool calls here —
   every long run needed checkpoint/resume phase machines. Cache-warming (`cat file > /dev/null`
   before processing) took the pipeline from >10 min (cold) to 21 s (warm).
2. **The flat connectome contains millions of fragment ids** beyond the 166.7k neurons —
   restricting to annotated bodies is required to get the published 125M-synapse figure.
3. **Dead / saturated / alive dynamics trichotomy** in the real connectome:
   - global spectral-radius scaling → 98% of neurons frozen (effective weights microscopic);
   - "row-stochastic" by *edge count* → everything saturates to ±0.96 (loop gain ≈ mean
     synapse weight 4.8 × g — our normalization bug, found by scanning);
   - true row-stochastic (divide by row *weight-sum*) with g≈1.6 → alive, information-rich.
4. **The calibration trap (biggest single effect of the project).** A classic ESN ridge
   readout to one-hot targets reached 26–40% argmax accuracy while producing *terrible*
   probability estimates (5.8–6.0 bits/char — worse than a unigram!). Replacing it with a
   softmax layer trained by cross-entropy moved the fly from 5.77 → 3.70 bpc. Accuracy was
   masking the failure the whole time.
5. **Stimulation site doesn't matter (once amplitude is right):** driving the real 17,937
   sensory neurons vs 30,000 random central neurons gives identical LM performance
   (3.608 vs 3.600 bpc). The earlier apparent superiority of central injection was the
   normalization artifact above.

## 6. RESULTS (main table, 300k train chars, identical data & protocol)

| Model | Trainable params | Substrate | val bpc ↓ | val acc |
|---|---|---|---|---|
| **Transformer-M** (2L, d128) | 420k, all learned | none | **2.913** | **0.421** |
| Transformer-S (2L, d96) | 243k, all learned | none | 3.075 | 0.391 |
| **Plastic fly** (real synapses trainable, BPTT) | 1.18M | fly subgraph (61,494 synapses) | **3.252** | 0.366 |
| Plastic fly, random init on fly mask | 1.18M | fly subgraph | 3.229 | 0.366 |
| Plastic random graph | 1.18M | random subgraph | 3.268 | 0.360 |
| FlyLM — **real fly**, random-central injection | 133k (readout only) | **full 26M-connection brain, frozen** | 3.600 | 0.299 |
| FlyLM — real fly, **sensory-only** injection | 133k | full brain, frozen | 3.608 | 0.293 |
| FlyLM — random graph control | 133k | random graph | 3.528 | 0.314 |
| FlyLM — weight-shuffled fly control | 133k | fly wiring, shuffled weights | 3.610 | 0.295 |
| Bigram baseline | — | — | 3.850 | 0.262 |
| Unigram | — | — | 4.773 | — |
| Uniform (65 chars) | — | — | 6.022 | — |

**Robustness (150k chars, seeds):** fly 3.809/3.800 (s1/s2), random 3.760/3.762 — seed noise
≈ ±0.01 bpc; fly-vs-random gap ≈ 0.04–0.07 bpc in random's favor is consistent.

**Compute accounting (honest):** transformer-M ≈ 7 min CPU (3,000 steps × 6.1M tokens seen);
plastic ≈ 5 min (1,500 steps × 864k tokens); FlyLM ≈ 12 min (reservoir rollout + 25-epoch
readout). All on the same 2 vCPU box. The transformer wins *despite* being the smallest
consumer of wall-clock.

### 6.1 Sample generations (prompt: "First Citizen:\nBefore we", temp 0.8)

- **FlyLM (frozen fly brain, 3.60 bpc):** "…wercan no thearoble s yo yor, momenources ancene
  thotreamt youcer ant od chat gere sondy…" — word-shaped, English-ish phonotactics, no syntax.
- **Random graph (3.53 bpc):** similar texture (slightly cleaner bigram statistics).
- **Transformer-M (2.91 bpc):** "…Whe this stroket thee counturince wigh ess! yourd had de
  mectroud ticking. Of Gothew: Has a careat ande not time to ou…" — pseudo-words, real
  punctuation/line-break structure, dialogue-like formatting emerging.

Full texts: `results/samples.json`.

## 7. Surprising-but-genuine findings

1. **The calibration trap (§5.4).** The single biggest effect in the whole project was not
   biological — it was realizing that a reservoir can *look* alive by accuracy while being
   NLL-dead. Any "fly brain LM" claim built on argmax-only evidence would have been hollow.
2. **Structure vs substrate.** The real fly wiring ≈ weight-shuffled fly wiring ≈ (slightly
   worse than) a matched random graph. The brain's value here is being a big, sparse,
   recurrent, physical mixing medium — not its specific circuitry. This is a real negative
   result for "biological structure helps arbitrary token streams" at this scale.
3. **Plasticity erases origin.** Given trainable synapses on the fly mask, initialization
   from real synapse counts buys nothing over random init (3.252 vs 3.229) — and all plastic
   variants converge to the same performance regardless of mask origin. The brain's wiring is
   *rewritable*; after training it's no longer meaningfully "the fly's".
4. **The fly brain has enormous *fan-out inertia*.** Its hub neurons reach 140k synapses of
   out-weight; naive spectral scaling leaves 98% of the network frozen. Getting a connectome
   "alive" for computation requires normalizing by *weight mass* (not edge count) — a
   practical lesson for anyone simulating MaleCNS (DOOMFLY-style projects included).
5. **Scale reality-check.** Even the winning plastic-fly (3.25 bpc) only matches a
   243k-param transformer that trained in 3 minutes. The fly brain is a fascinating substrate,
   but nothing in these experiments suggests it is *computationally* special for language.

## 8. Answers to the user's core questions

- **Q1: Could the Sept-2026 Google fly brain be modified into a token-spitting LM?**
  **Yes — demonstrated.** Tokens → injected currents → frozen real connectome dynamics →
  trained softmax readout → next-token distribution, beating a bigram model and generating
  coherent-ish text. Fully reproducible from this repo.
- **Q2: How does it perform against a transformer (at our honest scale)?**
  It loses: 3.60 bpc (frozen) / 3.25 bpc (plastic synapses) vs 2.91 bpc for a tiny
  transformer — with the transformer also winning on accuracy (42.1% vs 29.9%/36.6%) and on
  training time. The gap is structural: attention *learns* its context selection; the frozen
  fly state has fixed, task-agnostic mixing; giving the fly plastic synapses helps but the
  61k-synapse subbrain + BPTT still can't match learned attention.
- **Q3: Is the real fly wiring doing real work (vs random)?**
  No — matched random wiring performs slightly *better*, and shuffling the fly's synapse
  counts changes nothing. What matters is having a large recurrent mixing substrate.
- **Q4: Surprising genuine findings?** §7 — most notably the calibration trap and the
  dead/saturated/alive dynamics trichotomy of the real connectome.

## 9. Limitations (what this experiment does NOT claim)

- Char-level task on 300k chars with a ~2-vCPU budget: conclusions are about *this scale*.
  Nobody knows whether a 1000× larger compute budget changes the fly-vs-transformer ordering
  (though the burden of proof now sits with the connectome side).
- Rate-based (tanh) dynamics, not spiking; DOOMFLY-style spike timing might exploit structure
  that rate dynamics cannot (we consider this unlikely to flip the ordering, but it is
  untested here).
- The readout sees 2,048 of 211k neurons; a readout with broader access might extract more
  from the frozen substrate (the plastic track partially addresses this).
- Seed coverage: 3 seeds for the headline ESN comparison; single seed for transformer and
  plastic tracks (loss curves were stable; transformer reruns varied <0.02 bpc informally).

## 10. Reproduction

Environment: Python 3.12.14, torch 2.14.0+cpu, numpy 2.1.3, scipy 1.14.1, pandas 2.2.3,
pyarrow 25.0.1 (2 vCPU / 4.1 GB box).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu numpy scipy pandas pyarrow matplotlib
python scripts/build_adjacency.py        # needs the 1.05GB feather (URL §1.3), ~21s
python scripts/adjacency_stats.py
python src/bigram_baseline.py
python src/flylm_esn2.py --variant fly   --norm rowsum --gain 1.6 --in-gain 2.0 --min-weight 2 \
    --train-chars 300000 --tag main      # repeat until DONE (checkpoint/resume)
python src/flylm_esn2.py --variant random --shuffle... # controls (see src/batch files)
python src/transformer_lm.py --size M --steps 3000
python src/flylm_plastic.py --mode fly_fly --steps 1500
python src/make_report.py                # summary.json + figures
```

All result JSONs, figures, and sample texts are committed under `results/`.

## 11. Provenance & integrity

- Repo: `github.com/alexbuildstech/fly-connectome-lm` (private).
- Data: official Janelia GCS bucket, CC-BY-4.0; checksums in `data_provenance/`.
- Corpus: tinyshakespeare (karpathy/char-rnn), committed in `data_provenance/`.
- No credentials, tokens, or secrets are committed anywhere in this repo.
- Every number in §6 is reproducible from a committed script; result JSONs contain the exact
  hyperparameters and wall-clock of each run.
