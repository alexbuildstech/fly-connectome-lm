# fly-connectome-lm

**Can the fruit fly brain released by Google (MaleCNS v1.0, Sept 2026) be modified into a
token-in/token-out language machine — and can it stand up to a transformer?**

This repo is a real-data, real-training experiment. No mocks, no toys.

- The **grand narrative + all results**: read [`experiment.md`](experiment.md).
- Data: official Janelia/Google MaleCNS v1.0 flat connectome (CC-BY-4.0) — 211,577 annotated
  bodies, 26M directed neuron→neuron connections, **125.37M chemical synapses (verified
  against the published figure)**.
- Method: the fly connectome is a **frozen recurrent backbone** (Echo-State style); learnable
  token encoder + readout turn it into a character-level LM. Baselines: a torch transformer,
  random-graph reservoirs, and weight-shuffled fly controls, all on tinyshakespeare.
- Everything reproducible: `scripts/` (data pipeline) + `src/` (models & training).

## Layout

```
experiment.md            the full experiment log (context, method, results, verdicts)
scripts/                 data acquisition + preprocessing (connectome -> adjacency.npz)
src/                     FlyLM, transformer baseline, ablations, eval
results/                 metrics, curves, samples (committed)
data_provenance/         corpus (tinyshakespeare) + source checksums
```

Data files themselves are not committed (1GB+); the pipeline re-downloads them from the
official GCS bucket (URLs in experiment.md §1.3).
