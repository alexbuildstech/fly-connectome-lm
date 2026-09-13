# fly-connectome-lm

**Can the fruit fly brain released by Google (MaleCNS v1.0, Sept 2026) be turned into a
token-in/token-out language model — and can it stand up to a transformer at matched compute?**

Real data, real training, full scale. No mocks, no toys, no subsampling in the final runs.

## Headline result (full corpus, identical train/val split, same 64k-char val)

| Arm | bits/char ↓ | top-1 acc ↑ |
|---|---|---|
| **Transformer-L** (6.25M trainable, 5 layers, ctx 128) | **2.266** | **0.537** |
| Random graph reservoir (frozen, 26M synapses) | 3.546 | 0.309 |
| Shuffled fly connectome (frozen) | 3.629 | 0.296 |
| **Real fly connectome MaleCNS v1.0 (frozen, FULL)** | 3.644 | 0.294 |
| Bigram baseline | 3.572 | 0.272 |
| Uniform | 6.033 | 0.015 |

**Mechanistic findings** (details + figures in [`experiment.md`](experiment.md)):
- The fly reservoir's usable memory horizon is **~1–2 characters** (linear probes on brain
  states: lag-0 27.3%, lag-1 18.5%, lag ≥ 2 at chance) — exactly what its leaky dynamics
  (0.3^k trace decay) predict. This, not capacity, is why it loses to attention.
- The fly's **specific wiring contributes nothing** for next-token statistics: it ties its
  own degree-shuffled control and slightly trails a matched random graph.
- Zero-shot induction test: the transformer shows strong **anti-induction** (−39% gain on
  repeated text fragments — its corpus prior fights verbatim repetition); the fly is neutral.
- A cautionary negative result: naive online Adam on a 211k-dim reservoir readout scores
  *worse than its own bias* — high-dimensional reservoir readouts need gradient accumulation
  + decoupled weight decay (both fixes are in `src/flylm_full2.py`).

## The full model (what "no weak version" means here)

- ALL 211,577 annotated bodies simulated; ALL 26,028,386 directed connections kept
  (min-weight 1); 125,365,936 synapses (verified against the published figure).
- Tokens drive **all 17,937 sensory neurons** (olfactory / optic / auditory / mechanosensory
  populations from official annotations) through a fixed random encoder.
- Readout = softmax over the **full 211,577-neuron state** (13.75M trainable params),
  AdamW + gradient accumulation, one online pass over the corpus.
- Controls trained at the identical protocol: transformer-L, shuffled wiring, random wiring.

## Layout

```
experiment.md            full experiment record: context, method, results, verdict, repro
worklog.md               append-only session log
scripts/                 data pipeline + eval suite (01-08) — build adjacency, benchmarks, probes, induction, figures
src/                     flylm_full2.py (final trainer), transformer_lm.py, reservoir_lib.py, ablations
results/                 metrics JSONs, figures, generation samples, search provenance (committed)
data_provenance/         corpus + source URLs/checksums
```

Data files (>100 MB) are not committed (GitHub limit); re-download from the official
Janelia GCS bucket — URLs in `experiment.md` §2.3 / `data_provenance/SOURCES.txt`.
Adjacency caches (77–85 MB) are committed for convenience and are regenerable via
`scripts/build_adjacency.py` (41 s, exact-match verified).

## Reproduce

```bash
python3 scripts/build_adjacency.py
python3 src/flylm_full2.py --variant fly --tag full2 --store-proj     # repeat until RESULT
python3 src/flylm_full2.py --variant shuffled --tag full2
python3 src/flylm_full2.py --variant random  --tag full2
python3 src/transformer_lm.py --size L --steps 4000 --ctx 128 \
    --train-chars 1051394 --val-chars 64000 --tag full
python3 scripts/06_eval_suite.py bigram|probes|induction|generate
python3 scripts/07_induction_fragments.py && python3 scripts/08_report_assets.py
```
