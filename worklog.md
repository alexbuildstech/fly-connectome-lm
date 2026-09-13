
---
Task ID: 2
Agent: main (Super Z)
Task: Recover prior session state from GitHub; audit compromises; plan FULL model (v3).

Work Log:
- Discovered origin/main already contains 5 commits of prior-session work (context was lost, repo persisted).
- Prior results (tiny Shakespeare, 300k train chars): transformer_M 2.913 bpc / acc 0.421; plastic subbrain 3.23-3.27; ESN-fly 3.600; ESN-random 3.528; ESN-shuffled 3.610; bigram 3.850.
- AUDIT of prior compromises (the "weak version" user rejects):
  1. ESN: tokens drove only 30,000 random neurons (--pool-size 30000) of 166k+
  2. ESN: readout sampled only 2,048 neurons of full state (--readout 2048)
  3. ESN: --min-weight 2 dropped all 1-synapse connections
  4. Corpus: only 300,000 of 1,115,394 chars for BOTH arms
  5. Plastic arm: K=1024-neuron SUBGRAPH only (BPTT on subbrain)
  6. Transformer: 300k chars only
- data/malecns/processed/ is EMPTY (processed adjacency lost between sessions). Raw feathers re-downloaded fresh (data/): connectome-weights 1.05GB (151.9M rows), body-annotations 14.5MB, body-nt 43.3MB.
- Prior connectome meta (to verify against): 211,577 bodies; 26,028,386 unique directed connections; 125,365,936 synapses; largest strong component 181,273.

Stage Summary:
- Plan v3 "FlyLM-Full": full corpus, full sensory input drive, FULL 166k-neuron state readout, min-weight 1 (all 26M edges), 3 seeds on fly, controls (shuffled/random/bigram/transformer-full), extended eval suite (induction probe, memory-depth probes, loss-vs-position, samples, compute ledger). BPTT-full arm (trainable encoder+readout through frozen full brain) as stretch.

---
Task ID: 3
Agent: main (Super Z)
Task: Build + run FlyLM-FULL v2 (the full model the user demanded).

Work Log:
- Rebuilt full adjacency from raw feather: VERIFIED exact match to prior meta (211,577 bodies, 26,028,386 connections, 125,365,936 synapses).
- Benchmarks: torch sparse.mm (int32+RCM) 183ms/step at B=64; dense gemm 100-140 GFLOPS.
- v1 (flylm_full.py): full-sensory input (17,937 neurons), full-state readout (211,577), online Adam per-batch.
  FAILED scientifically: val 4.55 bpc, worse than bias-only; diagnosed (a) per-batch noise-fitting/interference,
  (b) stacked-seed collapse: Adam scale-invariance makes multi-init seeds converge to IDENTICAL weights (verified: diff norm 2e-4 vs col-norm 24).
- v2 (flylm_full2.py): grad accumulation E=4 (256 ex/update) + AdamW decoupled wd 1e-2 + single readout
  + projection storage (4096-d, every 4th val... train position) for offline-fit arm + probes.
  Healthy: train loss 4.18 -> 2.59 nats @ pos 9.5k and falling.
- Transformer baseline upgraded: size L = d320 5L ffn1280 ctx128 = 5.15M params (12x prior M), 0.62s/step at B=32 ctx128 measured on (256,4,1024) config.
- Env lesson: background processes get reaped between tool calls -> drive training via repeated 10-min checkpointed calls.

Stage Summary:
- Fly-full sweep ~60% done. Next: finish fly train+val, offline-fit arm from projections, transformer-L run, shuffled/random controls, eval suite (induction, memory-depth probes, generation, compute ledger), experiment.md, push.

---
Task ID: 4
Agent: main (Super Z)
Task: Complete all training arms + full evaluation suite + final report.

Work Log:
- FlyLM-FULL v2 fly: TRAIN+VAL complete -> 3.644 bpc / 0.294 acc (full corpus, 211k state readout, 17,937 sensory inputs).
- Transformer-L (6.25M params, d320 5L ffn1280 ctx128, 4000 steps): 2.266 bpc / 0.537 acc.
- Shuffled control: 3.629 / 0.296. Random control: 3.546 / 0.309. Bigram (full corpus): 3.572 / 0.272.
- Memory-depth probes: lag0 0.273, lag1 0.185, lag2+ ~ prior 0.148 -> fly memory horizon ~1-2 chars (matches leak 0.7 dynamics).
- Induction zero-shot: fly +1.9% gain (neutral), transformer -39.1% (ANTI-induction: corpus statistics suppress verbatim copying).
- Generation: fly = word-adjacent char soup; transformer = readable pseudo-Shakespeare.
- Compute ledger: comparable flop-equivalents (~6.1e14 vs ~8.6e14) and wall-clock (1.21h vs 1.76h).
- Fixed: transformer_lm.py RESULTS path bug (wrote to old dir; recovered JSON+ckpt); fig scripts.
- experiment.md rewritten as the comprehensive final report (question, background, design, all results, verdict, repro).

Stage Summary:
- EXPERIMENT COMPLETE. Final commits + push follow.
