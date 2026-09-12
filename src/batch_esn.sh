#!/bin/bash
# Sequential ESN experiment batch: sweep -> main table -> sensitivity.
cd /home/z/my-project/fly-connectome-lm/src
LOG=/home/z/my-project/fly-connectome-lm/results/batch_esn.log
mkdir -p /home/z/my-project/fly-connectome-lm/results

echo "=== ESN BATCH START $(date) ===" > $LOG

# ---- Phase 1: config sweep (pruned matrix, 100k chars) ----
for cfg in "0.5 1.2" "0.9 1.2" "0.7 0.6"; do
  set -- $cfg
  leak=$1; gain=$2
  echo "--- sweep leak=$leak gain=$gain $(date)" >> $LOG
  python3 flylm_esn.py --variant fly --norm global --leak $leak --gain $gain --in-gain $gain \
    --min-weight 2 --seed 0 --train-chars 100000 --val-chars 30000 --burn-in 200 \
    --subsample 3 --tag sweep >> $LOG 2>&1
done

# ---- Phase 2: main table (pruned >=2, 250k chars) ----
for var in fly random shuffled; do
  echo "--- main variant=$var $(date)" >> $LOG
  python3 flylm_esn.py --variant $var --norm global --leak 0.7 --gain 1.2 --in-gain 1.2 \
    --min-weight 2 --seed 0 --train-chars 250000 --val-chars 40000 --burn-in 200 \
    --subsample 3 --tag main >> $LOG 2>&1
done

echo "=== ESN BATCH DONE $(date) ===" >> $LOG
