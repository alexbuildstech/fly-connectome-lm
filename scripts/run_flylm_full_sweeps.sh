#!/bin/bash
# Sequential full-model sweeps: fly -> shuffled -> random
cd /home/z/my-project
for V in fly shuffled random; do
  echo "=== START variant=$V $(date) ==="
  until python3 src/flylm_full.py --variant $V --tag full --max-seconds 7200 2>&1 | grep -E "RESULT|train [0-9]+000/|val [0-9]+00/|budget|Error|RESULT"; do
    echo "=== RESUME variant=$V $(date) ==="
    sleep 2
  done
  echo "=== DONE variant=$V $(date) ==="
done
echo "ALL SWEEPS COMPLETE"
