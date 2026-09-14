#!/bin/bash
# Session-4 frozen battery: one variable changed per run vs the v2 protocol.
# Sequential by design — 4 GB box, nothing else may run concurrently.
cd "$(dirname "$0")/../src"
R=../results
MAINLOG=${MAINLOG:-/tmp/v3frozen.log}

bpc() { python3 -c "import json; print(json.load(open('$1'))['bits_per_char'][0])" 2>/dev/null || echo 99; }

run () {  # name, args...
  local name=$1; shift
  echo "=== $name : $* ===" | tee -a "$MAINLOG"
  python3 flylm_v3_frozen.py --tag v3 "$@" > "/tmp/v3run_$name.log" 2>&1
  local rc=$?
  grep -E "RESULT|budget" "/tmp/v3run_$name.log" | tee -a "$MAINLOG"
  if [ $rc -ne 0 ]; then echo "$name FAILED rc=$rc (see /tmp/v3run_$name.log)" | tee -a "$MAINLOG"; tail -5 "/tmp/v3run_$name.log" | tee -a "$MAINLOG"; fi
}

# ---- E/I sign axis (critique #2) ----
run flysigned --variant flysigned --gain 1.6
b=$(bpc $R/flylm_v3_flysigned_leak0.7_dl0.0_gain1.6_s0.json)
if [ "$b" != "99" ] && python3 -c "exit(0 if float('$b') > 4.5 else 1)"; then
  echo "flysigned gain1.6 degenerate (bpc=$b) -> retry gain 2.4" | tee -a "$MAINLOG"
  run flysigned_g24 --variant flysigned --gain 2.4
  b2=$(bpc $R/flylm_v3_flysigned_g24_leak0.7_dl0.0_gain2.4_s0.json)
  if [ "$b2" != "99" ] && python3 -c "exit(0 if float('$b2') > 4.5 else 1)"; then
    echo "gain 2.4 still degenerate (bpc=$b2) -> retry gain 3.2" | tee -a "$MAINLOG"
    run flysigned_g32 --variant flysigned --gain 3.2
  fi
fi
run shuffledsigned --variant shuffledsigned --gain 1.6
run randomsigned   --variant randomsigned   --gain 1.6

# ---- leak / timescale axis (critique #3) ----
run fly_leak09  --variant fly --leak 0.9
run fly_leak099 --variant fly --leak 0.99
run fly_multi   --variant fly --leak multi:0.3,0.7,0.99

# ---- delays (critique #4) ----
run fly_delay      --variant fly --delay-frac 0.3
run fly_slow_delay --variant fly --leak 0.99 --delay-frac 0.3

echo "ALL FROZEN RUNS DONE" | tee -a "$MAINLOG"
