#!/bin/bash
# Download the raw MaleCNS v1.0 connectome files (public GCS, CC-BY 4.0).
#
# You only need this if you want to rebuild data/malecns/processed/adjacency.npz
# from scratch (scripts/build_adjacency.py). The repo already ships the rebuilt
# adjacency matrices, so for reproducing the training/eval results you can skip
# this step entirely.
#
# Total download: ~1.11 GB.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
mkdir -p data/malecns

BASE="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

download () {  # name, expected_bytes
  local name=$1 want=$2 out="data/malecns/$1"
  if [ -f "$out" ]; then
    local have=$(stat -c%s "$out")
    if [ "$have" = "$want" ]; then
      echo "OK (cached): $name ($have bytes)"
      return 0
    fi
    echo "RE-DOWNLOADING $name: found $have bytes, expected $want"
  fi
  echo "downloading $name ($want bytes)..."
  curl -fL --retry 3 -o "$out" "$BASE/$name"
  local have=$(stat -c%s "$out")
  if [ "$have" != "$want" ]; then
    echo "SIZE MISMATCH for $name: got $have, expected $want" >&2
    exit 1
  fi
}

# 151,856,684 raw rows (body_pre, body_post, weight=synapse count), 1.05 GB
download connectome-weights-male-cns-v1.0-minconf-0.5.feather 1051241946
# 211,577 annotated bodies x 36 columns (type, class, superclass, status, ...)
download body-annotations-male-cns-v1.0-minconf-0.5.feather 14483314
# neurotransmitter predictions per body (acetylcholine, gaba, ...)
download body-neurotransmitters-male-cns-v1.0.feather 43282834

echo
echo "All files present under data/malecns/. Sanity check the adjacency rebuild:"
echo "  python3 scripts/build_adjacency.py"
echo "Expected (verified repeatedly on this repo):"
echo "  211,577 bodies; 26,028,386 unique directed connections; 125,365,936 synapses."
