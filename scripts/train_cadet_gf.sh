#!/bin/bash
# CaDet-GF (generator-free): artifact path with one head, trained on the real source clips of the VAE pairs plus
# the pairs; the epoch is chosen on the held-out pairs, so no generator-produced video is used at any point.
# Usage: bash scripts/train_cadet_gf.sh SEED data/aigvd/manifest_aigvd.csv data/pairs/manifest_pairs.csv data/frames runs/cadet_gf
set -euo pipefail
SEED=${1:?}; AIGVD=${2:?}; PAIRS=${3:?}; FRAMES=${4:?}; OUT=${5:?}/seed$SEED
mkdir -p "$OUT"
python - "$AIGVD" "$PAIRS" "$OUT/train_reals.csv" <<'PY'
import csv, sys
aigvd, pairs, out = sys.argv[1:]
src = {r["pair_id"] for r in csv.DictReader(open(pairs))}
rows = [r for r in csv.DictReader(open(aigvd)) if r["split"] == "train" and r["label"] == "0" and r["pair_id"] in src]
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} real source clips -> {out}")
PY
vae-rhythm train-artifact --single-head --train "$OUT/train_reals.csv" "$PAIRS:train" --val "$PAIRS:val" \
  --frames-root "$FRAMES" --out "$OUT" --seed "$SEED" 2>&1 | tee "$OUT/train.log"
