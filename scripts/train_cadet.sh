#!/bin/bash
# Train CaDet (both paths) for one seed, calibrate on AIGVDBench validation reals, evaluate on a test manifest.
# Usage: bash scripts/train_cadet.sh SEED data/aigvd/manifest_aigvd.csv data/pairs/manifest_pairs.csv data/frames runs/cadet
# Prerequisite: vae-rhythm cache-frames for both manifests into the same frames root.
# Each path takes several hours per seed on a single 24-48 GB GPU.
set -euo pipefail
SEED=${1:?seed}; AIGVD=${2:?}; PAIRS=${3:?}; FRAMES=${4:?}; OUT=${5:?}/seed$SEED
mkdir -p "$OUT"
vae-rhythm train-artifact --train "$AIGVD:train" "$PAIRS:train" --val "$AIGVD:val" --frames-root "$FRAMES" \
  --out "$OUT" --seed "$SEED" 2>&1 | tee "$OUT/train_artifact.log"
vae-rhythm train-semantic --train "$AIGVD:train" --frames-root "$FRAMES" --out "$OUT" --seed "$SEED" 2>&1 | tee "$OUT/train_semantic.log"
vae-rhythm calibrate --weights "$OUT" --val "$AIGVD:val" --frames-root "$FRAMES"
vae-rhythm predict --weights "$OUT" --manifest "$AIGVD:test" --frames-root "$FRAMES" --out "$OUT/scores_aigvd_test.csv"
vae-rhythm evaluate "$OUT/scores_aigvd_test.csv" --manifest "$AIGVD:test" --out "$OUT/aigvd_test_per_generator.csv"
