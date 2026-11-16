#!/bin/bash
# VAE pairs of the paper (Sec. 4.1): two engine passes over the AIGVDBench training reals, merged.
# Usage: bash scripts/make_paper_pairs.sh data/aigvd/manifest_aigvd.csv data/pairs [NUM_GPUS]
# Pass A runs NUM_GPUS shards in parallel (random parameters are fixed per clip, so the shard count does not
# change the result); pass B reproduces the paper's 1,333 clips = shards 0 and 1 of 3. Re-running resumes.
# The paper's run gave 5,333 pairs: train 4,818 (3,607 A + 1,211 B), val 515 (393 A + 122 B).
set -euo pipefail
MANIFEST=${1:?manifest of AIGVDBench (with split column)}; OUT=${2:?output folder}; NG=${3:-1}
run_pass () {  # $1 tag, $2 out subfolder, $3 nshards, $4 shards to run, remaining: engine options
  local TAG=$1 DIR=$OUT/$2 NS=$3 SHARDS=$4; shift 4; local g=0
  for i in $SHARDS; do
    CUDA_VISIBLE_DEVICES=$((g % NG)) vae-rhythm make-pairs --reals-manifest "$MANIFEST_SRC" --out "$DIR" --tag "$TAG" \
      --shard $i --nshards $NS "$@" > "$DIR.shard$i.log" 2>&1 &
    g=$((g + 1))
  done
  wait
}
mkdir -p "$OUT"
python - "$MANIFEST" "$(dirname "$0")/paper_pair_sources.txt" "$OUT/sources.csv" <<'PY'
import csv, sys
man, ids, out = sys.argv[1:]
keep = set(open(ids).read().split())
rows = [r for r in csv.DictReader(open(man)) if r["label"] == "0" and r["split"] == "train" and r["pair_id"] in keep]
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} of {len(keep)} source clips found -> {out}")
PY
MANIFEST_SRC="$OUT/sources.csv"
# pass A: 4,000 clips, G in {256 x4, 360 x2, 480 x2, 720}, interpolation 20 %, duplicate-frame conversion 25 %
run_pass _v3 passA $NG "$(seq -s ' ' 0 $((NG - 1)))" --vaes wan21,hunyuan,cogx,ltx --gen-scales 256,256,256,256,360,360,480,480,720 \
  --out-scales 480,720,1080 --allow-upscale --interp-prob 0.2 --dup-prob 0.25 --n 4000 --seed 0
# pass B: 2,000 clips, mostly clean, G in {256 x3, 360}, interpolation 10 %, duplicate-frame conversion 10 %
run_pass _v3c passB 3 "0 1" --vaes wan21,hunyuan,cogx,ltx --gen-scales 256,256,256,360 \
  --out-scales 480,720,1080 --allow-upscale --interp-prob 0.1 --dup-prob 0.1 --n 2000 --seed 13
python - "$OUT" <<'PY'
import csv, glob, os, sys
out = sys.argv[1]
rows = [r for f in sorted(glob.glob(os.path.join(out, "pass*", "manifest*.csv"))) for r in csv.DictReader(open(f))]
rows = [r for r in rows if os.path.exists(r["path"])]
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(os.path.join(out, "manifest_pairs.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
n = {s: sum(1 for r in rows if r["split"] == s and r["label"] == "1") for s in ("train", "val")}
print(f"{len(rows) // 2} pairs (train {n['train']}, val {n['val']}) -> {out}/manifest_pairs.csv")
PY
