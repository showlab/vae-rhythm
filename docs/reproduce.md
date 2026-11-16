# Reproducing the paper

Commands assume the environment of `README.md` and the manifests of `docs/data.md`
(`data/aigvd/manifest_aigvd.csv`, `data/df26/manifest_df26.csv`, …). Long commands resume when re-run;
split them over GPUs with `--shard i --nshards n` and concatenate the CSVs.

## 1. Census and zero-shot detection (Sec. 3, Fig. 3, Table 1)

```bash
# 480p scans of the AIGVDBench test split (and likewise DF26 and the 2026 sets)
vae-rhythm scan-manifest data/aigvd/manifest_aigvd.csv:test --cap 480 --out scans/aigvd_480.csv --npz-dir scans/npz_aigvd_480
# zero-shot AUROC per generator from the rhythm statistic alone
vae-rhythm evaluate scans/aigvd_480.csv --manifest data/aigvd/manifest_aigvd.csv:test --column C4
vae-rhythm evaluate scans/aigvd_480.csv --manifest data/aigvd/manifest_aigvd.csv:test --column Ppk
```

Mean residual profiles (Fig. 3) are the per-generator means of `vae_rhythm.stats.residual_curve(E)` over the
saved error tables (`np.load(npz)["E"]`); phase concentration `R` uses `stats.phase_angle` and
`stats.phase_concentration` after subtracting the mean real profile.

## 2. VAE Rhythm plug-in (Sec. 4.3, Table 3)

```bash
# Ppk at native resolution capped at 720p, for the test sets and the calibration reals
vae-rhythm scan-manifest data/df26/manifest_df26.csv --cap 720 --out scans/df26_720.csv --npz-dir scans/npz_df26_720
# detector scores: any detector; for CaDet see step 4. det_val_reals.csv = its scores on the AIGVDBench validation reals
vae-rhythm plugin --scores det_df26.csv --det-real det_val_reals.csv --ppk scans/df26_720.csv \
                  --manifest data/df26/manifest_df26.csv --out fused_df26.csv
```

`--norm meanstd` gives the paper's original standardisation (sensitivity table); `--tau` shifts the rhythm
term. The bundled `z(Ppk)` calibration comes from 299 AIGVDBench validation reals scanned at the 720p cap;
to recompute it, scan those reals and take the mean and standard deviation of `Ppk`.

## 3. VAE pairs (Sec. 4.1)

```bash
bash scripts/make_paper_pairs.sh data/aigvd/manifest_aigvd.csv data/pairs 4      # 4 GPUs
```

Two engine passes over the AIGVDBench training reals (4,000 + 2,000 clips) with the chain parameters of the
paper; the merged manifest `data/pairs/manifest_pairs.csv` has a 9:1 train / val split by source clip. The
released pairs (`docs/data.md`) can be used instead; exact files differ slightly across ffmpeg versions.

## 4. CaDet (Sec. 4.2, Table 2)

```bash
vae-rhythm cache-frames data/aigvd/manifest_aigvd.csv --out data/frames
vae-rhythm cache-frames data/pairs/manifest_pairs.csv --out data/frames
vae-rhythm cache-frames data/df26/manifest_df26.csv --out data/frames
for s in 0 1 2; do bash scripts/train_cadet.sh $s data/aigvd/manifest_aigvd.csv data/pairs/manifest_pairs.csv data/frames runs/cadet; done
```

`train_cadet.sh` trains the artifact path (benchmark + pairs) and the semantic path (benchmark only), writes
`calibration.json` from the validation reals, and evaluates on the AIGVDBench test split. Other test sets:

```bash
vae-rhythm predict --weights runs/cadet/seed0 --manifest data/df26/manifest_df26.csv --frames-root data/frames --out runs/cadet/seed0/scores_df26.csv
vae-rhythm evaluate runs/cadet/seed0/scores_df26.csv --manifest data/df26/manifest_df26.csv
```

With the released weights replace `runs/cadet/seed0` by the downloaded `checkpoints/cadet/seed0`.

**CaDet-GF** (generator-free): `bash scripts/train_cadet_gf.sh SEED data/aigvd/manifest_aigvd.csv data/pairs/manifest_pairs.csv data/frames runs/cadet_gf`
trains one LoRA path with a single head and the original labels on the 4,000 real source clips plus the pairs,
and keeps the epoch with the best AUROC on the held-out pairs. Its score is the head's logit.

## Expected numbers

See the paper's Tables 1–3. Seed-to-seed spread for CaDet is about 0.6–0.7 AUROC on AIGVDBench; CaDet-GF
varies more (about 3 AUROC).
