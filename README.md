# VAE Rhythm: Temporal Decoding Traces for AI-Generated Video Detection

**Paper** (arXiv, coming soon) · [**Project page**](https://showlab.github.io/vae-rhythm/) · **Models** (coming soon) · **VAE pairs & scores** (coming soon) · [**Online demo**](https://checkai.space/dashboard?mode=video)

Most video generators decode their latents with a 3D causal VAE that groups consecutive frames in time
(4 frames per group for most families, 8 for LTX). Frame fidelity therefore varies with a frame's position
inside its group, which leaves a periodic trace in every generated video. We call it **VAE Rhythm**.

A training-free **phase scan** reveals it: reconstruct a clip with an off-the-shelf VAE (Wan2.1) at 24
successive temporal offsets and measure the *periodicity*, not the level, of the reconstruction error.
Generated videos give periodic residual curves; real videos give nearly flat ones.

This repository contains everything used in the paper:

| Part | What it does | Entry point |
|---|---|---|
| **Phase scan** | zero-shot rhythm statistics (`C4`, `C8`, `Ppk`) and residual curve for any video, no training | `vae-rhythm scan` |
| **VAE Rhythm plug-in** | adds rhythm evidence to any existing detector's scores (paper Eq. 3) | `vae-rhythm plugin` |
| **VAE-pair engine** | turns real videos into training pairs that differ only by one VAE round trip | `vae-rhythm make-pairs` |
| **CaDet** | detector with two paths on one frozen DINOv3 ViT-L/16 (artifact path trained with VAE pairs, semantic path without) | `vae-rhythm train-*`, `predict` |

## Installation

Requirements: Linux (tested; macOS works for the CPU tests), Python ≥ 3.10, **ffmpeg ≥ 4.4 with libx264 and libx265** on `PATH`,
and a CUDA GPU for scanning and training (a 24 GB card is enough for 720p scans with one window per VAE call; `--batch 2` needs about 40 GB).

```bash
git clone https://github.com/showlab/vae-rhythm vae-rhythm && cd vae-rhythm
conda create -n vae-rhythm python=3.10 -y && conda activate vae-rhythm
conda install -c conda-forge ffmpeg -y          # or use your system ffmpeg
pip install -e ".[cadet,dev]"                   # core + CaDet + tests
pytest -q                                       # CPU tests, ~5 s
```

The Wan2.1 VAE (≈ 0.5 GB) is downloaded from Hugging Face on first use. CaDet additionally needs the
DINOv3 ViT-L/16 weights through `timm` (`vit_large_patch16_dinov3.lvd1689m`); accept the DINOv3 license
on Hugging Face and log in with `huggingface-cli login` if the download is refused.

> Handing this repository to a coding agent? Point it to [`AGENTS.md`](AGENTS.md): it contains a
> step-by-step, verifiable setup procedure.

## Quick start: scan a video (training-free)

```bash
vae-rhythm scan generated.mp4 real.mp4 --out scan_out/
# generated.mp4: Ppk 0.8321 (period 4.00), C4 0.9425, C8 0.9576, z(Ppk) +3.80, above 100% of reference real videos
# real.mp4: Ppk 0.2007 (period 8.73), C4 0.1552, C8 0.3665, z(Ppk) -0.71, above 27% of reference real videos
```

(A Wan 2.2 text-to-video clip and a real clip from DF26, scanned at 1280 × 720 in about two minutes each on a 24 GB GPU.)

`scan_out/<name>.png` shows the residual curve r(k) and its spectrum; `scan_out/<name>.json` holds every
statistic. In Python:

```python
from vae_rhythm import scan_video
res = scan_video("my_video.mp4")        # Wan2.1 VAE, K=24 offsets, W=13-frame windows, short side capped at 720
print(res.ppk, res.z_ppk, res.stats["C4"], res.segments[0].curve)
```

How to read the numbers:

* `Ppk` is the period-agnostic rhythm score (largest normalised periodogram bin for periods 2.5–9 frames);
  `C4` / `C8` are the shares of residual variance in the period-4 / period-8 harmonic combs.
* `z(Ppk)` standardises `Ppk` against 299 real validation videos scanned with the same protocol
  (bundled in `vae_rhythm/calibration/`). About 5 % of real clips exceed z ≈ 2.35.
* **A weak rhythm is not evidence that a video is real.** Low motion, frame-rate conversion, upscaling
  and heavy re-encoding all weaken it, and the scan must run near the generation scale: scanning at a much
  higher resolution than the video was generated at hides the rhythm (keep the default 720p cap, or 480p
  for typical 2024–2026 generators). See the paper's limitations.
* `--segments 3` also scans the middle and the end of longer videos and reports the median.

## The plug-in with your own detector

Given per-video scores of any detector and the `Ppk` of the same videos (from `vae-rhythm scan-manifest`),
the plug-in computes `s = z(s_d) + max(0, z(Ppk))`, where `z(s_d)` is the hybrid rank standardisation of the
detector score on real calibration videos:

```bash
vae-rhythm scan-manifest test.csv --out ppk.csv --npz-dir npz/                 # Ppk of the test videos
vae-rhythm plugin --scores det_test.csv --det-real det_val_reals.csv --ppk ppk.csv \
                  --manifest test.csv --out fused.csv                          # prints AUROC before / after
```

All CSVs use a `video_id` column; manifests also have `path`, `label` (0 real, 1 generated) and `subset`
(generator name). See [`docs/data.md`](docs/data.md).

## Pretrained CaDet

The CaDet weights (three seeds, a few MB each) are coming soon on Hugging Face. Once they are public:

```bash
huggingface-cli download <org>/CaDet --local-dir checkpoints/cadet       # seed0/ seed1/ seed2/
vae-rhythm predict my_video.mp4 --weights checkpoints/cadet/seed0
# my_video.mp4: CaDet score ... (semantic z ..., artifact z ...)
```

Scores are sums of z-scores relative to real validation videos (0 ≈ a typical real video; higher = more
likely generated); they are not probabilities. For a binary decision, pick a threshold on real videos of
your own domain (e.g. `vae_rhythm.plugin.threshold_at_fpr`).

## Reproducing the paper

[`docs/reproduce.md`](docs/reproduce.md) gives the commands for every stage:

1. **Data**: AIGVDBench (train / val / test), DF26, GenBuster-Bench++ and the 2026 closed-source sets; manifests.
2. **Census and zero-shot** (Fig. 3, Table 1): `scan-manifest` at 480p (and 720p) + `evaluate`.
3. **Plug-in** (Table 3): native-resolution scans capped at 720p, calibration on AIGVDBench validation reals.
4. **VAE pairs** (Sec. 4.1): `scripts/make_paper_pairs.sh` (two engine passes, 9:1 train / val split).
5. **CaDet** (Table 2): `scripts/train_cadet.sh` trains both paths for one seed, calibrates and evaluates.

Main results (AUROC %, mean over generators; 3 seeds; trained on AIGVDBench Open-Sora + real clips only):

| | AIGVDBench 20 open | 19 unseen | 11 closed | 2026 closed (3 models) | DF26 all | DF26 closed | DF26 2AFC |
|---|---|---|---|---|---|---|---|
| Best retrained baseline (Effort) | 84.3 | 83.4 | 92.5 | 81.6 | 70.4 | 73.4 | 82.6 |
| CaDet-GF (no generated video in training) | 90.9 | 90.5 | 91.9 | 84.8 | 65.8 | 51.3 | 70.5 |
| **CaDet** | **93.9** | **93.6** | **95.8** | **88.3** | **78.5** | **79.0** | **87.8** |

On DF26 the plug-in improves all 19 tested detectors by 6.7 to 23.7 AUROC points.

## Repository layout

```
vae_rhythm/
  stats.py          phase-scan statistics: residual curve r(k), C4 / C8 / Ppk, phase, spectrum (NumPy only)
  vaes.py           VAE wrappers (Wan2.1 / Wan2.2 / HunyuanVideo / CogVideoX / LTX-Video) for round trips
  video.py          ffmpeg / PyAV I/O and the scan protocol (cap, x264 CRF 23, GOP 32)
  scan.py           phase scan of frames, videos and manifests; plots; z(Ppk) calibration
  plugin.py         Eq. 3: hybrid standardisation, positive-part fusion, operating points
  metrics.py        AUROC with ties, per-generator tables, paired 2AFC
  engine/           VAE-pair engine (chain.py: post-processing chain, pairs.py: driver)
  cadet/            CaDet: frame cache, augmentations, models, training, inference / export
  calibration/      bundled real-video calibration of Ppk (Wan2.1, K24/W13, 720p cap)
  cli.py            `vae-rhythm` command line
scripts/            paper pipelines (pairs, CaDet training)
docs/               data preparation, protocol details, reproduction guide
tests/              CPU unit tests
```

## Limitations

The rhythm is a property of how a video was decoded, so it is strongest for clips delivered close to the
generator's own output. It weakens with low motion, frame-rate conversion, interpolation, upscaling and
strong re-encoding, and it depends on scan scale. Videos that were never VAE-decoded in time (frame-wise
image generators, heavy editing) carry no rhythm. Use it as additional evidence, as the plug-in does, not
as the only test.

## Citation

```bibtex
@article{sun2026vaerhythm,
  title   = {VAE Rhythm: Temporal Decoding Traces for AI-Generated Video Detection},
  author  = {Sun, Ke and Li, Shuwei and Mei, Haiyang and Shou, Mike Zheng},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License and acknowledgements

Code: MIT (see `LICENSE`). Released CaDet weights are adapters for DINOv3 ViT-L/16 and are subject to the
[DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/); the phase scan uses
the Wan2.1 VAE (Apache-2.0). The semantic path follows the training recipe of
[Effort](https://github.com/YZY-stack/Effort-AIGI-Detection); benchmarks: AIGVDBench, DF26, GenBuster-Bench++.
