# Phase-scan protocol

Every number in the paper uses the protocol below. Changing any item changes the statistics and
invalidates the bundled real-video calibration (`vae_rhythm/calibration/wan21_k24w13_cap720.json`).

| Item | Value |
|---|---|
| Ruler VAE | Wan2.1 (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, `vae/`), posterior mode, bf16 |
| Offsets × window | K = 24 start offsets, W = 13 frames per window (36 consecutive frames) |
| Error | per-frame RGB MSE with pixels in [-1, 1] |
| Resolution | native; area downscale only when the short side exceeds the cap (720 for the plug-in and CaDet comparisons, 480 for the census); never upscale |
| Re-encoding | the whole clip, real or generated: libx264, CRF 23, preset medium, GOP 32 (`-g 32 -keyint_min 32`), `-sc_threshold 0`, 2 B-frames, yuv420p, frame rate unchanged |
| Frames | the first 36 frames of the re-encoded clip; centre crop to a multiple of 8 |
| Short clips | 33-frame clips use W = 9 (the paper marks these profiles separately) |

## Why each choice

* **Same re-encoding for both classes.** Real and generated clips have different compression histories;
  without a shared encode, reconstruction statistics pick up codec differences.
* **Fixed GOP, no scene-cut keyframes.** Keeps the codec's own periodic structure (GOP, B-pyramid) at a
  period that cannot be confused with 4 or 8 frames.
* **No upscaling, cap at 720p.** The rhythm is a property of the scale at which the generator's VAE
  decoded. Scanning far above that scale averages the trace away; scanning at the generation scale or
  slightly below keeps it. Typical 2024–26 generators decode at 480–720p.
* **Residual, not raw error.** `r(k)` subtracts the mean error of each absolute frame over all windows that
  contain it, which removes content difficulty and slow drift; window position 0 is excluded because a
  causal VAE encodes the first frame of each window on its own.

## Statistics

* `C4`, `C8`: share of the residual variance in the period-4 / period-8 harmonic combs (K = 24 makes both exact).
* `Ppk`: largest normalised periodogram bin after 4× zero padding, restricted to periods 2.5–9 frames;
  covers 4, 8 and the period of about 5 that appears when a 4-frame pattern at 24 fps is delivered at 30 fps.
* Phase: argument of the period-4 (or period-8) Fourier coefficient; its concentration over many clips is
  `R = |mean exp(i phi)|` (`vae_rhythm.stats.phase_concentration`).

## Zero-shot score and plug-in

* Zero-shot (Table 1): AUROC of `C4` (or `C8` / `Ppk` for the families marked in the paper) between each
  generator's clips and the real clips of the same benchmark; no calibration needed.
* Plug-in (Eq. 3): `s = z(s_d) + max(0, z(Ppk))`; `z(Ppk)` from 299 AIGVDBench validation reals scanned with
  this protocol at the 720p cap; `z(s_d)` hybrid rank standardisation on the detector's scores for the
  3,000 AIGVDBench validation reals. Both calibrations are reused unchanged on every benchmark.

## Reproducibility across machines

* The scan itself (VAE round trips in bf16, error tables, statistics) is deterministic for a given GPU and
  library version; this package reproduces the error tables of the paper's code element for element.
* The protocol re-encode is **not** bit-identical across machines: x264 picks SIMD code for the CPU
  (e.g. AVX-512 vs. AVX2) and sizes its frame threading by the number of cores, and both change a few
  encoding decisions. The paper's preprocessing ran x264 with its defaults on an AVX-512 server. On ten clips
  re-scanned on a second server, half matched the paper's Ppk to three decimals, including the two with a
  strong rhythm; the largest change, 0.12, was on a clip with a weak rhythm. Compare benchmark-level
  numbers, not single clips.
* `--portable` (Python: `scan_video(..., portable=True)`) runs x264 single-threaded without assembly; its
  output is identical on every machine (verified on two CPU generations), at the cost of a slower encode.
* Single-segment scans encode only the first 36 + 64 frames: x264 looks 40 frames ahead, so the 36 scanned
  frames are bit-identical to those of a whole-video encode (verified), while long videos are encoded quickly.

## Cost

One 720p window needs about 10 GB of GPU memory; a 720p scan (24 windows) takes 1–3 minutes on a 24 GB
card and well under a minute on recent data-centre GPUs. Time grows with K × W × pixels. Reducing K below
24 costs accuracy (period-4 and period-8 combs need 24 offsets).
