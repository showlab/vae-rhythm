# AGENTS.md — setup guide for coding agents

You are a coding agent asked to set up and use this repository. Follow the steps in order. Every step
has a **check**; do not continue past a failed check — fix it using the notes under that step.
Human-oriented documentation is in `README.md`; protocol details in `docs/protocol.md`; full paper
reproduction in `docs/reproduce.md`.

## 0. Decide what the user needs

| User goal | Steps |
|---|---|
| Score a few videos with the training-free phase scan | 1 → 2 → 3 → 4 |
| Add the VAE Rhythm plug-in to their own detector | 1 → 2 → 3 → 4 → 6 |
| Run the pretrained CaDet detector | 1 → 2 → 3 → 5 |
| Reproduce paper tables / train CaDet | all steps, then `docs/reproduce.md` |

Ask the user only for things you cannot determine: where their videos are, whether a GPU machine is
available, and (for CaDet) whether they have accepted the DINOv3 license on Hugging Face.

## 1. System prerequisites

* Python ≥ 3.10.
* **ffmpeg ≥ 4.4 with libx264** (libx265 and the `minterpolate` filter are needed only for the VAE-pair engine).
* A CUDA GPU is strongly recommended: one 720p scan (24 offsets × 13 frames) takes about 1–3 minutes on a
  24 GB card; CPU works but is 50–100× slower. Apple-silicon `mps` is untested.

**Check**: `ffmpeg -hide_banner -encoders | grep -c libx264` prints `1` or more; `nvidia-smi` lists a GPU.

If ffmpeg is missing or lacks libx264: `conda install -c conda-forge ffmpeg` (inside the environment of
step 2), or download a static build and set `VAE_RHYTHM_FFMPEG=/path/to/ffmpeg VAE_RHYTHM_FFPROBE=/path/to/ffprobe`.
Do not use `apt` on shared servers without asking the user.

## 2. Python environment

```bash
conda create -n vae-rhythm python=3.10 -y && conda activate vae-rhythm      # or: python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# Install a CUDA build of PyTorch that matches the driver first (see https://pytorch.org), e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[cadet,dev]"        # use plain "-e ." if CaDet is not needed
```

**Check**: `vae-rhythm doctor` prints only `OK` lines for the required items (lines starting with
`(optional)` matter only for CaDet). It exits with status 0.

Notes:
* `diffusers >= 0.33` is required (`AutoencoderKLWan`). Tested with Python 3.10, torch 2.14 (CUDA 12.6),
  diffusers 0.40, timm 1.0.29, peft 0.21, PyAV 17, ffmpeg 4.4 and 7.0.
* If `huggingface_hub` and `transformers` conflict after installation, upgrade both together
  (`pip install -U transformers huggingface_hub`).
* Behind a firewall, set `HF_ENDPOINT` to a mirror, or download the models elsewhere and pass local
  folders (`--vae-path /path/to/Wan2.1-T2V-1.3B-Diffusers`).

## 3. Unit tests (CPU, seconds)

```bash
pytest -q
```

**Check**: all tests pass (the ffmpeg/PyAV test is skipped if PyAV is missing — it must not be skipped
after step 2).

## 4. GPU self-test of the phase scan (no dataset needed)

```bash
vae-rhythm selftest
```

It synthesises a 3-second moving test pattern with ffmpeg, makes a copy that went through one Wan2.1 VAE
round trip, and scans both (downloads the Wan2.1 VAE, ≈ 0.5 GB, on first use).

**Check**: the last line is `SELFTEST PASSED`, and the round-trip copy has the higher `Ppk` and `C4`
with a period of 4 frames. Example output:

```
source          Ppk 0.139  period 4.17  C4 0.175  z(Ppk) -1.15
VAE round trip  Ppk 0.360  period 4.00  C4 0.956  z(Ppk) +0.43
SELFTEST PASSED
```

(values from an RTX A5000; small differences across GPUs and library versions are normal — the check is the
ordering, with the round-trip copy's C4 near 1).

If it fails with CUDA out-of-memory: free the GPU or use a larger card (one 720p window needs ~10 GB).
If the VAE download fails: see the Hugging Face notes in step 2.

Then scan the user's video(s):

```bash
vae-rhythm scan path/to/video.mp4 --out scan_out/
```

Report `Ppk`, its period, `z(Ppk)` and the real-video percentile, and show `scan_out/<video>.png`.
**Interpretation rules you must pass on to the user**: a high `z(Ppk)` (above ≈ 2.35, the 95th
percentile of real videos) is evidence of VAE decoding; a low value is *not* evidence that a video is
real (low motion, frame-rate conversion, upscaling and re-encoding weaken the rhythm). Never present the
scan alone as a definitive real/fake verdict.

## 5. Pretrained CaDet (optional)

1. The user must accept the DINOv3 license for `facebook/dinov3-vitl16-pretrain-lvd1689m` on Hugging Face;
   then `huggingface-cli login`.
2. Download the adapters (a few MB per seed). The Hugging Face release is coming soon; until then ask the user where the weights are. Then run:

```bash
huggingface-cli download <org>/CaDet --local-dir checkpoints/cadet
vae-rhythm predict path/to/video.mp4 --weights checkpoints/cadet/seed0
```

**Check**: `predict` prints a score without errors such as `checkpoint lacks trained parameters`. The score is
a sum of z-scores relative to real validation videos (≈ 0 for typical real videos; larger = more likely generated).

## 6. Plug-in with an existing detector (optional)

Inputs: the detector's scores on the test videos (`video_id,score`), its scores on real calibration videos
from a similar source, and `Ppk` of the test videos from `vae-rhythm scan-manifest`. Then:

```bash
vae-rhythm scan-manifest test_manifest.csv --out ppk.csv --npz-dir npz/
vae-rhythm plugin --scores det_test.csv --det-real det_real_cal.csv --ppk ppk.csv --manifest test_manifest.csv --out fused.csv
```

Manifest format: `docs/data.md`. The bundled Ppk calibration matches only the default protocol
(Wan2.1, K=24, W=13, 720p cap); if you change any of these, recalibrate on real videos.

## Rules for agents

* Do not change the scan protocol constants (VAE, K, W, cap, x264 CRF 23 / GOP 32, bf16) when reporting
  numbers that are compared with the paper or with the bundled calibration.
* Never upscale videos before scanning; the cap only downscales.
* Per-clip values can differ slightly between machines (x264 SIMD / threads, see `docs/protocol.md`); use
  `--portable` when the user needs identical numbers across machines.
* Manifest scans (with `--npz-dir`) and pair generation are resumable: re-running the same command skips
  finished items. Training is not resumable; run it under `nohup`/`tmux`.
* Datasets are not redistributed here; see `docs/data.md` for sources and licences.
