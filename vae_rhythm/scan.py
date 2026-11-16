"""Phase scan (paper Sec. 3.1).

For each start offset k = 0..K-1, a W-frame window [k, k + W) is reconstructed by a frozen
causal video VAE (Wan2.1 by default) and the per-frame RGB MSE e(k, j) is recorded. When the
scan lattice and the source's frame groups line up, reconstruction is easier, so generated
videos give an error profile that repeats with the group period while real videos give a
nearly flat one. The defaults K = 24 and W = 13 need 36 consecutive frames.

    from vae_rhythm import scan_video
    res = scan_video("clip.mp4")             # Wan2.1, native resolution capped at 720p
    print(res.ppk, res.z_ppk, res.stats["C4"])
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from statistics import median
from typing import Sequence

import numpy as np

from . import stats as S
from .vaes import VAE, crop_to_multiple, to_tensor
from .video import PreparedVideo, count_frames, motion_energy, read_frames

K_DEFAULT, W_DEFAULT = 24, 13
DEFAULT_CALIBRATION = "wan21_k24w13_cap720.json"


def scan_frames(vae: VAE, frames: np.ndarray, K: int = K_DEFAULT, W: int = W_DEFAULT, batch: int = 1) -> np.ndarray:
    """Error table e(k, j), shape (K, W), for uint8 frames (T, H, W, 3) with T >= K - 1 + W.

    Frames must already be cropped to a multiple of the VAE's spatial factor. Windows are
    sliced on the GPU and ``batch`` of them are reconstructed together (a 720p window needs
    about 10 GB; batch 2 suits 40 GB+ cards). A batched CUDA OOM is retried with batch 1.
    """
    import torch

    try:
        return _scan_frames(vae, frames, K, W, batch)
    except torch.cuda.OutOfMemoryError:
        if batch == 1:
            raise
        vae.clear_cache()
        torch.cuda.empty_cache()
        return _scan_frames(vae, frames, K, W, 1)


def _scan_frames(vae, frames, K, W, batch):
    import torch

    if len(frames) < K - 1 + W:
        raise ValueError(f"Need {K - 1 + W} frames, got {len(frames)}")
    if W != vae.valid_T(W):
        raise ValueError(f"W={W} is not a valid window for {vae.name} (1 + {vae.t_step}n)")
    x = to_tensor(frames[:K - 1 + W]).to(vae.device)
    errors = []
    vae.clear_cache()
    for i in range(0, K, batch):
        xb = torch.cat([x[:, :, k:k + W] for k in range(i, min(i + batch, K))], 0)
        yb = vae.roundtrip(xb)
        errors.append(((yb - xb.float()) ** 2).mean(dim=(1, 3, 4)).cpu())
    vae.clear_cache()
    return torch.cat(errors).numpy().astype(np.float64)


def plan_segments(total_frames: int, n_max: int = 3, length: int = K_DEFAULT + W_DEFAULT - 1,
                  align: int = 8) -> list[int]:
    """Start frames of up to ``n_max`` disjoint segments (start / middle / end).

    Starts are multiples of ``align`` (8 covers both 4x and 8x VAEs), so a generated
    video's frame groups sit at the same phase in every segment.
    """
    if total_frames < length:
        raise ValueError(f"The video needs at least {length} frames; it has {total_frames}.")
    span = total_frames - length
    last = (span // align) * align
    min_gap = math.ceil(length / align) * align
    count = max(1, min(n_max, last // min_gap + 1))
    if count == 1:
        return [0]
    if count == 2:
        return [0, last]
    return [0, ((last // align) // 2) * align, last]


def load_calibration(name_or_path: str = DEFAULT_CALIBRATION) -> dict:
    """Real-video calibration of Ppk (mean / std / reference scores) for z(Ppk)."""
    p = Path(name_or_path)
    if p.is_file():
        return json.loads(p.read_text())
    return json.loads(resources.files("vae_rhythm.calibration").joinpath(name_or_path).read_text())


@dataclass
class SegmentResult:
    start: int
    errors: np.ndarray                     # (K, W) error table
    stats: dict
    curve: list[float]                     # residual profile r(k)
    motion: float

    @property
    def ppk(self) -> float:
        return float(self.stats["Ppk"])


@dataclass
class ScanResult:
    path: str
    vae: str
    K: int
    W: int
    short_side_cap: int
    scan_size: tuple[int, int]             # (width, height) actually scanned
    segments: list[SegmentResult]
    calibration: dict | None = None
    z_ppk: float | None = None             # median over segments of (Ppk - mu) / sd on real videos
    real_percentile: float | None = None   # share of calibration real videos with a lower Ppk
    extra: dict = field(default_factory=dict)

    @property
    def ppk(self) -> float:
        return float(median(s.ppk for s in self.segments))

    @property
    def stats(self) -> dict:
        """Statistics of the first segment (the paper's single-segment protocol)."""
        return self.segments[0].stats

    def to_dict(self) -> dict:
        return {
            "path": self.path, "vae": self.vae, "K": self.K, "W": self.W, "short_side_cap": self.short_side_cap,
            "scan_size": list(self.scan_size), "Ppk_median": self.ppk, "z_ppk": self.z_ppk,
            "real_percentile": self.real_percentile,
            "segments": [{"start": s.start, "motion": s.motion, "curve": s.curve,
                          "stats": {k: v for k, v in s.stats.items() if isinstance(v, (int, float))}}
                         for s in self.segments],
            **self.extra,
        }


def scan_video(path: str, vae: VAE | str = "wan21", short_side_cap: int = 720, K: int = K_DEFAULT,
               W: int = W_DEFAULT, segments: int = 1, batch: int = 1, calibration: str | dict | None = "default",
               prepared_copy: str | None = None, device: str = "cuda", dtype: str = "bf16",
               portable: bool = False) -> ScanResult:
    """Scan one video with the paper's protocol.

    1. Re-encode the whole video with the protocol (area downscale only when the short side
       exceeds ``short_side_cap``; x264 CRF 23, GOP 32) -- :func:`vae_rhythm.video.prepare`.
    2. Read K - 1 + W consecutive frames per segment. ``segments=1`` takes the first frames
       (the protocol behind every number in the paper); ``segments=3`` adds the middle and
       the end of longer videos (starts aligned to multiples of 8) and reports the median.
    3. Centre-crop to the VAE's spatial factor and run :func:`scan_frames`.

    ``calibration`` gives z(Ppk); the bundled default was measured with Wan2.1, K=24, W=13
    and a 720p cap on real videos and is only applied when the scan matches that protocol.
    ``portable=True`` makes the re-encode bit-identical across machines (see :func:`vae_rhythm.video.prepare`).
    """
    if isinstance(vae, str):
        vae = VAE(vae, device=device, dtype=dtype)
    need = K - 1 + W
    seg_results = []
    from .video import LOOKAHEAD_MARGIN

    prefix = need + LOOKAHEAD_MARGIN if segments == 1 else None      # multi-segment scans encode the whole video
    with PreparedVideo(path, short_side_cap, keep=prepared_copy, max_frames=prefix, portable=portable) as prepared:
        total = count_frames(prepared) if segments > 1 else len(read_frames(prepared, num=need))
        if total < need:
            # Short clips (e.g. 33 frames): shrink the window first, as in the paper (W >= 9).
            W2 = vae.valid_T(total - K + 1)
            if W2 < 9:
                raise ValueError(f"{path}: only {total} frames; the scan needs at least {K + 8}.")
            W, need = W2, K - 1 + W2
        starts = plan_segments(total, segments, need) if segments > 1 else [0]
        for st in starts:
            fr = crop_to_multiple(read_frames(prepared, num=need, start=st), vae.s_factor)
            E = scan_frames(vae, fr, K, W, batch)
            seg_results.append(SegmentResult(st, E, S.phase_stats(E), S.residual_curve(E).round(6).tolist(),
                                             motion_energy(fr[:W])))
        scan_size = (int(fr.shape[2]), int(fr.shape[1]))
    res = ScanResult(str(path), vae.name, K, W, short_side_cap, scan_size, seg_results)
    if calibration is not None:
        cal = load_calibration() if calibration == "default" else (
            load_calibration(calibration) if isinstance(calibration, str) else calibration)
        proto = cal.get("protocol", {})
        if (proto.get("vae"), proto.get("K"), proto.get("W"), proto.get("short_side_cap")) == (vae.name, K, W, short_side_cap):
            res.calibration = cal
            res.z_ppk = float(median((s.ppk - cal["ppk_mean"]) / cal["ppk_std"] for s in seg_results))
            ref = np.sort(np.asarray(cal["ppk_real_scores"], dtype=np.float64))
            v = res.ppk
            res.real_percentile = float((np.searchsorted(ref, v, "left") + np.searchsorted(ref, v, "right")) / 2 / len(ref))
        else:
            res.extra["calibration_note"] = "no matching real-video calibration for this protocol; z(Ppk) not reported"
    return res


def plot_result(res: ScanResult, out: str) -> None:
    """Residual profile r(k) per segment and the spectrum of the first segment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for s in res.segments:
        ax[0].plot(np.arange(res.K), np.asarray(s.curve) * 100, marker="o", ms=3, label=f"frames {s.start}-{s.start + res.K + res.W - 2}")
    ax[0].axhline(0, color="0.6", lw=0.8)
    for k in range(0, res.K, 4):
        ax[0].axvline(k, color="0.85", lw=0.6, zorder=0)
    ax[0].set_xlabel("start offset k")
    ax[0].set_ylabel("residual r(k) (%)")
    ax[0].legend(fontsize=7)
    sp = S.spectrum(res.segments[0].curve)
    ax[1].plot(sp["periods"], sp["power"])
    ax[1].set_xlabel("period (frames)")
    ax[1].set_ylabel("normalised power")
    title = f"Ppk = {res.ppk:.3f}"
    if res.z_ppk is not None:
        title += f", z = {res.z_ppk:.2f}, real percentile = {100 * res.real_percentile:.0f}%"
    fig.suptitle(f"{Path(res.path).name}: {title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def scan_manifest(rows: Sequence[dict], vae: VAE, out_csv: str, npz_dir: str | None = None,
                  short_side_cap: int = 720, K: int = K_DEFAULT, W: int = W_DEFAULT, batch: int = 1,
                  portable: bool = False) -> None:
    """Scan every row of a manifest (columns: video_id, path, ...) into one CSV of statistics.

    Already-scanned videos (an existing NPZ) are not scanned again, so runs can resume.
    """
    import csv
    import os

    if npz_dir:
        os.makedirs(npz_dir, exist_ok=True)
    out_rows = []
    for r in rows:
        npz = os.path.join(npz_dir, r["video_id"].replace("/", "__") + ".npz") if npz_dir else None
        try:
            if npz and os.path.exists(npz):
                d = np.load(npz)
                E, mot = d["E"], float(d["motion"])
            else:
                res = scan_video(r["path"], vae, short_side_cap, K, W, segments=1, batch=batch, calibration=None,
                                 portable=portable)
                seg = res.segments[0]
                E, mot = seg.errors, seg.motion
                if npz:
                    np.savez_compressed(npz, E=E.astype(np.float32), motion=mot, vae=vae.name, K=E.shape[0], W=E.shape[1],
                                        short_side_cap=short_side_cap)
            s = S.phase_stats(E)
            out_rows.append({**r, "vae": vae.name, "K": E.shape[0], "W": E.shape[1], "motion": round(mot, 3),
                             **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in s.items()}})
        except Exception as e:  # noqa: BLE001 - keep going, record the failure
            out_rows.append({**r, "vae": vae.name, "error": str(e)[:200]})
    keys = list(dict.fromkeys(k for r in out_rows for k in r))
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out_rows)
