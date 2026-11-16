"""Post-processing chain shared by both members of a VAE pair (paper Sec. 4.1, supplement).

The chain imitates how videos reach a viewer: leading-frame trimming, rescaling to a
delivery short side S, optional frame-rate conversion (motion-compensated interpolation,
duplicate-frame conversion or decimation to 16 fps) and x264 / x265 compression with
random CRF, B-frames, B-pyramid and GOP. Randomising the codec puts its own periodic
structure (e.g. a 4-frame B-pyramid) equally often in positives and negatives.
"""
from __future__ import annotations

import subprocess

import numpy as np

from ..video import FFMPEG

SCALE_EXPR = "scale='if(gt(iw,ih),-2,{S})':'if(gt(iw,ih),{S},-2)':flags=area"


def sample_chain(rng: np.random.Generator, native_short: int, fps: float, out_scales=(360, 480, 720),
                 allow_upscale: bool = False, interp_prob: float = 0.2, dup_prob: float = 0.0, crf_max: int = 28) -> dict:
    """Random chain parameters. ``allow_upscale`` lets S exceed the generation scale (closed
    platforms often generate at 480-720p, upscale to 1080p, then re-encode)."""
    cand = [s for s in out_scales if allow_upscale or s <= native_short] or [native_short]
    S = rng.choice(cand)
    codec = rng.choice(["x264", "x265"])
    u = rng.random()
    interp = float(rng.choice([1.25, 1.5, 2.0])) if u < interp_prob else 0.0
    dup = int(rng.choice([24, 30])) if interp_prob <= u < interp_prob + dup_prob else 0   # exclusive with interpolation
    return dict(S=int(S), codec=codec,
                crf=int(rng.integers(18, crf_max + 1) if codec == "x264" else rng.integers(20, crf_max + 1)),
                bframes=int(rng.integers(0, 9)), pyramid=int(rng.integers(0, 3)), gop=int(rng.integers(12, 251)),
                interp=interp, dup=dup, dup_k=int(rng.choice([2, 3])) if dup else 1,
                drop16=int(rng.random() < 0.3 and fps > 17 and not dup), trim=int(rng.integers(0, 4)))


def run_chain(frames: np.ndarray, fps: float, p: dict, dst: str) -> None:
    """Apply chain ``p`` to uint8 frames (T, H, W, 3) at their native frame rate and encode to ``dst``."""
    T, H, W, _ = frames.shape
    vf = [f"select='gte(n,{p['trim']})'", "setpts=N/FRAME_RATE/TB"]
    if p.get("dup"):
        vf.append(f"fps={p['dup']}")              # duplicate-frame conversion back to 24 / 30 fps
    vf.append(SCALE_EXPR.format(S=p["S"]))
    if p["interp"]:
        vf.append(f"minterpolate=fps={fps * p['interp']:g}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")
    if p["drop16"]:
        vf.append("fps=16")
    if p["codec"] == "x264":
        enc = ["-c:v", "libx264", "-preset", "medium", "-crf", str(p["crf"]), "-bf", str(p["bframes"]),
               "-b-pyramid", ["none", "strict", "normal"][p["pyramid"]], "-g", str(p["gop"]),
               "-keyint_min", str(min(p["gop"], 25))]
    else:
        enc = ["-c:v", "libx265", "-preset", "medium", "-crf", str(p["crf"]),
               "-x265-params", f"bframes={p['bframes']}:b-pyramid={min(p['pyramid'], 1)}:keyint={p['gop']}:log-level=error"]
    cmd = [FFMPEG, "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", f"{fps:g}", "-i", "-", "-an", "-vf", ",".join(vf), *enc, "-pix_fmt", "yuv420p", dst]
    r = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[-400:])


def downscale(frames: np.ndarray, short: int) -> np.ndarray:
    """Area-downscale uint8 frames to a short side of ``short`` (even sizes; never upscale)."""
    T, H, W, _ = frames.shape
    if min(H, W) <= short:
        return frames
    if W >= H:
        nh, nw = short, int(round(W * short / H))
    else:
        nw, nh = short, int(round(H * short / W))
    nw -= nw % 2
    nh -= nh % 2
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-i", "-",
           "-vf", f"scale={nw}:{nh}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    r = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[-300:])
    return np.frombuffer(r.stdout, np.uint8).reshape(T, nh, nw, 3)
