"""Video input/output and the scan protocol's preprocessing.

Protocol (paper Sec. 3.1 / 4.3): every clip, real or generated, is re-encoded with the
same command before scanning -- libx264, CRF 23, a fixed GOP of 32 without scene-cut
keyframes, two B-frames, no frame-rate change -- after an area downscale that is applied
only when the short side exceeds the scan cap (480 for the census, 720 for the plug-in).
Equal compression history for both classes keeps codec differences from masquerading
as rhythm, and the fixed GOP keeps the codec's own period away from the VAE's.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

FFMPEG = os.getenv("VAE_RHYTHM_FFMPEG", "ffmpeg")
FFPROBE = os.getenv("VAE_RHYTHM_FFPROBE", "ffprobe")
X264_ARGS = ["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p", "-bf", "2", "-sc_threshold", "0"]
CRF, GOP = 23, 32

_PASSTHROUGH: list[str] | None = None


def passthrough_args() -> list[str]:
    """Keep every decoded frame: ``-fps_mode passthrough`` (ffmpeg >= 5.1) or ``-vsync passthrough``."""
    global _PASSTHROUGH
    if _PASSTHROUGH is None:
        modern = True
        try:
            first = subprocess.run([FFMPEG, "-hide_banner", "-version"], capture_output=True, timeout=20,
                                   check=False).stdout.decode(errors="ignore").split("\n", 1)[0].split()
            version = first[2] if len(first) > 2 and first[1] == "version" else ""
            digits = version.lstrip("nN").split("-")[0].split(".")
            if digits and digits[0].isdigit():
                major = int(digits[0])
                minor = int(digits[1]) if len(digits) > 1 and digits[1].isdigit() else 0
                modern = (major, minor) >= (5, 1)
        except (OSError, subprocess.TimeoutExpired):
            pass
        _PASSTHROUGH = ["-fps_mode", "passthrough"] if modern else ["-vsync", "passthrough"]
    return list(_PASSTHROUGH)


def check_ffmpeg() -> None:
    for tool in (FFMPEG, FFPROBE):
        if shutil.which(tool) is None and not Path(tool).is_file():
            raise RuntimeError(f"{tool} not found. Install ffmpeg (e.g. `conda install -c conda-forge ffmpeg`) "
                               "or point VAE_RHYTHM_FFMPEG / VAE_RHYTHM_FFPROBE to the binaries.")


def probe(path: str | os.PathLike) -> dict:
    """Width, height, frame rate and duration of the first video stream."""
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames:format=duration",
                          "-of", "json", str(path)], capture_output=True, check=True).stdout
    data = json.loads(out)
    st = data["streams"][0]
    fps = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            num, den = str(st.get(key)).split("/")
            if float(den) > 0 and float(num) > 0:
                fps = float(num) / float(den)
                break
        except (ValueError, ZeroDivisionError):
            continue
    nb = st.get("nb_frames")
    return {"width": int(st["width"]), "height": int(st["height"]), "fps": fps,
            "nb_frames": int(nb) if str(nb).isdigit() else None,
            "duration": float(data.get("format", {}).get("duration") or 0) or None}


def count_frames(path: str | os.PathLike) -> int:
    """Number of video packets (= frames for the protocol-encoded files this package writes)."""
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, check=True)
    return int(out.stdout.decode().strip().split(",")[0])


def target_size(w: int, h: int, short_side: int) -> tuple[int, int]:
    """Downscale so that the short side equals ``short_side`` (never upscale); even sizes."""
    if short_side <= 0 or min(w, h) <= short_side:
        return w - w % 2, h - h % 2
    if w >= h:
        nh, nw = short_side, int(round(w * short_side / h))
    else:
        nw, nh = short_side, int(round(h * short_side / w))
    return nw - nw % 2, nh - nh % 2


# x264's rate control looks 40 frames ahead at preset medium, so encoding the first N + 40 frames gives the
# same first N frames as encoding the whole video (checked bit for bit); 64 leaves a margin.
LOOKAHEAD_MARGIN = 64


def prepare(src: str | os.PathLike, dst: str | os.PathLike, short_side: int = 720,
            crf: int = CRF, gop: int = GOP, max_frames: int | None = None, portable: bool = False) -> Path:
    """Re-encode ``src`` with the protocol (area downscale only above ``short_side``,
    x264 CRF 23, GOP 32, no scene-cut keyframes, frame rate unchanged).

    ``max_frames`` encodes only a prefix (pass the frames you will read + LOOKAHEAD_MARGIN).
    ``portable`` runs x264 single-threaded without assembly: identical output on every machine.
    By default x264 uses its SIMD code and all cores, as in the paper; its output then differs
    slightly between CPU types and thread counts, which can move Ppk of clips with a weak rhythm."""
    info = probe(src)
    w, h = target_size(info["width"], info["height"], short_side)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.mp4")
    extra = ["-frames:v", str(int(max_frames))] if max_frames else []
    if portable:
        extra += ["-threads", "1", "-x264-params", "asm=0"]
    cmd = [FFMPEG, "-nostdin", "-y", "-v", "error", "-i", str(src), "-an", *passthrough_args(),
           "-vf", f"scale={w}:{h}:flags=area", *X264_ARGS, "-crf", str(crf), "-g", str(gop),
           "-keyint_min", str(gop), *extra, str(tmp)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {src}: {p.stderr[-500:]}")
    os.replace(tmp, dst)
    return dst


def read_frames(path: str | os.PathLike, num: int | None = None, start: int = 0, stride: int = 1) -> np.ndarray:
    """Decode uint8 RGB frames (T, H, W, 3) with PyAV; ``num=None`` reads to the end."""
    import av

    out = []
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        st.thread_count = 4
        i = 0
        for fr in c.decode(st):
            if i >= start and (i - start) % stride == 0:
                out.append(fr.to_ndarray(format="rgb24"))
                if num is not None and len(out) >= num:
                    break
            i += 1
    if not out:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(out)


def write_frames(frames: np.ndarray, path: str | os.PathLike, fps: float, crf: int = CRF, gop: int = GOP) -> None:
    """Encode uint8 frames (T, H, W, 3) with the protocol's x264 settings (H, W even)."""
    T, H, W, _ = frames.shape
    if H % 2 or W % 2:
        raise ValueError(f"Frame size must be even for yuv420p, got {W}x{H}")
    cmd = [FFMPEG, "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", f"{fps:g}", "-i", "-", "-an", *passthrough_args(), *X264_ARGS, "-crf", str(crf), "-g", str(gop),
           "-keyint_min", str(gop), str(path)]
    p = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-2000:])


def motion_energy(frames: np.ndarray) -> float:
    """Mean absolute frame difference (uint8 scale); the rhythm is unreliable below ~2."""
    f = frames.astype(np.float32)
    return float(np.abs(f[1:] - f[:-1]).mean()) if len(f) > 1 else 0.0


class PreparedVideo:
    """Context manager: protocol-encoded copy of a video in a temporary directory."""

    def __init__(self, src: str | os.PathLike, short_side: int = 720, keep: str | os.PathLike | None = None,
                 max_frames: int | None = None, portable: bool = False):
        self.src, self.short_side, self.keep = src, short_side, keep
        self.max_frames, self.portable = max_frames, portable
        self._tmp = None

    def __enter__(self) -> Path:
        kw = dict(max_frames=self.max_frames, portable=self.portable)
        if self.keep:
            return prepare(self.src, self.keep, self.short_side, **kw)
        self._tmp = tempfile.TemporaryDirectory(prefix="vae_rhythm_")
        return prepare(self.src, Path(self._tmp.name) / "prepared.mp4", self.short_side, **kw)

    def __exit__(self, *exc) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
