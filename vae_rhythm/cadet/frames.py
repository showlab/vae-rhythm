"""Frame sampling shared by both CaDet paths (AIGVDBench protocol).

Cache: decode the first 31 frames, keep frames 0, 2, ..., 30 (16 frames), centre-crop the
short side to a square, resize to 256 x 256 (bicubic) and store as JPEG (quality 95).
Training and testing use the even cache slots, i.e. original frames 0, 4, ..., 28 (8 frames).
Clips with fewer than 31 frames spread the decoded frames uniformly over the 16 slots.

Layout: ``<root>/<dataset>/<video_id with "/" replaced by "__">/<slot>.jpg``, where ``dataset``
is the manifest column of that name (training mixes benchmark clips and VAE pairs).
Use :func:`cache_manifest` once per manifest; :func:`clip_frames` produces the same 16
frames in memory for single-video inference (including the JPEG round trip).
"""
from __future__ import annotations

import csv
import io
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

FRAME_IDX = [0, 2, 4, 6, 8, 10, 12, 14]          # cache slots = original frames 0, 4, ..., 28
N_SLOTS, SIZE, JPEG_Q = 16, 256, 95
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def key_of(video_id: str) -> str:
    return video_id.replace("/", "__")


def _slots(frames: np.ndarray) -> list[np.ndarray]:
    from PIL import Image

    fr = frames[::2][:N_SLOTS]
    if len(fr) < N_SLOTS:
        fr = fr[np.round(np.linspace(0, len(fr) - 1, N_SLOTS)).astype(int)]
    out = []
    for f in fr:
        H, W = f.shape[:2]
        s = min(H, W)
        y0, x0 = (H - s) // 2, (W - s) // 2
        out.append(Image.fromarray(f[y0:y0 + s, x0:x0 + s]).resize((SIZE, SIZE), Image.BICUBIC))
    return out


def clip_frames(path: str) -> np.ndarray:
    """The 16 cached frames of one video, in memory: uint8 (16, 256, 256, 3)."""
    from PIL import Image

    from ..video import read_frames

    imgs = []
    for im in _slots(read_frames(path, num=31)):
        b = io.BytesIO()
        im.save(b, "JPEG", quality=JPEG_Q)
        b.seek(0)
        imgs.append(np.asarray(Image.open(b).convert("RGB")))
    return np.stack(imgs)


def _cache_one(args):
    vid, path, out = args
    d = os.path.join(out, key_of(vid))
    if os.path.exists(os.path.join(d, f"{N_SLOTS - 1}.jpg")):
        return vid, "skip"
    try:
        from ..video import read_frames

        imgs = _slots(read_frames(path, num=31))
        os.makedirs(d, exist_ok=True)
        for k, im in enumerate(imgs):
            im.save(os.path.join(d, f"{k}.jpg"), quality=JPEG_Q)
        return vid, "ok"
    except Exception as e:  # noqa: BLE001
        return vid, f"fail: {e}"[:200]


def cache_manifest(manifest: str, root: str, workers: int = 16) -> None:
    """Cache every row (dataset, video_id, path) of ``manifest`` under ``root/<dataset>/``."""
    with open(manifest) as fh:
        rows = list(csv.DictReader(fh))
    jobs = [(r["video_id"], r["path"], os.path.join(root, r.get("dataset") or "default")) for r in rows]
    n_fail = 0
    with ProcessPoolExecutor(workers) as ex:
        for i, (vid, status) in enumerate(ex.map(_cache_one, jobs, chunksize=8)):
            if status.startswith("fail"):
                n_fail += 1
                print(vid, status, flush=True)
            if (i + 1) % 2000 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    print(f"cached {len(jobs) - n_fail}/{len(jobs)} videos -> {root}", flush=True)


def cache_dir(root: str, row: dict) -> str:
    return os.path.join(root, row.get("dataset") or "default", key_of(row["video_id"]))


def load_cached(root: str, row: dict, idx=FRAME_IDX) -> np.ndarray:
    from PIL import Image

    d = cache_dir(root, row)
    return np.stack([np.asarray(Image.open(os.path.join(d, f"{k}.jpg")).convert("RGB")) for k in idx])


def has_cache(root: str, row: dict) -> bool:
    return os.path.exists(os.path.join(cache_dir(root, row), f"{N_SLOTS - 1}.jpg"))
