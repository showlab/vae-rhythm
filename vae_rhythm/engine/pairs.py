"""VAE-pair engine (paper Sec. 4.1, Eq. 2).

For each real clip x the engine emits a pair that differs only by one VAE round trip:

    x+ = c(s_G(x); theta)                 labelled real
    x- = c(D(E(s_G(x))); theta)           labelled generated

s_G rescales to a reconstruction short side G, D(E(.)) is one round trip through a frozen
video VAE drawn from {Wan2.1, HunyuanVideo, CogVideoX} (4x) and LTX-Video (8x), and c is the
post-processing chain of :mod:`vae_rhythm.engine.chain` with the *same* random parameters
for both members. Content, resolution, frame rate, codec and every random choice are shared.

Paper setting (two passes over the 4,000 AIGVDBench training reals, merged; see
``scripts/make_paper_pairs.sh``):
    pass A  G in {256 x4, 360 x2, 480 x2, 720}, S in {480, 720, 1080} with upscaling allowed,
            interpolation 20 %, duplicate-frame conversion 25 %, 4,000 pairs, seed 0
    pass B  G in {256 x3, 360}, interpolation 10 %, duplicate-frame conversion 10 %,
            2,000 pairs, seed 13
Pairs are split 9:1 into train / val by a hash of the source clip id (``split_of``).
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
import zlib

import numpy as np

from ..vaes import VAE, crop_to_multiple, to_tensor, to_uint8
from ..video import FFPROBE, read_frames
from .chain import downscale, run_chain, sample_chain


def split_of(pair_id: str) -> str:
    """Deterministic 9:1 train / val split by source clip."""
    return "val" if zlib.crc32(pair_id.encode()) % 10 == 0 else "train"


def vae_prep(frames: np.ndarray, vae: VAE, max_T: int | None = None) -> np.ndarray:
    """Shared crop / trim: H, W to multiples of 16 and T to the VAE's valid frame count."""
    fr = crop_to_multiple(frames, 16)
    T = vae.valid_T(len(fr) if max_T is None else min(len(fr), max_T))
    return fr[:T]


def vae_pass(frames: np.ndarray, vae: VAE) -> np.ndarray:
    """One round trip (posterior mode). On CUDA OOM, reconstruct the first 41 frames and
    keep the rest unchanged (rare; recorded through ``vae_T`` in the manifest)."""
    import torch

    try:
        y = vae.roundtrip(to_tensor(frames))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        T = vae.valid_T(min(len(frames), 41))
        y = vae.roundtrip(to_tensor(frames[:T]))
        if T < len(frames):
            y = torch.cat([y, to_tensor(frames[T:]).to(y.device)], 2)
    return to_uint8(y)


def _fps(path: str) -> float:
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                          "-of", "json", path], capture_output=True, text=True).stdout
    num, den = json.loads(out)["streams"][0]["r_frame_rate"].split("/")
    return float(num) / float(den)


def make_pairs(real_paths: list[str], out: str, vaes=("wan21", "hunyuan", "cogx", "ltx"), gen_scales=(480,),
               out_scales=(360, 480, 720), allow_upscale: bool = False, interp_prob: float = 0.2,
               dup_prob: float = 0.0, crf_max: int = 28, frames: int = 81, n: int = 0, seed: int = 0,
               tag: str = "", shard: int = 0, nshards: int = 1, device: str = "cuda") -> str:
    """Build pairs for ``real_paths`` into ``out/pos``, ``out/neg`` and a manifest CSV; returns its path."""
    import torch

    torch.set_num_threads(8)
    paths = sorted(real_paths)
    np.random.default_rng(seed).shuffle(paths)
    if n:
        paths = paths[:n]
    # random parameters are fixed per clip index before sharding, so the shard count does not change them
    plan = [(p, np.random.default_rng([seed, i])) for i, p in enumerate(paths)][shard::nshards]
    models = {}
    for v in vaes:
        models[v] = VAE(v, device, "bf16")
        if hasattr(models[v].vae, "enable_tiling"):
            models[v].vae.enable_tiling()
    os.makedirs(os.path.join(out, "pos"), exist_ok=True)
    os.makedirs(os.path.join(out, "neg"), exist_ok=True)
    mname = os.path.join(out, "manifest.csv" if nshards == 1 else f"manifest_shard{shard}.csv")
    rows, nfail, t0 = [], 0, time.time()
    for i, (fpath, rng) in enumerate(plan):
        vid = os.path.splitext(os.path.basename(fpath))[0]
        try:
            fps = _fps(fpath)
            G = int(rng.choice([int(g) for g in gen_scales]))
            p = sample_chain(rng, G, fps, tuple(out_scales), allow_upscale, interp_prob, dup_prob, crf_max)
            k = p["dup_k"]
            raw = read_frames(fpath, num=frames * k)
            while k > 1 and len(raw) < 41 * k:          # source too short: fall back to a smaller factor
                k -= 1
            p["dup_k"] = k
            if k == 1:
                p["dup"] = 0
            if len(raw) < 41:
                raise RuntimeError("too short")
            if k > 1:                                  # duplicate-frame mode: decimate the source first
                raw, fps = raw[::k], fps / k
            H, W = raw.shape[1:3]
            if not allow_upscale:
                p["S"] = min(int(p["S"]), min(H, W))
            vname = str(rng.choice(sorted(models)))
            base = vae_prep(downscale(raw, G), models[vname])
            pos, neg = os.path.join(out, "pos", vid + ".mp4"), os.path.join(out, "neg", vid + ".mp4")
            if not os.path.exists(pos):
                run_chain(base, fps, p, pos)
            if not os.path.exists(neg):
                run_chain(vae_pass(base, models[vname]), fps, p, neg)
            common = dict(dataset="synth", pair_id=vid, split=split_of(vid), native_short=min(H, W), fps=round(fps, 3),
                          vae=vname, G=G, vae_T=len(base), **p)
            rows.append(dict(subset="pos" + tag, label=0, video_id=f"pos{tag}/{vid}", path=pos, **common))
            rows.append(dict(subset="neg" + tag, label=1, video_id=f"neg{tag}/{vid}", path=neg, **common))
        except Exception as e:  # noqa: BLE001
            nfail += 1
            print("FAIL", fpath, str(e)[:160], flush=True)
        if (i + 1) % 50 == 0 or i + 1 == len(plan):
            _dump(rows, mname)
            print(f"{i + 1}/{len(plan)} clips, {len(rows) // 2} pairs, {nfail} failed, {time.time() - t0:.0f}s", flush=True)
    return mname


def _dump(rows: list[dict], path: str) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
