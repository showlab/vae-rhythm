"""Command line interface: ``vae-rhythm <command> ...`` (or ``python -m vae_rhythm ...``).

Commands
  doctor          check ffmpeg, Python packages and the GPU
  selftest        GPU check without data (synthetic clip vs. its Wan2.1 round trip)
  scan            phase scan of one or more videos (zero-shot rhythm score, curve plot)
  scan-manifest   phase scan of every video in a manifest CSV -> statistics CSV (+ NPZ tables)
  make-pairs      VAE-pair engine: real videos -> (real, VAE round trip) pairs
  cache-frames    16-frame JPEG cache used by CaDet training and evaluation
  train-artifact  train the artifact path of CaDet (benchmark training set + VAE pairs)
  train-semantic  train the semantic path of CaDet (benchmark training set only)
  calibrate       fusion statistics of a trained CaDet on validation real videos
  predict         CaDet scores for videos or for a manifest
  plugin          VAE Rhythm plug-in (Eq. 3) on detector scores + Ppk, with AUROC
  evaluate        per-generator AUROC table of a score CSV against a manifest
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys


def _rows(spec: str) -> list[dict]:
    path, _, split = spec.partition(":")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if not split or r.get("split", "") == split]


def _write_csv(path: str, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def cmd_scan(a):
    from .scan import plot_result, scan_video
    from .vaes import VAE

    vae = VAE(a.vae, a.device, a.dtype, source=a.vae_path)
    for path in a.videos:
        res = scan_video(path, vae, a.cap, a.K, a.W, a.segments, a.batch, portable=a.portable)
        d = res.to_dict()
        line = f"{path}: Ppk {res.ppk:.4f} (period {res.stats['ppk']:.2f}), C4 {res.stats['C4']:.4f}, C8 {res.stats['C8']:.4f}"
        if res.z_ppk is not None:
            line += f", z(Ppk) {res.z_ppk:+.2f}, above {100 * res.real_percentile:.0f}% of reference real videos"
        print(line, flush=True)
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            stem = os.path.splitext(os.path.basename(path))[0]
            json.dump(d, open(os.path.join(a.out, stem + ".json"), "w"), indent=1)
            plot_result(res, os.path.join(a.out, stem + ".png"))


def cmd_scan_manifest(a):
    from .scan import scan_manifest
    from .vaes import VAE

    rows = _rows(a.manifest)[a.shard::a.nshards]
    scan_manifest(rows, VAE(a.vae, a.device, a.dtype, source=a.vae_path), a.out, a.npz_dir, a.cap, a.K, a.W, a.batch,
                  portable=a.portable)


def cmd_make_pairs(a):
    from .engine.pairs import make_pairs

    if a.reals_manifest:
        paths = [r["path"] for r in _rows(a.reals_manifest) if int(r.get("label", 0)) == 0]
    else:
        paths = [os.path.join(a.reals, f) for f in os.listdir(a.reals) if f.lower().endswith((".mp4", ".mov", ".mkv", ".webm"))]
    make_pairs(paths, a.out, a.vaes.split(","), [int(x) for x in a.gen_scales.split(",")],
               [int(x) for x in a.out_scales.split(",")], a.allow_upscale, a.interp_prob, a.dup_prob, a.crf_max,
               a.frames, a.n, a.seed, a.tag, a.shard, a.nshards, a.device)


def cmd_cache_frames(a):
    from .cadet.frames import cache_manifest

    cache_manifest(a.manifest, a.out, a.workers)


def cmd_train_artifact(a):
    from .cadet.train import read_manifest, train_artifact

    train = [r for s in a.train for r in read_manifest(s)]
    print(train_artifact(train, read_manifest(a.val), a.frames_root, a.out, a.seed, a.epochs, a.bs, a.accum, a.workers,
                         single_head=a.single_head))


def cmd_train_semantic(a):
    from .cadet.train import read_manifest, train_semantic

    train_semantic(read_manifest(a.train), a.frames_root, a.out, a.seed, a.epochs, a.bs, a.workers)


def cmd_calibrate(a):
    """Write <weights>/calibration.json: head and fusion statistics on validation reals."""
    import numpy as np
    import torch

    from .cadet.detector import _find, _load_partial, _load_tensors
    from .cadet.models import ArtifactPath, SemanticPath
    from .cadet.train import ClipDataset, predict_artifact, predict_semantic, read_manifest

    reals = [r for r in read_manifest(a.val) if int(r["label"]) == 0]
    art = ArtifactPath().to(a.device)
    _load_partial(art, _load_tensors(_find(a.weights, "artifact_path")), "artifact path")
    A, B = predict_artifact(art, torch.utils.data.DataLoader(ClipDataset(reals, a.frames_root, False), batch_size=8, num_workers=8), a.device)
    heads = {"A_mean": float(A.mean()), "A_std": float(A.std() + 1e-6), "B_mean": float(B.mean()), "B_std": float(B.std() + 1e-6)}
    s_art = (A - heads["A_mean"]) / heads["A_std"] + (B - heads["B_mean"]) / heads["B_std"]
    del art
    sem = SemanticPath(device=a.device).to(a.device)
    _load_partial(sem, _load_tensors(_find(a.weights, "semantic_path")), "semantic path")
    s_sem = predict_semantic(sem, reals, a.frames_root, a.device)
    cal = {"heads": heads, "fusion": {"sem": [float(np.mean(s_sem)), float(np.std(s_sem, ddof=1))],
                                      "art": [float(np.mean(s_art)), float(np.std(s_art, ddof=1))]},
           "n_val_reals": len(reals)}
    json.dump(cal, open(os.path.join(a.weights, "calibration.json"), "w"), indent=1)
    print(json.dumps(cal, indent=1))


def cmd_predict(a):
    from .cadet.detector import CaDet

    det = CaDet(a.weights, a.device)
    out = []
    if a.manifest:
        for r in _rows(a.manifest)[a.shard::a.nshards]:
            try:
                s = det.score_cached(a.frames_root, r) if a.frames_root else det.score_video(r["path"])
                out.append({"video_id": r["video_id"], "label": r.get("label", ""), **s})
            except Exception as e:  # noqa: BLE001
                out.append({"video_id": r["video_id"], "label": r.get("label", ""), "error": str(e)[:200]})
    for path in a.videos:
        s = det.score_video(path)
        if "z_semantic" in s:
            print(f"{path}: CaDet score {s['score']:+.2f} (semantic z {s['z_semantic']:+.2f}, artifact z {s['z_artifact']:+.2f})")
        else:
            print(f"{path}: CaDet-GF logit {s['score']:+.2f}")
        out.append({"video_id": path, **s})
    if a.out:
        _write_csv(a.out, out)


def cmd_plugin(a):
    import numpy as np

    from .metrics import per_subset, read_scores, summarize
    from .plugin import HybridStandardizer, MeanStdStandardizer, fuse, z_ppk
    from .scan import load_calibration

    det = read_scores(a.scores)
    ppk = read_scores(a.ppk, "Ppk")
    real_det = list(read_scores(a.det_real).values())
    cal = load_calibration(a.ppk_calibration)
    std = HybridStandardizer(real_det) if a.norm == "hybrid" else MeanStdStandardizer(real_det)
    ids = [v for v in det if v in ppk]
    zd = std(np.array([det[v] for v in ids]))
    zc = z_ppk(np.array([ppk[v] for v in ids]), cal["ppk_mean"], cal["ppk_std"])
    fused = dict(zip(ids, fuse(zd, zc, a.tau)))
    rows = [{"video_id": v, "score": f, "z_detector": d, "z_ppk": c} for v, f, d, c in zip(ids, fused.values(), zd, zc)]
    if a.out:
        _write_csv(a.out, rows)
    if a.manifest:
        man = _rows(a.manifest)
        for name, sc in (("detector", {v: float(z) for v, z in zip(ids, zd)}), ("rhythm only", {v: float(z) for v, z in zip(ids, zc)}),
                         ("detector + plug-in", fused)):
            print(f"{name:20s}", {k: round(v, 2) for k, v in summarize(per_subset(man, sc)).items()})


def cmd_evaluate(a):
    from .metrics import per_subset, read_scores, summarize

    recs = per_subset(_rows(a.manifest), read_scores(a.scores, a.column))
    for r in recs:
        print(f"{r['subset']:32s} n_fake {r['n_fake']:5d}  AUROC {r['AUROC']:6.2f}" + (f"  2AFC {r['paired_2afc']:6.2f}" if "paired_2afc" in r else ""))
    print({k: round(v, 2) for k, v in summarize(recs).items()})
    if a.out:
        _write_csv(a.out, recs)


def cmd_doctor(a):
    """Check the environment; prints one OK / FAIL line per item and exits non-zero on failures."""
    import importlib
    import shutil
    import subprocess

    from .video import FFMPEG, FFPROBE

    ok = True

    def report(name, good, detail=""):
        nonlocal ok
        ok &= bool(good) or name.startswith("(optional)")
        print(f"{'OK  ' if good else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)

    for tool in (FFMPEG, FFPROBE):
        report(tool, shutil.which(tool) or os.path.isfile(tool))
    if shutil.which(FFMPEG) or os.path.isfile(FFMPEG):
        enc = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        report("ffmpeg libx264", "libx264" in enc)
        report("ffmpeg libx265 (VAE-pair engine)", "libx265" in enc)
        filt = subprocess.run([FFMPEG, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
        report("ffmpeg minterpolate (VAE-pair engine)", "minterpolate" in filt)
    for mod, need in (("numpy", None), ("torch", None), ("diffusers", "0.33"), ("av", None), ("PIL", None),
                      ("matplotlib", None), ("safetensors", None)):
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "")
            good = need is None or tuple(int(x) for x in ver.split(".")[:2]) >= tuple(int(x) for x in need.split("."))
            report(f"{mod} {ver}", good, f"needs >= {need}" if not good else "")
        except Exception as e:  # noqa: BLE001
            report(mod, False, str(e)[:120])
    for mod in ("timm", "peft", "cv2", "sklearn"):
        try:
            m = importlib.import_module(mod)
            report(f"(optional) {mod} {getattr(m, '__version__', '')} [CaDet]", True)
        except Exception as e:  # noqa: BLE001
            report(f"(optional) {mod} [CaDet]", False, str(e)[:80])
    try:
        import diffusers

        report("diffusers AutoencoderKLWan", hasattr(diffusers, "AutoencoderKLWan"))
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        cuda = torch.cuda.is_available()
        report("CUDA GPU", cuda, torch.cuda.get_device_name(0) if cuda else "scans on CPU are very slow")
        if cuda:
            report("bf16 support", torch.cuda.is_bf16_supported())
    except Exception:  # noqa: BLE001
        pass
    sys.exit(0 if ok else 1)


def cmd_selftest(a):
    """No dataset needed: synthesise a moving clip, make a Wan2.1 round-trip copy of it, scan both.
    The round-trip copy carries a VAE lattice, so its rhythm must be clearly stronger."""
    import subprocess
    import tempfile

    from .engine.pairs import vae_pass, vae_prep
    from .scan import scan_video
    from .vaes import VAE
    from .video import FFMPEG, read_frames, write_frames

    vae = VAE("wan21", a.device, "bf16", source=a.vae_path)
    with tempfile.TemporaryDirectory() as tmp:
        src, dec = os.path.join(tmp, "source.mp4"), os.path.join(tmp, "vae_roundtrip.mp4")
        subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"{a.pattern}=size=640x360:rate=24",
                        "-t", "3", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "12", src], check=True)
        frames = vae_prep(read_frames(src, num=49), vae)
        write_frames(vae_pass(frames, vae), dec, 24.0, crf=12)
        res = {name: scan_video(p, vae, 720, segments=1) for name, p in (("source", src), ("VAE round trip", dec))}
        for name, r in res.items():
            print(f"{name:15s} Ppk {r.ppk:.3f}  period {r.stats['ppk']:.2f}  C4 {r.stats['C4']:.3f}  z(Ppk) {r.z_ppk:+.2f}", flush=True)
        good = res["VAE round trip"].ppk > res["source"].ppk and res["VAE round trip"].stats["C4"] > res["source"].stats["C4"]
        print("SELFTEST", "PASSED" if good else "FAILED", flush=True)
        sys.exit(0 if good else 1)


def main(argv=None):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    p = argparse.ArgumentParser(prog="vae-rhythm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("doctor", help="check ffmpeg, Python packages and the GPU")
    q.set_defaults(fn=cmd_doctor)

    q = sub.add_parser("selftest", help="GPU check without data: synthetic clip vs. its Wan2.1 round trip")
    q.add_argument("--device", default="cuda")
    q.add_argument("--vae-path", default=None)
    q.add_argument("--pattern", default="testsrc2", help="ffmpeg lavfi source (testsrc2, mandelbrot, ...)")
    q.set_defaults(fn=cmd_selftest)

    def scan_args(q):
        q.add_argument("--vae", default="wan21")
        q.add_argument("--vae-path", default=None, help="local diffusers folder instead of the Hugging Face repository")
        q.add_argument("--cap", type=int, default=720, help="short-side cap of the scan (paper: 720 for the plug-in, 480 for the census)")
        q.add_argument("--K", type=int, default=24)
        q.add_argument("--W", type=int, default=13)
        q.add_argument("--batch", type=int, default=1, help="windows per VAE call (2 needs ~40 GB at 720p)")
        q.add_argument("--device", default="cuda")
        q.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "fp16"])
        q.add_argument("--portable", action="store_true",
                       help="single-threaded x264 without assembly: identical results on every machine (slower encode)")

    q = sub.add_parser("scan", help="phase scan of videos")
    q.add_argument("videos", nargs="+")
    scan_args(q)
    q.add_argument("--segments", type=int, default=1, help="1 = first 36 frames (paper); 3 = start / middle / end, median")
    q.add_argument("--out", default="", help="folder for <video>.json and <video>.png")
    q.set_defaults(fn=cmd_scan)

    q = sub.add_parser("scan-manifest", help="phase scan of a manifest")
    q.add_argument("manifest", help="CSV with video_id, path[, label, subset]; 'file.csv:split' filters by split")
    q.add_argument("--out", required=True)
    q.add_argument("--npz-dir", default=None)
    q.add_argument("--shard", type=int, default=0)
    q.add_argument("--nshards", type=int, default=1)
    scan_args(q)
    q.set_defaults(fn=cmd_scan_manifest)

    q = sub.add_parser("make-pairs", help="VAE-pair engine")
    q.add_argument("--reals", default="", help="folder of real videos")
    q.add_argument("--reals-manifest", default="", help="or a manifest (rows with label 0), e.g. 'aigvd.csv:train'")
    q.add_argument("--out", required=True)
    q.add_argument("--vaes", default="wan21,hunyuan,cogx,ltx")
    q.add_argument("--gen-scales", default="256,256,256,256,360,360,480,480,720", help="reconstruction short sides G (repeat = weight)")
    q.add_argument("--out-scales", default="480,720,1080", help="delivery short sides S")
    q.add_argument("--allow-upscale", action="store_true", help="S may exceed G (upscaled delivery)")
    q.add_argument("--interp-prob", type=float, default=0.2)
    q.add_argument("--dup-prob", type=float, default=0.25)
    q.add_argument("--crf-max", type=int, default=28)
    q.add_argument("--frames", type=int, default=81)
    q.add_argument("--n", type=int, default=0)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--tag", default="")
    q.add_argument("--shard", type=int, default=0)
    q.add_argument("--nshards", type=int, default=1)
    q.add_argument("--device", default="cuda")
    q.set_defaults(fn=cmd_make_pairs)

    q = sub.add_parser("cache-frames", help="16-frame cache for CaDet")
    q.add_argument("manifest")
    q.add_argument("--out", required=True)
    q.add_argument("--workers", type=int, default=16)
    q.set_defaults(fn=cmd_cache_frames)

    q = sub.add_parser("train-artifact", help="train the artifact path")
    q.add_argument("--train", nargs="+", required=True, help="e.g. aigvd.csv:train pairs.csv:train")
    q.add_argument("--val", required=True)
    q.add_argument("--frames-root", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--epochs", type=int, default=8)
    q.add_argument("--bs", type=int, default=8)
    q.add_argument("--accum", type=int, default=3)
    q.add_argument("--workers", type=int, default=10)
    q.add_argument("--single-head", action="store_true", help="CaDet-GF: one head, original labels, best epoch on --val")
    q.set_defaults(fn=cmd_train_artifact)

    q = sub.add_parser("train-semantic", help="train the semantic path")
    q.add_argument("--train", required=True, help="benchmark training split only, e.g. aigvd.csv:train")
    q.add_argument("--frames-root", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--epochs", type=int, default=11)
    q.add_argument("--bs", type=int, default=32)
    q.add_argument("--workers", type=int, default=8)
    q.set_defaults(fn=cmd_train_semantic)

    q = sub.add_parser("calibrate", help="fusion statistics on validation reals")
    q.add_argument("--weights", required=True, help="folder with artifact_path.* and semantic_path.*")
    q.add_argument("--val", required=True, help="validation manifest, e.g. aigvd.csv:val")
    q.add_argument("--frames-root", required=True)
    q.add_argument("--device", default="cuda")
    q.set_defaults(fn=cmd_calibrate)

    q = sub.add_parser("predict", help="CaDet scores")
    q.add_argument("videos", nargs="*")
    q.add_argument("--weights", required=True)
    q.add_argument("--manifest", default="")
    q.add_argument("--frames-root", default="", help="use cached frames for manifest rows")
    q.add_argument("--out", default="")
    q.add_argument("--device", default="cuda")
    q.add_argument("--shard", type=int, default=0, help="score rows shard::nshards (run several processes / GPUs)")
    q.add_argument("--nshards", type=int, default=1)
    q.set_defaults(fn=cmd_predict)

    q = sub.add_parser("plugin", help="VAE Rhythm plug-in (Eq. 3)")
    q.add_argument("--scores", required=True, help="detector score CSV (video_id, score)")
    q.add_argument("--det-real", required=True, help="the same detector's scores on calibration real videos")
    q.add_argument("--ppk", required=True, help="CSV with video_id, Ppk (from scan-manifest)")
    q.add_argument("--ppk-calibration", default="wan21_k24w13_cap720.json")
    q.add_argument("--norm", default="hybrid", choices=["hybrid", "meanstd"])
    q.add_argument("--tau", type=float, default=0.0)
    q.add_argument("--manifest", default="", help="manifest for AUROC (video_id, label, subset)")
    q.add_argument("--out", default="")
    q.set_defaults(fn=cmd_plugin)

    q = sub.add_parser("evaluate", help="per-generator AUROC")
    q.add_argument("scores")
    q.add_argument("--manifest", required=True)
    q.add_argument("--column", default="score")
    q.add_argument("--out", default="")
    q.set_defaults(fn=cmd_evaluate)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
