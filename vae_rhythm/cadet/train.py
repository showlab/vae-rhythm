"""Training both CaDet paths (paper Sec. 4.2 and supplement).

Artifact path (``train_artifact``): benchmark training set + VAE pairs (dataset == "synth"),
8 frames per clip, random resized crop + flip, AdamW (adapters 1e-4, heads 1e-3, weight decay
0.05), one warm-up epoch then cosine decay, batch 24 clips (8 x 3 accumulation steps), bf16
autocast, gradient clipping 1.0, 8 epochs, last epoch kept. Loss = BCE(generic head, label with
every VAE-pair member counted real) + BCE(artifact head, original label). Afterwards both heads
are standardised on the validation reals (``calib.json``).

Semantic path (``train_semantic``): benchmark training set only (no VAE pairs), each of the 8
frames of a clip is an independent sample, batches of 32 frames from random clips, full-frame
bicubic resize to 224 then the 11 per-frame transforms, 2-way cross-entropy, Adam lr 2e-4,
weight decay 5e-4, no schedule, fp32, no gradient clipping, 11 epochs, last epoch kept.

Manifests are CSV files with at least: dataset, video_id, label (0 real / 1 generated),
subset, split. ``path:split`` selects rows of one split.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .augment import dfb_frame, rrc_clip
from .frames import FRAME_IDX, IMNET_MEAN, IMNET_STD, has_cache, load_cached
from .models import ArtifactPath, SemanticPath, video_logit_from_frames


def read_manifest(spec: str) -> list[dict]:
    """'file.csv[:split]' -> rows (filtered by the ``split`` column when given)."""
    path, _, split = spec.partition(":")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if not split or r.get("split", "") == split]


def _cached(rows, root, name):
    keep = [r for r in rows if has_cache(root, r)]
    if len(keep) < len(rows):
        print(f"{name}: {len(rows) - len(keep)} rows without cached frames dropped ({len(keep)} kept)", flush=True)
    return keep


def _seed_worker(_):
    np.random.seed(torch.initial_seed() % 2 ** 32)


# --------------------------------------------------------------------------- artifact path
class ClipDataset(Dataset):
    """8 frames per clip -> (T, 3, 224, 224) normalised, label, is-VAE-pair flag."""

    def __init__(self, rows, root, train):
        self.rows, self.root, self.train = rows, root, train
        self.mean = torch.tensor(IMNET_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(IMNET_STD).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = torch.from_numpy(load_cached(self.root, r)).permute(0, 3, 1, 2).float() / 255.0
        x = (rrc_clip(x, self.train) - self.mean) / self.std
        return x, int(r["label"]), float(r.get("dataset") == "synth"), i


@torch.no_grad()
def predict_artifact(model: ArtifactPath, loader, device="cuda"):
    model.eval()
    A, B = [], []
    for x, *_ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, b = model(x.to(device, non_blocking=True))
        A.append(a.float().cpu().numpy())
        B.append(b.float().cpu().numpy())
    return np.concatenate(A), np.concatenate(B)


def train_artifact(train: list[dict], val: list[dict], frames_root: str, out: str, seed: int = 0, epochs: int = 8,
                   bs: int = 8, accum: int = 3, workers: int = 10, lr: float = 1e-4, device: str = "cuda",
                   single_head: bool = False) -> dict:
    """``single_head=True`` is CaDet-GF: one linear head with the original labels (VAE negatives are
    generated), trained on real clips + pairs, keeping the epoch with the best AUROC on ``val`` (use the
    held-out pairs so that no generator-produced video is involved)."""
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out, exist_ok=True)
    train, val = _cached(train, frames_root, "train"), _cached(val, frames_root, "val")
    ltr = DataLoader(ClipDataset(train, frames_root, True), batch_size=bs, shuffle=True, num_workers=workers,
                     drop_last=True, pin_memory=True, persistent_workers=workers > 0, worker_init_fn=_seed_worker)
    lva = DataLoader(ClipDataset(val, frames_root, False), batch_size=bs, shuffle=False, num_workers=workers)
    model = ArtifactPath().to(device)
    is_syn = torch.tensor([float(r.get("dataset") == "synth") for r in train])
    bb = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("vision")]
    hd = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("vision")]
    opt = torch.optim.AdamW([{"params": bb, "lr": lr}, {"params": hd, "lr": 1e-3}], weight_decay=0.05)
    steps, warm = epochs * (len(ltr) // accum), max(1, len(ltr) // accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    yva = np.array([int(r["label"]) for r in val])
    best, best_state = -1.0, None
    print(f"artifact path: trainable {sum(p.numel() for p in bb + hd) / 1e6:.2f}M, train {len(train)} "
          f"(VAE-pair rows {int(is_syn.sum())}), val {len(val)}, {len(ltr)} batches/epoch", flush=True)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        tot, n = 0.0, 0
        for it, (x, y, _, idx) in enumerate(ltr):
            x, y = x.to(device, non_blocking=True), y.float().to(device)
            syn = is_syn[idx].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a, b = model(x)
            if single_head:
                loss = F.binary_cross_entropy_with_logits(a.float(), y)
            else:
                loss = F.binary_cross_entropy_with_logits(a.float(), y * (1 - syn)) \
                    + F.binary_cross_entropy_with_logits(b.float(), y)
            (loss / accum).backward()
            if (it + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            tot, n = tot + loss.item() * len(y), n + len(y)
        A, B = predict_artifact(model, lva, device)
        auc = roc_auc_score(yva, A)
        print(f"epoch {ep} loss {tot / n:.4f} val AUROC generic {100 * auc:.2f} "
              f"artifact {100 * roc_auc_score(yva, B):.2f} ({time.time() - t0:.0f}s)", flush=True)
        if single_head and (best_state is None or auc >= best):        # ties keep the later epoch
            best, best_state = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    if single_head:
        model.load_state_dict(best_state)
        A, _ = predict_artifact(model, lva, device)
        calib = {"single_head": True, "A_mean": float(A[yva == 0].mean()), "A_std": float(A[yva == 0].std() + 1e-6),
                 "best_val_auroc": float(best)}
        json.dump(calib, open(os.path.join(out, "calib.json"), "w"), indent=1)
        torch.save(model.trainable_state_dict(), os.path.join(out, "artifact_path.pt"))
        return calib
    A, B = predict_artifact(model, lva, device)
    r0 = yva == 0
    calib = {"A_mean": float(A[r0].mean()), "A_std": float(A[r0].std() + 1e-6),
             "B_mean": float(B[r0].mean()), "B_std": float(B[r0].std() + 1e-6)}
    json.dump(calib, open(os.path.join(out, "calib.json"), "w"), indent=1)
    torch.save(model.trainable_state_dict(), os.path.join(out, "artifact_path.pt"))
    return calib


# --------------------------------------------------------------------------- semantic path
class FrameDataset(Dataset):
    """Every cached frame of every clip is one sample (frame-level training, as in Effort)."""

    def __init__(self, rows, root, train):
        import cv2  # noqa: F401  (fail early if OpenCV is missing)

        self.items = [(r, k) for r in rows for k in FRAME_IDX]
        self.root, self.train = root, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import cv2

        r, k = self.items[i]
        im = load_cached(self.root, r, [k])[0]
        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)
        if self.train:
            im = dfb_frame(im)
        return semantic_tensor(im), int(r["label"])


def semantic_tensor(im: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0
    return (x - torch.tensor(IMNET_MEAN).view(3, 1, 1)) / torch.tensor(IMNET_STD).view(3, 1, 1)


def train_semantic(train: list[dict], frames_root: str, out: str, seed: int = 0, epochs: int = 11, bs: int = 32,
                   workers: int = 8, device: str = "cuda") -> None:
    torch.manual_seed(1024 + seed)
    np.random.seed(1024 + seed)
    os.makedirs(out, exist_ok=True)
    train = [r for r in _cached(train, frames_root, "train") if r.get("dataset") != "synth"]
    loader = DataLoader(FrameDataset(train, frames_root, True), batch_size=bs, shuffle=True, num_workers=workers,
                        drop_last=True, pin_memory=True, worker_init_fn=_seed_worker)
    model = SemanticPath(device=device).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4)
    print(f"semantic path: trainable {sum(p.numel() for p in params)} parameters, {len(loader.dataset)} frames from "
          f"{len(train)} clips, {len(loader)} steps/epoch", flush=True)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
        print(f"epoch {ep} loss {tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.trainable_state_dict(), os.path.join(out, "semantic_path.pt"))


@torch.no_grad()
def predict_semantic(model: SemanticPath, rows, frames_root, device="cuda", bs: int = 32) -> np.ndarray:
    import cv2

    model.eval()
    out = []
    for i in range(0, len(rows), max(1, bs // len(FRAME_IDX))):
        chunk = rows[i:i + max(1, bs // len(FRAME_IDX))]
        x = torch.stack([semantic_tensor(cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC))
                         for r in chunk for im in load_cached(frames_root, r)])
        out.append(video_logit_from_frames(model(x.to(device)), len(FRAME_IDX)).cpu().numpy())
    return np.concatenate(out)
