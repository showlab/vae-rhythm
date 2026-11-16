"""CaDet inference: two paths, standardised on validation reals, summed (paper Sec. 4.2).

    s_d = (s_sem - mu_sem) / sd_sem + (s_art - mu_art) / sd_art

where s_art is itself z(generic head) + z(artifact head). All means and standard deviations
come from AIGVDBench validation real videos; no weight is fitted.

A released checkpoint folder holds
    artifact_path.safetensors   LoRA adapters + both heads of the artifact path
    semantic_path.safetensors   rank-one SVD residuals + 2-way head of the semantic path
    calibration.json            {"heads": {A_mean, A_std, B_mean, B_std},
                                 "fusion": {"sem": [mean, std], "art": [mean, std]}}
The frozen DINOv3 ViT-L/16 backbone is downloaded separately through timm.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .frames import FRAME_IDX, IMNET_MEAN, IMNET_STD, clip_frames, load_cached
from .models import ArtifactPath, SemanticPath, video_logit_from_frames


def _load_tensors(path: str) -> dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu")


def _load_partial(model: torch.nn.Module, sd: dict, kind: str) -> None:
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"{kind}: unexpected keys {unexpected[:5]}")
    trained = [k for k in missing if "lora_" in k or "residual" in k or k.startswith("head")]
    if trained:
        raise RuntimeError(f"{kind}: checkpoint lacks trained parameters {trained[:5]}")


class CaDet:
    """Two-path CaDet. A folder with only ``artifact_path.*`` and a ``calib.json`` marked ``single_head``
    (written by ``train-artifact --single-head``) is loaded as CaDet-GF, whose score is the head's logit."""

    def __init__(self, weights: str, device: str = "cuda"):
        self.device = device
        gf = os.path.join(weights, "calib.json")
        self.single_head = not os.path.exists(os.path.join(weights, "calibration.json")) and os.path.exists(gf) \
            and json.load(open(gf)).get("single_head", False)
        self.art = ArtifactPath().to(device).eval()
        _load_partial(self.art, _load_tensors(_find(weights, "artifact_path")), "artifact path")
        if self.single_head:
            self.heads, self.fusion, self.sem = None, None, None
        else:
            cal = json.load(open(os.path.join(weights, "calibration.json")))
            self.heads, self.fusion = cal["heads"], cal["fusion"]
            self.sem = SemanticPath(device=device).to(device).eval()
            _load_partial(self.sem, _load_tensors(_find(weights, "semantic_path")), "semantic path")
        self._mean = torch.tensor(IMNET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(IMNET_STD).view(1, 3, 1, 1)

    @torch.no_grad()
    def score_frames(self, frames: np.ndarray) -> dict:
        """frames: uint8 (8, 256, 256, 3) -- the 8 cache frames 0, 4, ..., 28 of one clip."""
        import cv2

        x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
        xa = ((x[:, :, 16:240, 16:240] - self._mean) / self._std).unsqueeze(0).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            a, b = self.art(xa)
        if self.single_head:
            return {"score": a.float().item(), "generic_logit": a.float().item()}
        h = self.heads
        s_art = (a.float().item() - h["A_mean"]) / h["A_std"] + (b.float().item() - h["B_mean"]) / h["B_std"]
        xs = torch.stack([torch.from_numpy(cv2.resize(f, (224, 224), interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).float() / 255.0
                          for f in frames])
        xs = ((xs - self._mean) / self._std).to(self.device)
        s_sem = float(video_logit_from_frames(self.sem(xs), len(frames))[0])
        (ms, ss), (ma, sa) = self.fusion["sem"], self.fusion["art"]
        z_sem, z_art = (s_sem - ms) / ss, (s_art - ma) / sa
        return {"score": z_sem + z_art, "z_semantic": z_sem, "z_artifact": z_art, "semantic": s_sem, "artifact": s_art,
                "generic_logit": a.float().item(), "artifact_logit": b.float().item()}

    def score_video(self, path: str) -> dict:
        return self.score_frames(clip_frames(path)[FRAME_IDX])

    def score_cached(self, frames_root: str, row: dict) -> dict:
        return self.score_frames(load_cached(frames_root, row))


def _find(folder: str, stem: str) -> str:
    for ext in (".safetensors", ".pt", ".pth"):
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{stem}.safetensors not found in {folder}")


def export(artifact_ckpt: str, semantic_ckpt: str, head_calib: dict, fusion_stats: dict, out: str) -> None:
    """Convert full training checkpoints into a release folder (only trained tensors are kept)."""
    from safetensors.torch import save_file

    os.makedirs(out, exist_ok=True)
    art = torch.load(artifact_ckpt, map_location="cpu")
    art = {k: v.contiguous() for k, v in art.items() if "lora_" in k or k.startswith("head")}
    sem = torch.load(semantic_ckpt, map_location="cpu", weights_only=False)
    sem = sem.get("state_dict", sem)
    sem = {k: v.contiguous() for k, v in sem.items() if k.startswith("head") or any(s in k for s in ("S_residual", "U_residual", "V_residual"))}
    save_file(art, os.path.join(out, "artifact_path.safetensors"))
    save_file(sem, os.path.join(out, "semantic_path.safetensors"))
    json.dump({"heads": head_calib, "fusion": fusion_stats}, open(os.path.join(out, "calibration.json"), "w"), indent=1)
