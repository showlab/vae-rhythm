"""Open video VAEs used as rulers (phase scan) and as round-trip operators (VAE-pair engine).

Only encode -> decode round trips are used; no diffusion model is loaded.

name      diffusers class              weights (Hugging Face)                  time  space  valid frame counts
wan21     AutoencoderKLWan             Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae    4x    8x     1 + 4n
wan22     AutoencoderKLWan             Wan-AI/Wan2.2-TI2V-5B-Diffusers/vae     4x    16x    1 + 4n
hunyuan   AutoencoderKLHunyuanVideo    hunyuanvideo-community/HunyuanVideo/vae 4x    8x     1 + 4n
cogx      AutoencoderKLCogVideoX       THUDM/CogVideoX-5b/vae                  4x    8x     1 + 8n (*)
ltx       AutoencoderKLLTXVideo        Lightricks/LTX-Video/vae                8x    32x    1 + 8n

(*) The diffusers CogVideoX decoder only decodes the first frame on its own when the number
of latent frames is odd, so windows must have 1 + 8n frames (9, 17, 25, 33).

Inputs and outputs are float tensors (B, 3, T, H, W) in [-1, 1]; H and W must be multiples
of the spatial factor (use :func:`crop_to_multiple` first). The posterior mode is used, so
round trips are deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPECS = {
    # name: (diffusers class, repository, temporal factor, spatial factor, frame-count step)
    "wan21": ("AutoencoderKLWan", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", 4, 8, 4),
    "wan22": ("AutoencoderKLWan", "Wan-AI/Wan2.2-TI2V-5B-Diffusers", 4, 16, 4),
    "hunyuan": ("AutoencoderKLHunyuanVideo", "hunyuanvideo-community/HunyuanVideo", 4, 8, 4),
    "cogx": ("AutoencoderKLCogVideoX", "THUDM/CogVideoX-5b", 4, 8, 8),
    "ltx": ("AutoencoderKLLTXVideo", "Lightricks/LTX-Video", 8, 32, 8),
}

DTYPES = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}


@dataclass
class VAESpec:
    name: str
    cls: str
    repo: str
    t_factor: int
    s_factor: int
    t_step: int


def spec(name: str) -> VAESpec:
    if name not in SPECS:
        raise KeyError(f"Unknown VAE {name!r}; choose from {sorted(SPECS)}")
    return VAESpec(name, *SPECS[name])


class VAE:
    """Thin wrapper: ``VAE("wan21").roundtrip(x)``.

    ``source`` may override the Hugging Face repository with a local diffusers folder
    (either the model root containing ``vae/`` or the ``vae/`` folder itself).
    """

    def __init__(self, name: str = "wan21", device: str = "cuda", dtype: str = "bf16", source: str | None = None):
        import torch
        import diffusers
        from pathlib import Path

        s = spec(name)
        self.name, self.t_factor, self.s_factor, self.t_step = s.name, s.t_factor, s.s_factor, s.t_step
        self.device = device
        self.dtype = getattr(torch, DTYPES.get(dtype, dtype))
        repo = source or s.repo
        kwargs = {"torch_dtype": self.dtype}
        if not (Path(repo).is_dir() and (Path(repo) / "config.json").is_file()):
            kwargs["subfolder"] = "vae"
        self.vae = getattr(diffusers, s.cls).from_pretrained(repo, **kwargs).to(device).eval()
        self.vae.requires_grad_(False)

    def valid_T(self, T: int) -> int:
        """Largest valid frame count (1 + n * t_step) not above T."""
        return ((T - 1) // self.t_step) * self.t_step + 1

    def roundtrip(self, x):
        """x: (B, 3, T, H, W) in [-1, 1] -> reconstruction of the same shape (float32)."""
        import torch

        with torch.inference_mode():
            x = x.to(self.device, self.dtype)
            post = self.vae.encode(x).latent_dist
            z = post.mode() if hasattr(post, "mode") else post.mean
            y = self.vae.decode(z, temb=None).sample if self.name == "ltx" else self.vae.decode(z).sample
        if y.shape[2] != x.shape[2]:
            raise RuntimeError(f"{self.name}: decoded {y.shape[2]} frames from {x.shape[2]}; "
                               f"windows must have 1 + {self.t_step}n frames")
        return y.float()

    def clear_cache(self) -> None:
        clear = getattr(self.vae, "clear_cache", None)
        if callable(clear):
            clear()


def crop_to_multiple(frames: np.ndarray, m: int) -> np.ndarray:
    """Centre-crop uint8 frames (T, H, W, 3) so that H and W are multiples of m."""
    T, H, W, _ = frames.shape
    h, w = H - H % m, W - W % m
    y0, x0 = (H - h) // 2, (W - w) // 2
    return frames[:, y0:y0 + h, x0:x0 + w]


def to_tensor(frames: np.ndarray):
    """uint8 (T, H, W, 3) -> float tensor (1, 3, T, H, W) in [-1, 1]."""
    import torch

    t = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).float().unsqueeze(0)
    return t / 127.5 - 1.0


def to_uint8(y) -> np.ndarray:
    """float tensor (1, 3, T, H, W) in [-1, 1] -> uint8 frames (T, H, W, 3)."""
    return ((y.clamp(-1, 1) + 1) * 127.5).round().byte()[0].permute(1, 2, 3, 0).cpu().numpy()
