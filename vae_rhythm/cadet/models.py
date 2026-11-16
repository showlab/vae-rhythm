"""The two CaDet paths on one frozen DINOv3 ViT-L/16 (paper Sec. 4.2).

ArtifactPath  LoRA (rank 16, alpha 32, dropout 0.05) on the fused qkv projections; one forward
              pass over 8 frames feeds two heads:
              * generic head  -- mean per-frame CLS token -> Dropout(0.2) -> Linear(1024, 1);
                trained with *both* members of every VAE pair labelled real;
              * artifact head -- [CLS, patch mean, patch std] of blocks 8, 12, 16, averaged
                over time -> LayerNorm -> Linear(9216, 256) -> GELU -> Dropout(0.2) -> Linear(256, 1);
                trained with the original labels (VAE negatives count as generated) and its
                gradient also updates the shared adapters.
              Path score = z(generic) + z(artifact), each standardised on validation reals.
SemanticPath  Effort-style tuning: every attention projection W = U S V^T (q, k, v split out
              of the fused qkv, and proj; 96 matrices) keeps its top 1023 components frozen and
              trains only the smallest one (u s v^T), plus a Linear(1024, 2) head on the final
              CLS token. Frame-level; a clip's score averages its frame probabilities.

Attribute names match the training code, so the released state dicts load directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TIMM_ID = "vit_large_patch16_dinov3.lvd1689m"
EMBED = 1024


def dinov3(pretrained: bool = True):
    import timm

    return timm.create_model(TIMM_ID, pretrained=pretrained, num_classes=0)


# --------------------------------------------------------------------------- artifact path
class ArtifactPath(nn.Module):
    def __init__(self, b_layers=(8, 12, 16), lora_r: int = 16, pretrained: bool = True):
        super().__init__()
        from peft import LoraConfig, get_peft_model

        m = dinov3(pretrained)
        self.vision = get_peft_model(m, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                                                   target_modules=["qkv"]))
        self.n_prefix = getattr(m, "num_prefix_tokens", 1)          # CLS + register tokens
        self.b_layers = [int(x) for x in b_layers]
        self._cap: dict[int, torch.Tensor] = {}
        blocks = [mod for n, mod in self.vision.named_modules() if n.endswith("blocks") and isinstance(mod, nn.ModuleList)][0]
        for li in self.b_layers:
            blocks[li - 1].register_forward_hook(lambda mod, inp, out, li=li: self._cap.__setitem__(li, out))
        d_b = 3 * EMBED * len(self.b_layers)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(EMBED, 1))
        self.head_b = nn.Sequential(nn.LayerNorm(d_b), nn.Linear(d_b, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x):
        """x: (B, T, 3, 224, 224), ImageNet-normalised -> (generic logit, artifact logit), each (B,)."""
        B, T = x.shape[:2]
        self._cap.clear()
        f = self.vision.forward_features(x.flatten(0, 1))[:, 0]
        parts = []
        for li in self.b_layers:
            t = self._cap[li]
            pt = t[:, self.n_prefix:]
            parts += [t[:, 0], pt.mean(1), pt.std(1)]
        fb = torch.cat(parts, -1).view(B, T, -1).mean(1)
        f = f.view(B, T, -1).mean(1)
        with torch.autocast("cuda", enabled=False):                   # heads in fp32
            a = self.head(f.float()).squeeze(-1)
            b = self.head_b(fb.float()).squeeze(-1)
        return a, b

    def trainable_state_dict(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if "lora_" in k or k.startswith("head")}


# --------------------------------------------------------------------------- semantic path
class SVDResidualLinear(nn.Module):
    """W = weight_main (top-r components, frozen) + U_residual diag(S_residual) V_residual (trained)."""

    def __init__(self, lin: nn.Linear, r: int):
        super().__init__()
        W = lin.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = min(r, len(S))
        self.weight_main = nn.Parameter(U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :], requires_grad=False)
        self.S_residual = nn.Parameter(S[r:].clone())
        self.U_residual = nn.Parameter(U[:, r:].clone())
        self.V_residual = nn.Parameter(Vh[r:, :].clone())
        self.bias = nn.Parameter(lin.bias.data.clone().float(), requires_grad=False) if lin.bias is not None else None

    def forward(self, x):
        w = self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        return F.linear(x, w, self.bias)


class QKVSplit(nn.Module):
    """timm's fused qkv (3072 x 1024) split into q / k / v SVD-residual layers, output re-concatenated."""

    def __init__(self, qkv: nn.Linear, r: int):
        super().__init__()
        D = qkv.in_features
        for i, n in enumerate("qkv"):
            lin = nn.Linear(D, D, bias=qkv.bias is not None)
            lin.weight.data.copy_(qkv.weight.data[i * D:(i + 1) * D])
            if qkv.bias is not None:
                lin.bias.data.copy_(qkv.bias.data[i * D:(i + 1) * D])
            setattr(self, n, SVDResidualLinear(lin, r))

    def forward(self, x):
        return torch.cat([self.q(x), self.k(x), self.v(x)], -1)


class SemanticPath(nn.Module):
    def __init__(self, pretrained: bool = True, device: str | None = None):
        super().__init__()
        m = dinov3(pretrained).float()
        if device is not None:
            m = m.to(device)                     # the 96 SVDs below are much faster on a GPU
        r = m.embed_dim - 1
        for blk in m.blocks:
            blk.attn.qkv = QKVSplit(blk.attn.qkv, r)
            blk.attn.proj = SVDResidualLinear(blk.attn.proj, r)
        for n, p in m.named_parameters():
            p.requires_grad = any(k in n for k in ("S_residual", "U_residual", "V_residual"))
        self.backbone = m
        self.head = nn.Linear(m.embed_dim, 2)

    def forward(self, x):
        """x: (N, 3, 224, 224) frames -> (N, 2) logits."""
        return self.head(self.backbone.forward_features(x)[:, 0])

    def trainable_state_dict(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.state_dict().items()
                if any(s in k for s in ("S_residual", "U_residual", "V_residual")) or k.startswith("head")}


def video_logit_from_frames(frame_logits: torch.Tensor, n_frames: int) -> torch.Tensor:
    """(B * T, 2) frame logits -> (B,) logit of the mean frame probability (computed stably)."""
    z = (frame_logits[:, 1] - frame_logits[:, 0]).double().view(-1, n_frames)
    return torch.logsumexp(F.logsigmoid(z), 1) - torch.logsumexp(F.logsigmoid(-z), 1)
