"""VAE Rhythm: temporal decoding traces for AI-generated video detection.

Training-free phase scan, the VAE Rhythm plug-in, the VAE-pair engine and the CaDet detector.
Heavy dependencies (torch, diffusers, timm) are imported lazily, so ``vae_rhythm.stats`` and
``vae_rhythm.plugin`` work with NumPy alone.
"""
__version__ = "1.0.0"

from .stats import phase_stats, residual_curve  # noqa: F401


def scan_video(*args, **kwargs):
    """See :func:`vae_rhythm.scan.scan_video`."""
    from .scan import scan_video as _scan_video

    return _scan_video(*args, **kwargs)
