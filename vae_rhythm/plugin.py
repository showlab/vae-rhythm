"""VAE Rhythm plug-in (paper Sec. 4.3, Eq. 3).

    s = z(s_d) + max(0, z(s_c)),    s_c = Ppk

Both standardisations use real videos only (the paper calibrates on AIGVDBench validation
reals and reuses the calibration on every benchmark):

* z(s_c) = (Ppk - mu) / sd over real videos scanned with the same protocol.
* z(s_d) is the *hybrid* standardisation of a detector score: inside the range of the
  real-video scores it is the standard-normal quantile of the score's mid-rank among them;
  beyond that range it continues linearly with slope 1/sigma (sigma = std of the real
  scores). This keeps saturated detectors (most real scores below 1e-3) usable and keeps
  the detector's ordering of clips scored above every real video.

The positive part adds rhythm evidence when it is strong without treating its absence as
evidence of authenticity: low motion, resampling and rescaling all weaken the rhythm.
"""
from __future__ import annotations

from statistics import NormalDist
from typing import Sequence

import numpy as np

_ND = NormalDist()


class HybridStandardizer:
    """z(s_d) of Eq. 3 fitted on detector scores of real videos."""

    def __init__(self, real_scores: Sequence[float]):
        r = np.sort(np.asarray(real_scores, dtype=np.float64))
        r = r[np.isfinite(r)]
        if len(r) < 2:
            raise ValueError("Need at least two real-video scores.")
        self.sorted, self.n = r, len(r)
        self.sigma = float(r.std(ddof=1)) or 1e-6
        self.lo, self.hi = float(r[0]), float(r[-1])
        self.z_lo = _ND.inv_cdf(0.5 / (self.n + 1))
        self.z_hi = _ND.inv_cdf((self.n + 0.5) / (self.n + 1))

    def rank_z(self, s) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        mid = (np.searchsorted(self.sorted, s, "left") + np.searchsorted(self.sorted, s, "right")) / 2
        q = (mid + 0.5) / (self.n + 1)
        return np.vectorize(_ND.inv_cdf)(q) if q.ndim else np.float64(_ND.inv_cdf(float(q)))

    def __call__(self, s) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        z = np.asarray(self.rank_z(s), dtype=np.float64)
        z = np.where(s > self.hi, self.z_hi + (s - self.hi) / self.sigma, z)
        z = np.where(s < self.lo, self.z_lo - (self.lo - s) / self.sigma, z)
        return z

    def to_dict(self) -> dict:
        return {"real_scores": self.sorted.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "HybridStandardizer":
        return cls(d["real_scores"])


class MeanStdStandardizer:
    """Plain (s - mean) / std on real videos (the paper's original rule, kept for comparison)."""

    def __init__(self, real_scores: Sequence[float]):
        r = np.asarray(real_scores, dtype=np.float64)
        r = r[np.isfinite(r)]
        self.mu, self.sigma = float(r.mean()), float(r.std(ddof=1)) or 1e-6

    def __call__(self, s) -> np.ndarray:
        return (np.asarray(s, dtype=np.float64) - self.mu) / self.sigma


def z_ppk(ppk, mu: float, sd: float) -> np.ndarray:
    return (np.asarray(ppk, dtype=np.float64) - mu) / sd


def fuse(z_detector, z_rhythm, tau: float = 0.0) -> np.ndarray:
    """Eq. 3: z(s_d) + max(0, z(s_c) - tau); tau = 0 in the paper."""
    return np.asarray(z_detector, dtype=np.float64) + np.clip(np.asarray(z_rhythm, dtype=np.float64) - tau, 0, None)


def apply_plugin(det_scores, ppk, det_real_scores, ppk_mean: float, ppk_std: float,
                 norm: str = "hybrid", tau: float = 0.0) -> np.ndarray:
    """Fused scores for aligned arrays of detector scores and Ppk values."""
    standardize = HybridStandardizer(det_real_scores) if norm == "hybrid" else MeanStdStandardizer(det_real_scores)
    return fuse(standardize(det_scores), z_ppk(ppk, ppk_mean, ppk_std), tau)


def threshold_at_fpr(real_scores: Sequence[float], fpr: float = 0.05) -> float:
    """Operating point: the (1 - fpr) quantile of real-video scores (higher = more likely generated)."""
    return float(np.quantile(np.asarray(real_scores, dtype=np.float64), 1 - fpr, method="higher"))
