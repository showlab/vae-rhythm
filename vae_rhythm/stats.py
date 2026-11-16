"""Statistics of a phase-scan error table (pure NumPy).

Input: an error table ``E`` of shape (K, W). Row k holds the per-frame reconstruction
error (RGB MSE in [-1, 1]) of the W-frame window that starts at offset k. The paper's
default is K = 24 (divisible by 2, 3, 4, 6, 8 and 12) and W = 13.

Main statistics (paper Sec. 3.1) are computed on the per-frame baseline residual

    r(k) = 1 / ((W - 1) * Ebar) * sum_{j=1}^{W-1} [ e(k, j) - ebar(k + j) ],

where ebar(t) is the mean error of absolute frame t over every window that covers it
(window position j = 0 is excluded because a causal VAE encodes the first frame of a
window on its own) and Ebar is the mean of the whole table.

    C{p}   fraction of the residual variance in the period-p harmonic comb
           (fundamental plus harmonics; C4 uses the bins of periods 4 and 2)
    P{p}   fraction in the fundamental bin of period p only
    X8     C8 - C4, the part that only a period of 8 explains
    Ppk    period-agnostic score: the largest normalised periodogram bin after 4x zero
           padding, restricted to periods [2.5, 9]; ppk is the period of that bin
    dE4/8  alignment gain (mean residual off the lattice minus on it) / Ebar
    phase4/8  offset class with the lowest mean residual (a descriptive quantity)

Suffix ``_raw``: the same comb statistics on the raw curve E(k) = mean_j E[k, j]
(kept as a reference; the paper uses the residual versions).
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

PERIODS = (2, 3, 4, 6, 8, 12)


def periodogram_fraction(x) -> dict[int, float]:
    """Mean-removed sequence (K,) -> share of variance per positive frequency bin j = 1..K//2."""
    K = len(x)
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    tot = float((x ** 2).sum())
    if tot <= 0:
        return {j: 0.0 for j in range(1, K // 2 + 1)}
    X = np.fft.fft(x)
    out = {}
    for j in range(1, K // 2 + 1):
        w = 1.0 if 2 * j == K else 2.0           # the Nyquist bin has no conjugate partner
        out[j] = float(w * abs(X[j]) ** 2 / (K * tot))
    return out


def comb_fraction(frac: dict[int, float], K: int, p: int) -> float:
    """Harmonic comb of period p: bins j = m*K/p, m = 1..p//2 (requires K % p == 0)."""
    if K % p != 0:
        return float("nan")
    return float(sum(frac[m * K // p] for m in range(1, p // 2 + 1)))


def padded_peak(x, pad: int = 4, pmin: float = 2.5, pmax: float = 9.0) -> tuple[float, float]:
    """Period-agnostic peak: zero-pad the mean-removed sequence by ``pad`` and return
    (largest normalised periodogram bin with period in [pmin, pmax], its period).

    Zero padding only refines the frequency grid (periods such as 5 or 7 that do not
    divide K become reachable); it adds no information. The upper bound 9 avoids periods
    with fewer than ~3 cycles in K = 24 offsets, where slow drift leaks in; the lower
    bound 2.5 excludes period-2 patterns from frame dropping or codec pyramids.
    """
    K = len(x)
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    tot = float((x ** 2).sum())
    if tot <= 0:
        return 0.0, 0.0
    N = K * pad
    X = np.abs(np.fft.rfft(x, n=N)) ** 2 / (K * tot) * 2.0
    j = np.arange(1, N // 2 + 1)
    per = N / j
    band = (per >= pmin) & (per <= pmax)
    jb = j[band]
    if len(jb) == 0:
        return 0.0, 0.0
    k = int(jb[np.argmax(X[jb])])
    return float(X[k]), float(N / k)


def frame_baseline_residual(E, j0: int = 1) -> np.ndarray:
    """E (K, W) -> E' (K, W - j0): subtract the mean error of each absolute frame t = k + j
    (estimated from the observations with j >= j0 only)."""
    E = np.asarray(E, dtype=np.float64)
    K, W = E.shape
    T = K - 1 + W
    F, n = np.zeros(T), np.zeros(T)
    for k in range(K):
        F[k + j0:k + W] += E[k, j0:]
        n[k + j0:k + W] += 1
    F = F / np.maximum(n, 1)
    R = np.empty((K, W - j0))
    for k in range(K):
        R[k] = E[k, j0:] - F[k + j0:k + W]
    return R


def _align_gain(x, p: int) -> float:
    x = np.asarray(x, dtype=np.float64)
    k = np.arange(len(x))
    m = x.mean()
    if m <= 0:
        return 0.0
    return float((x[k % p != 0].mean() - x[k % p == 0].mean()) / m)


def _best_phase(x, p: int) -> int:
    x = np.asarray(x, dtype=np.float64)
    k = np.arange(len(x))
    return int(np.argmin([x[k % p == r].mean() for r in range(p)]))


def curve_stats(Ek, suffix: str = "") -> dict:
    """Comb statistics of a raw curve E(k) (used for the ``_raw`` reference values)."""
    Ek = np.asarray(Ek, dtype=np.float64)
    K = len(Ek)
    frac = periodogram_fraction(Ek)
    out = {}
    for p in PERIODS:
        out[f"P{p}{suffix}"] = frac.get(K // p, float("nan")) if K % p == 0 else float("nan")
        out[f"C{p}{suffix}"] = comb_fraction(frac, K, p)
    out[f"X8{suffix}"] = out[f"C8{suffix}"] - out[f"C4{suffix}"]
    out[f"dE4{suffix}"] = _align_gain(Ek, 4)
    out[f"dE8{suffix}"] = _align_gain(Ek, 8)
    out[f"phase4{suffix}"] = _best_phase(Ek, 4)
    out[f"phase8{suffix}"] = _best_phase(Ek, 8)
    out[f"E0{suffix}"] = float(Ek[0])
    out[f"Emean{suffix}"] = float(Ek.mean())
    return out


def residual_stats(R, Emean: float, suffix: str = "") -> dict:
    """Statistics of the residual table R (K, W-1); alignment gains are normalised by the
    mean raw error so they keep the meaning of a relative error."""
    rk = np.asarray(R, dtype=np.float64).mean(1)
    K = len(rk)
    frac = periodogram_fraction(rk)
    out = {}
    for p in PERIODS:
        out[f"P{p}{suffix}"] = frac.get(K // p, float("nan")) if K % p == 0 else float("nan")
        out[f"C{p}{suffix}"] = comb_fraction(frac, K, p)
    out[f"X8{suffix}"] = out[f"C8{suffix}"] - out[f"C4{suffix}"]
    k = np.arange(K)
    for p in (4, 8):
        out[f"dE{p}{suffix}"] = float((rk[k % p != 0].mean() - rk[k % p == 0].mean()) / Emean) if Emean > 0 else 0.0
        out[f"phase{p}{suffix}"] = int(np.argmin([rk[k % p == r].mean() for r in range(p)]))
    out[f"amp4{suffix}"] = float(rk[k % 4 != 0].mean() - rk[k % 4 == 0].mean())
    jbest = max((j for j in frac if j >= 2), key=lambda j: frac[j]) if len(frac) > 1 else 1
    out[f"pmax{suffix}"] = float(K / jbest)
    out[f"Pmax{suffix}"] = float(frac[jbest]) if jbest in frac else 0.0
    # alignment gain without assuming the aligned phase is 0 (min over the four classes)
    m4 = np.array([rk[k % 4 == r].mean() for r in range(4)])
    b = int(np.argmin(m4))
    out[f"dE4x{suffix}"] = float((np.delete(m4, b).mean() - m4[b]) / Emean) if Emean > 0 else 0.0
    out[f"Ppk{suffix}"], out[f"ppk{suffix}"] = padded_peak(rk)
    return out


def phase_stats(E) -> dict:
    """Error table E (K, W) -> dict of statistics. Unsuffixed keys use the per-frame
    baseline residual (the paper's definitions); ``_raw`` keys use the raw curve."""
    E = np.asarray(E, dtype=np.float64)
    Ek = E.mean(1)
    Emean = float(Ek.mean())
    s = residual_stats(frame_baseline_residual(E, 1), Emean) if E.shape[1] > 2 else {}
    s.update(curve_stats(Ek, "_raw"))
    s["E0"] = float(Ek[0])
    s["Emean"] = Emean
    return s


def residual_curve(E) -> np.ndarray:
    """The dimensionless residual profile r(k) of paper Eq. (1), centred to zero mean."""
    E = np.asarray(E, dtype=np.float64)
    if E.ndim != 2 or not np.all(np.isfinite(E)) or np.any(E < 0):
        raise ValueError("Expected a finite non-negative (K, W) error table.")
    r = frame_baseline_residual(E, 1).mean(1) / (E.mean() + 1e-12)
    return r - r.mean()


def phase_angle(curve, period: int = 4) -> float:
    """Argument of the period-``period`` Fourier coefficient of a curve (radians)."""
    x = np.asarray(curve, dtype=np.float64)
    x = x - x.mean()
    k = np.arange(len(x))
    return float(np.angle(np.sum(x * np.exp(-2j * np.pi * k / period))))


def phase_concentration(angles: Iterable[float]) -> float:
    """R = |mean_i exp(i * phi_i)| in [0, 1]: 0 for dispersed phases, 1 for identical ones."""
    a = np.asarray(list(angles), dtype=np.float64)
    if len(a) == 0:
        return float("nan")
    return float(abs(np.mean(np.exp(1j * a))))


def spectrum(curve, pmin: float = 2.5, pmax: float = 12.0) -> dict:
    """Normalised power of a curve over periods [pmin, pmax] (4x zero padding), for plots."""
    x = np.asarray(curve, dtype=np.float64)
    x = x - x.mean()
    total = float(np.square(x).sum())
    size = 4 * len(x)
    power = np.zeros(size // 2 + 1) if total <= 1e-24 else 2 * np.abs(np.fft.rfft(x, n=size)) ** 2 / (len(x) * total)
    periods = size / np.arange(1, len(power))
    band = (periods >= pmin) & (periods <= pmax)
    periods, power = periods[band], power[1:][band]
    order = np.argsort(periods)
    return {"periods": periods[order].tolist(), "power": power[order].tolist()}


__all__ = [
    "PERIODS", "periodogram_fraction", "comb_fraction", "padded_peak", "frame_baseline_residual",
    "curve_stats", "residual_stats", "phase_stats", "residual_curve", "phase_angle",
    "phase_concentration", "spectrum",
]


def _self_test() -> None:  # pragma: no cover - exercised by tests/test_stats.py
    rng = np.random.default_rng(0)
    K, W = 24, 33
    D = 1.0 + 0.5 * rng.random(K - 1 + W) + 0.02 * np.arange(K - 1 + W)
    Ef = np.array([[D[k + j] * (0.8 if k % 4 == 0 else 1.0) for j in range(W)] for k in range(K)])
    sf = phase_stats(Ef + 0.005 * rng.standard_normal((K, W)))
    assert sf["C4"] > 0.9 and sf["dE4"] > 0.1 and sf["phase4"] == 0
    Er = np.array([[D[k + j] for j in range(W)] for k in range(K)])
    sr = phase_stats(Er + 0.005 * rng.standard_normal((K, W)))
    assert sr["C4"] < 0.5 and abs(sr["dE4"]) < 0.02
    assert math.isclose(sum(periodogram_fraction(rng.standard_normal(K)).values()), 1.0, abs_tol=1e-9)


if __name__ == "__main__":  # pragma: no cover
    _self_test()
    print("stats self-test ok")
