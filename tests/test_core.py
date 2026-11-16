"""CPU tests for the training-free parts. Run: ``pytest -q``."""
import math
import shutil

import numpy as np
import pytest

from vae_rhythm import stats as S
from vae_rhythm.metrics import auroc, paired_2afc, per_subset, summarize
from vae_rhythm.plugin import HybridStandardizer, fuse, threshold_at_fpr, z_ppk
from vae_rhythm.scan import plan_segments


def _table(period=None, K=24, W=13, seed=0, depth=0.8):
    rng = np.random.default_rng(seed)
    D = 1.0 + 0.5 * rng.random(K - 1 + W) + 0.02 * np.arange(K - 1 + W)   # content difficulty + drift
    f = (lambda k: depth if k % period == 0 else 1.0) if period else (lambda k: 1.0)
    E = np.array([[D[k + j] * f(k) for j in range(W)] for k in range(K)])
    return E + 0.005 * rng.standard_normal((K, W))


def test_periodic_vs_flat():
    s4, s0 = S.phase_stats(_table(4)), S.phase_stats(_table(None))
    assert s4["C4"] > 0.9 and s4["dE4"] > 0.1 and s4["phase4"] == 0
    assert s0["C4"] < 0.5 and abs(s0["dE4"]) < 0.02
    assert abs(s4["ppk"] - 4) < 0.35 or abs(s4["ppk"] - 2) < 0.35
    assert s4["Ppk"] > s0["Ppk"]


def test_period_8_and_5():
    s8 = S.phase_stats(_table(8, depth=0.7))
    assert s8["C8"] > 0.8 and s8["X8"] > 0.2
    pk, per = S.padded_peak(1.0 + 0.3 * (np.arange(24) % 5 != 0))
    assert abs(per - 5) < 0.35 and pk > 0.3


def test_parseval_and_curve():
    fr = S.periodogram_fraction(np.random.default_rng(1).standard_normal(24))
    assert math.isclose(sum(fr.values()), 1.0, abs_tol=1e-9)
    r = S.residual_curve(_table(4))
    assert r.shape == (24,) and abs(r.mean()) < 1e-12


def test_phase_concentration():
    assert S.phase_concentration([0.3] * 10) > 0.999
    assert S.phase_concentration(np.linspace(0, 2 * np.pi, 400, endpoint=False)) < 1e-6


def test_auroc_ties_and_2afc():
    assert auroc([0, 0], [1, 1]) == 1.0
    assert auroc([1, 0], [1, 0]) == 0.5
    rng = np.random.default_rng(0)
    neg, pos = rng.normal(0, 1, 300), rng.normal(1, 1, 200)
    sk = pytest.importorskip("sklearn.metrics")
    y = np.r_[np.zeros(300), np.ones(200)]
    assert math.isclose(auroc(neg, pos), sk.roc_auc_score(y, np.r_[neg, pos]), abs_tol=1e-12)
    assert paired_2afc([(2, 1), (1, 1), (0, 1)]) == 0.5


def test_per_subset_summary():
    rows = [{"video_id": f"r{i}", "label": "0", "subset": "Real"} for i in range(5)] + \
           [{"video_id": f"f{i}", "label": "1", "subset": "Gen", "group": "closed"} for i in range(5)]
    sc = {f"r{i}": i for i in range(5)} | {f"f{i}": 10 + i for i in range(5)}
    recs = per_subset(rows, sc)
    assert recs[0]["AUROC"] == 100.0 and summarize(recs)["closed_AUROC"] == 100.0


def _reference_hybrid(v, real):
    """Direct transcription of the paper's rule for comparison."""
    from statistics import NormalDist
    nd, srt, n = NormalDist(), np.sort(real), len(real)
    sd = np.std(real, ddof=1)
    if v > srt[-1]:
        return nd.inv_cdf((n + 0.5) / (n + 1)) + (v - srt[-1]) / sd
    if v < srt[0]:
        return nd.inv_cdf(0.5 / (n + 1)) - (srt[0] - v) / sd
    mid = (np.searchsorted(srt, v, "left") + np.searchsorted(srt, v, "right")) / 2
    return nd.inv_cdf((mid + 0.5) / (n + 1))


def test_hybrid_standardizer():
    rng = np.random.default_rng(0)
    real = np.concatenate([np.full(50, 1e-4), rng.random(150) * 1e-2])      # saturated detector with ties
    h = HybridStandardizer(real)
    for v in [-1.0, 0.0, 1e-4, 5e-3, real.max(), 0.5, 3.0]:
        assert math.isclose(float(h(np.array([v]))[0]), _reference_hybrid(v, real), rel_tol=1e-9, abs_tol=1e-9)
    z = h(np.array([0.2, 0.4, 0.9]))
    assert np.all(np.diff(z) > 0)                                            # order kept above the real range


def test_fuse_positive_part():
    zd = np.array([0.0, 1.0, -1.0])
    zc = np.array([-3.0, 2.0, 0.5])
    assert np.allclose(fuse(zd, zc), [0.0, 3.0, -0.5])
    assert np.allclose(z_ppk([0.3, 0.44], 0.3, 0.14), [0.0, 1.0])
    assert threshold_at_fpr(np.arange(100), 0.05) == 95


def test_plan_segments():
    assert plan_segments(36) == [0]
    assert plan_segments(121) == [0, 40, 80]
    starts = plan_segments(1000)
    assert len(starts) == 3 and all(s % 8 == 0 for s in starts) and starts[-1] + 36 <= 1000
    with pytest.raises(ValueError):
        plan_segments(30)


class _FakeVAE:
    """Reconstruction error lower on frames aligned to a 4-frame lattice (no torch model needed)."""
    name, t_step, s_factor, device = "fake", 4, 8, "cpu"

    def valid_T(self, T):
        return ((T - 1) // 4) * 4 + 1

    def clear_cache(self):
        pass

    def roundtrip(self, x):
        import torch
        y = x.clone()
        T = x.shape[2]
        for t in range(T):
            y[:, :, t] += (0.01 if t % 4 else 0.004) * torch.sin(torch.arange(x.shape[-1]).float())
        return y


def test_scan_frames_with_fake_vae():
    pytest.importorskip("torch")
    from vae_rhythm.scan import scan_frames

    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, size=(36, 16, 16, 3), dtype=np.uint8)
    E = scan_frames(_FakeVAE(), frames, K=24, W=13, batch=3)
    assert E.shape == (24, 13)
    # the fake error depends on the window position only, so the residual has no offset structure
    assert S.phase_stats(E)["Ppk"] >= 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_chain_runs(tmp_path):
    pytest.importorskip("av")
    from vae_rhythm.engine.chain import run_chain, sample_chain
    from vae_rhythm.video import read_frames

    p = sample_chain(np.random.default_rng([0, 1]), 64, 24.0, (48, 64), False, 0.0, 0.0)
    p2 = sample_chain(np.random.default_rng([0, 1]), 64, 24.0, (48, 64), False, 0.0, 0.0)
    assert p == p2
    frames = np.random.default_rng(0).integers(0, 255, size=(41, 64, 96, 3), dtype=np.uint8)
    out = tmp_path / "x.mp4"
    run_chain(frames, 24.0, p, str(out))
    got = read_frames(out)
    assert min(got.shape[1:3]) == p["S"] and len(got) >= 41 - 4 - (41 * 8 // 24) - 2
