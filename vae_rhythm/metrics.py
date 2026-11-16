"""Evaluation helpers: AUROC with ties counted half, per-generator tables, summaries.

Protocol used throughout the paper: for every benchmark, each generated subset is scored
against *all* real videos of that benchmark (one AUROC per generator), and a benchmark's
number is the mean over its generators. Paired 2AFC (DF26) compares each generated clip
with the real clip it was made from.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np


def auroc(neg: Sequence[float], pos: Sequence[float]) -> float:
    """Mann-Whitney AUROC in [0, 1]; ties count half; NaN if a class is empty."""
    neg, pos = np.asarray(neg, dtype=np.float64), np.asarray(pos, dtype=np.float64)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    values = np.concatenate([neg, pos])
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values))
    sv = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[len(neg):].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def paired_2afc(pairs: Iterable[tuple[float, float]]) -> float:
    """Share of (generated, real) pairs where the generated clip scores higher; ties count half."""
    p = [(a > b) + 0.5 * (a == b) for a, b in pairs]
    return float(np.mean(p)) if p else float("nan")


def read_scores(path: str, score_col: str = "score") -> dict[str, float]:
    with open(path) as fh:
        return {r["video_id"]: float(r[score_col]) for r in csv.DictReader(fh) if r.get(score_col) not in ("", None)}


def per_subset(rows: Sequence[dict], scores: dict[str, float]) -> list[dict]:
    """rows: manifest rows (video_id, label, subset[, group, pair_id]). One AUROC per
    generated subset against every real row that has a score."""
    have = [r for r in rows if r["video_id"] in scores]
    real = [r for r in have if int(r["label"]) == 0]
    real_s = np.array([scores[r["video_id"]] for r in real])
    by_pair = {r.get("pair_id"): scores[r["video_id"]] for r in real if r.get("pair_id")}
    groups = defaultdict(list)
    for r in have:
        if int(r["label"]) == 1:
            groups[r["subset"]].append(r)
    out = []
    for sub, g in sorted(groups.items()):
        gs = np.array([scores[r["video_id"]] for r in g])
        rec = {"subset": sub, "group": g[0].get("group", ""), "n_fake": len(g), "n_real": len(real),
               "AUROC": 100 * auroc(real_s, gs)}
        pairs = [(scores[r["video_id"]], by_pair[r["pair_id"]]) for r in g if r.get("pair_id") in by_pair]
        if pairs:
            rec["paired_2afc"] = 100 * paired_2afc(pairs)
        out.append(rec)
    return out


def summarize(recs: Sequence[dict]) -> dict:
    """Mean AUROC over generators, plus open / closed means when a ``group`` column exists
    (AIGVDBench: open_t2v / open_i2v / open_v2v / closed)."""
    if not recs:
        return {}
    a = np.array([r["AUROC"] for r in recs])
    out = {"mean_AUROC": float(a.mean()), "n_generators": len(recs)}
    op = [r["AUROC"] for r in recs if str(r.get("group", "")).startswith("open")]
    cl = [r["AUROC"] for r in recs if r.get("group") == "closed"]
    if op:
        out["open_AUROC"] = float(np.mean(op))
    if cl:
        out["closed_AUROC"] = float(np.mean(cl))
    p = [r["paired_2afc"] for r in recs if "paired_2afc" in r]
    if p:
        out["mean_2AFC"] = float(np.mean(p))
    return out
