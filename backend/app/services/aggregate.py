"""Score aggregation + suspicious-interval detection (pure functions, unit-testable)."""
from __future__ import annotations

import numpy as np


def aggregate(scores: list[float], method: str, cut: float = 0.5, topk: int = 3) -> float | None:
    s = np.asarray([x for x in scores if x is not None], dtype=float)
    if s.size == 0:
        return None
    if method == "mean":
        return float(s.mean())
    if method == "median":
        return float(np.median(s))
    if method == "fraction":
        return float((s >= cut).mean())
    if method == "topk":
        k = min(max(topk, 1), s.size)
        return float(np.sort(s)[-k:].mean())
    raise ValueError(f"unknown aggregation: {method}")


def effective_k(n: int, topk: int = 3, frac: float = 0.0) -> int:
    """Top-k size: at least `topk`, and at least `frac` of the scored items (never more than n)."""
    import math
    return max(1, min(n, max(topk, math.ceil(frac * n))))


def all_aggregates(scores, cut=0.5, topk=3) -> dict:
    return {m: (None if (v := aggregate(scores, m, cut, topk)) is None else round(v, 4))
            for m in ("mean", "median", "fraction", "topk")}


def suspicious_intervals(items: list[dict], min_score: float, min_len: int, span_key=("start", "end")) -> list[dict]:
    """Group consecutive scored items with score >= min_score. Items need 'score' (None = unscored breaks nothing
    but is skipped). Returns intervals with start/end/n/max/mean. These are sampled-evidence intervals, not
    precise manipulation boundaries."""
    out, cur = [], []
    for it in items:
        sc = it.get("score")
        if sc is None:
            continue
        if sc >= min_score:
            cur.append(it)
        else:
            if len(cur) >= min_len:
                out.append(cur)
            cur = []
    if len(cur) >= min_len:
        out.append(cur)
    res = []
    for grp in out:
        sc = [g["score"] for g in grp]
        res.append({"start": grp[0][span_key[0]], "end": grp[-1][span_key[1]], "n": len(grp),
                    "max": round(max(sc), 4), "mean": round(float(np.mean(sc)), 4)})
    return res


def dispersion(scores: list[float]) -> dict:
    s = np.asarray([x for x in scores if x is not None], dtype=float)
    if s.size < 2:
        return {"std": None, "min": None, "max": None, "n": int(s.size)}
    return {"std": round(float(s.std()), 4), "min": round(float(s.min()), 4), "max": round(float(s.max()), 4),
            "n": int(s.size)}
