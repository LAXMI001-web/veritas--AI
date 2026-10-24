"""Model evaluation on LABELED data supplied by the user (never on uploads). Expected layout:
  EVAL_DATA_DIR/<image|audio|video>/<test|val>/<real|fake>/<files>
Only the 'test' split is used for reported metrics; 'val' may be used to fit a temperature (FA3 logits). If no data is
present every metric is reported as unavailable. Positive class = fake (score high = fake)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import SETTINGS, Policy
from . import audio as audio_svc, video as video_svc
from .aggregate import aggregate
from .media import detect_kind, ffprobe, has_audio, MediaError

EXT = {"image": {".jpg", ".jpeg", ".png", ".webp", ".bmp"}, "audio": {".wav", ".mp3", ".flac", ".ogg", ".m4a"},
       "video": {".mp4", ".mov", ".mkv", ".avi", ".webm"}}


def scan(root: str | None = None) -> dict:
    root_p = Path(root or SETTINGS.eval_data_dir)
    out: dict = {}
    for mod in ("image", "audio", "video"):
        for split in ("val", "test"):
            for cls in ("real", "fake"):
                d = root_p / mod / split / cls
                files = sorted(p for p in d.glob("*") if p.suffix.lower() in EXT[mod]) if d.is_dir() else []
                out.setdefault(mod, {}).setdefault(split, {})[cls] = files
    return out


def counts(layout: dict) -> dict:
    return {m: {s: {c: len(f) for c, f in cs.items()} for s, cs in ss.items()} for m, ss in layout.items()}


def roc_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # Mann-Whitney U with tie handling
    allv = np.concatenate([pos, neg]); ranks = _rank(allv)
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def _rank(x):
    order = np.argsort(x, kind="mergesort"); ranks = np.empty(len(x)); i = 0
    xs = x[order]
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def roc_curve(y, s, n=60):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return []
    pts = []
    for t in np.unique(np.concatenate([[0, 1], s])):
        pred = s >= t
        tpr = float((pred & (y == 1)).sum() / (y == 1).sum()); fpr = float((pred & (y == 0)).sum() / (y == 0).sum())
        pts.append((round(fpr, 4), round(tpr, 4)))
    pts = sorted(set(pts))
    return pts[:: max(len(pts) // n, 1)] + [pts[-1]]


def metrics(y, s, thr=0.5, abstain_margin: float | None = None) -> dict:
    y = np.asarray(y).astype(int); s = np.asarray([np.nan if v is None else v for v in s], dtype=float)
    keep = ~np.isnan(s)
    y, s = y[keep], s[keep]
    n_total = int(keep.size)
    if len(y) == 0:
        return {"available": False, "reason": "no scored samples"}
    coverage_mask = np.ones(len(y), bool)
    if abstain_margin:
        coverage_mask = np.abs(s - thr) >= abstain_margin
    yy, ss = y[coverage_mask], s[coverage_mask]
    pred = ss >= thr
    tp = int((pred & (yy == 1)).sum()); tn = int((~pred & (yy == 0)).sum())
    fp = int((pred & (yy == 0)).sum()); fn = int((~pred & (yy == 1)).sum())
    div = lambda a, b: None if b == 0 else round(a / b, 4)  # noqa: E731
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {"available": True, "n": len(y), "n_failed_or_unscored": n_total - len(y), "coverage": round(float(coverage_mask.mean()), 4),
            "threshold": thr, "accuracy": div(tp + tn, len(yy)), "precision": prec, "recall": rec,
            "f1": None if not prec or not rec else round(2 * prec * rec / (prec + rec), 4),
            "fpr": div(fp, fp + tn), "fnr": div(fn, fn + tp), "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "roc_auc": None if abstain_margin else (None if (a := roc_auc(y, s)) is None else round(a, 4)),
            "roc": [] if abstain_margin else roc_curve(y, s)}


def fit_temperature(logit_diffs: np.ndarray, y: np.ndarray) -> float | None:
    """Temperature scaling on validation data. logit_diff = logit_fake - logit_real; y=1 fake."""
    if len(y) < 20 or len(set(y.tolist())) < 2:
        return None
    best, bt = 1e9, 1.0
    for T in np.exp(np.linspace(np.log(0.25), np.log(20), 120)):
        p = 1 / (1 + np.exp(-logit_diffs / T)); p = np.clip(p, 1e-6, 1 - 1e-6)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best:
            best, bt = nll, float(T)
    return round(bt, 3)


def run(pol: Policy, progress=None, root: str | None = None) -> dict:
    layout = scan(root)
    c = counts(layout)
    res: dict = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "dataset_counts": c, "policy": pol.__dict__,
                 "modalities": {}, "notes": []}
    total = sum(len(f) for m in layout.values() for s in m.values() for f in s.values())
    if total == 0:
        res["status"] = "unavailable"
        res["notes"].append("No labeled evaluation data found. Provide EVAL_DATA_DIR/<modality>/<test|val>/<real|fake>/ "
                            "(see docs/EVALUATION.md). No metric is reported without labeled data.")
        return res
    res["status"] = "ok"
    done = 0
    cache: dict = {}
    for mod in ("image", "audio", "video"):
        test = layout[mod]["test"]
        if not test["real"] and not test["fake"]:
            res["modalities"][mod] = {"available": False, "reason": "no test-split files"}
            continue
        ys, rows, lat = [], [], []
        for cls, y in (("real", 0), ("fake", 1)):
            for p in test[cls]:
                t0 = time.time()
                row = {"file": p.name, "y": y}
                try:
                    if mod == "image":
                        r = video_svc.analyze_image(p, pol); row["score"] = r.get("score") if r["status"] == "ok" else None
                        row["logit_diff"] = (r["frames"][0]["logits"][0] - r["frames"][0]["logits"][1]) if r["status"] == "ok" else None
                    elif mod == "audio":
                        r = audio_svc.analyze_audio_file(p, pol); row["score"] = r.get("score") if r["status"] == "ok" else None
                        row["win_scores"] = [w["score"] for w in r.get("windows", []) if w["score"] is not None]
                    else:
                        pr = ffprobe(p)
                        r = video_svc.analyze_video(p, pol, None)
                        row["frame_scores"] = [f["score"] for f in r.get("frames", []) if f["score"] is not None]
                        row["score"] = r.get("score") if r["status"] == "ok" else None
                        if has_audio(pr):
                            a = audio_svc.analyze_audio_file(p, pol, from_video=True)
                            row["audio_score"] = a.get("score") if a["status"] == "ok" else None
                except (MediaError, Exception) as exc:  # one bad file must not abort the run
                    row["score"] = None; row["error"] = type(exc).__name__
                row["latency_s"] = round(time.time() - t0, 3); lat.append(row["latency_s"])
                rows.append(row); ys.append(y); done += 1
                if progress: progress(done / total, f"{mod}: {p.name}")
        m = {"available": True, "primary": metrics(ys, [r["score"] for r in rows], pol.suspicious_threshold),
             "latency_s": {"mean": round(float(np.mean(lat)), 3), "p95": round(float(np.percentile(lat, 95)), 3)}}
        if mod == "image":
            val_rows = None
        if mod == "video":
            m["frame_aggregation"] = {}
            for meth in ("mean", "median", "fraction", "topk"):
                sc = [aggregate(r.get("frame_scores", []), meth, pol.frame_suspicious_cut, pol.topk) for r in rows]
                m["frame_aggregation"][meth] = metrics(ys, sc, 0.5 if meth != "fraction" else 0.3)
            m["note"] = "Aggregation methods are compared on the same test split; fraction uses threshold 0.3 on the suspicious-frame ratio."
            # combined-system ablation on videos that have audio
            both = [(r["y"], r["score"], r.get("audio_score")) for r in rows if r.get("audio_score") is not None and r["score"] is not None]
            if len(both) >= 10:
                yb = [b[0] for b in both]; vs = np.array([b[1] for b in both]); as_ = np.array([b[2] for b in both])
                thr = pol.suspicious_threshold
                m["ablation"] = {
                    "n_videos_with_audio": len(both),
                    "video_frames_only": metrics(yb, vs, thr),
                    "audio_only": metrics(yb, as_, thr),
                    "combined_mean_no_uncertainty": metrics(yb, (vs + as_) / 2, thr),
                    "combined_max_no_uncertainty": metrics(yb, np.maximum(vs, as_), thr),
                    "combined_mean_with_uncertainty_abstain": metrics(yb, (vs + as_) / 2, thr, abstain_margin=pol.inconclusive_margin),
                    "temporal_consistency_features": {"available": False, "reason": "No validated temporal-consistency feature exists yet."},
                }
            else:
                m["ablation"] = {"available": False, "reason": "Need ≥10 test videos with audio and scorable frames."}
        res["modalities"][mod] = m
    res["notes"].append("Metrics are computed only on the 'test' split. Aggregation/threshold choices must be selected on 'val', never 'test'.")
    return res
