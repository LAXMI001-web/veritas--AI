"""Pick the audio model + preprocessing + polarity + calibration + threshold from LABELLED data, then report held-out
test metrics once. Used by scripts/select_audio_model.py (and unit-tested).

Protocol (no tuning on the test set):
  * candidates = every model in models/audio/ (+ the legacy model) x preprocessing variants (norm x n_fft)
  * on the VALIDATION split: AUC per candidate; polarity is decided from the data (AUC < 0.5 => high output means REAL);
    best candidate by validation AUC (ties: healthy models first). Platt calibration + decision threshold (Youden J) are
    also fitted on the validation split only.
  * the TEST split is evaluated once with the chosen candidate: accuracy, precision, recall, F1, FPR, FNR, confusion matrix, AUC.
  * `validated=True` is written only if the test split is big enough and test AUC >= min_auc."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import subprocess
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..config import ROOT
from ..models.audio_specs import AudioSpec, NORMS, candidate_specs, window_features
from .evaluation import roc_auc

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi"}
SILENCE_DBFS = -55.0


def load_clip_16k(path: Path) -> np.ndarray | None:
    import librosa
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"
        p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                            "-c:a", "pcm_s16le", "-t", "60", str(wav)], capture_output=True, timeout=120)
        if p.returncode != 0 or not wav.exists():
            return None
        a, _ = librosa.load(str(wav), sr=16000, mono=True)
    return a.astype(np.float32) if a.size else None


def collect(root: Path) -> dict:
    """root/{real,fake}/* (flat)  or  root/{val,test}/{real,fake}/*. -> {"flat": {...}} or {"val": ..., "test": ...}"""
    def files(d: Path):
        return sorted(p for p in d.glob("**/*") if p.is_file() and p.suffix.lower() in AUDIO_EXT | VIDEO_EXT) if d.is_dir() else []
    if (root / "val").is_dir() and (root / "test").is_dir():
        return {s: {"real": files(root / s / "real"), "fake": files(root / s / "fake")} for s in ("val", "test")}
    return {"flat": {"real": files(root / "real"), "fake": files(root / "fake")}}


def split_flat(flat: dict, group_regex: str | None, seed: int = 0) -> tuple[dict, dict, list[str]]:
    """Deterministic 50/50 split by group (default group = file). Keeps all files of a group (e.g. one speaker) together."""
    notes = []
    rx = re.compile(group_regex) if group_regex else None
    if rx is None:
        notes.append("No --group-regex: every file is split independently. If clips from the same speaker/source video appear in both "
                     "validation and test, the reported test metrics will be optimistic.")
    val = {"real": [], "fake": []}
    test = {"real": [], "fake": []}
    for cls in ("real", "fake"):
        for f in flat[cls]:
            g = (rx.search(f.name).group(1) if rx and rx.search(f.name) else f.name)
            h = int(hashlib.sha1(f"{seed}:{g}".encode()).hexdigest(), 16) % 2
            (val if h == 0 else test)[cls].append(f)
    return val, test, notes


def windows_of(audio: np.ndarray, spec: AudioSpec, max_windows: int) -> list[np.ndarray]:
    W = spec.window_samples
    if len(audio) <= W:
        segs = [np.pad(audio, (0, W - len(audio)))]
    else:
        starts = np.unique(np.linspace(0, len(audio) - W, min(max_windows, 1 + (len(audio) - W) // W)).astype(int))
        segs = [audio[s:s + W] for s in starts]
    keep = [g for g in segs if 20 * math.log10(float(np.sqrt(np.mean(np.square(g)))) + 1e-12) > SILENCE_DBFS]
    return keep


def variants_for(base: AudioSpec, quick: bool = False) -> list[AudioSpec]:
    if base.key.endswith("scamguard"):
        return [base]                                   # legacy model has one known preprocessing
    norms = ["db80", "db", "minmax", "zscore", "db_ref1", "log"]
    ffts = [2048] if quick else [2048, 1024]
    return [replace(base, norm=n, n_fft=f) for n, f in itertools.product(norms, ffts)]


def raw_scores(model, spec: AudioSpec, clips: list[np.ndarray], max_windows: int) -> np.ndarray:
    """Clip-level raw score = mean of window sigmoid outputs (NaN when the clip had no non-silent window)."""
    out = np.full(len(clips), np.nan)
    for i, a in enumerate(clips):
        segs = windows_of(a, spec, max_windows)
        if not segs:
            continue
        x = np.stack([window_features(g, spec) for g in segs])
        out[i] = float(np.mean(np.asarray(model.predict(x, verbose=0)).reshape(-1)))
    return out


def platt_fit(p_oriented: np.ndarray, y: np.ndarray, iters: int = 100) -> dict:
    """Fit p_fake = sigmoid(a*logit(p)+b) by Newton's method (tiny L2 for stability). y: 1 = fake."""
    q = np.clip(p_oriented, 1e-6, 1 - 1e-6)
    z = np.log(q / (1 - q))
    z = np.clip(z, -12, 12)
    a, b = 0.0, 0.0
    # a starts at 0 so a collapsed model gets a flat (base-rate) calibration instead of an overconfident one
    for _ in range(iters):
        s = 1 / (1 + np.exp(-(a * z + b)))
        g = np.array([np.sum((s - y) * z) + 1e-3 * a, np.sum(s - y)])
        w = s * (1 - s) + 1e-9
        H = np.array([[np.sum(w * z * z) + 1e-3, np.sum(w * z)], [np.sum(w * z), np.sum(w)]])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return {"a": float(a), "b": float(b)}


def apply_platt(p_oriented: np.ndarray, cal: dict) -> np.ndarray:
    q = np.clip(p_oriented, 1e-6, 1 - 1e-6)
    z = np.clip(np.log(q / (1 - q)), -12, 12)
    return 1 / (1 + np.exp(-(cal["a"] * z + cal["b"])))


def youden_threshold(score: np.ndarray, y: np.ndarray) -> float:
    cand = np.unique(np.concatenate([score, [0.5]]))
    best, bt = -2.0, 0.5
    for t in cand:
        pred = score >= t
        tpr = np.mean(pred[y == 1]) if np.any(y == 1) else 0
        fpr = np.mean(pred[y == 0]) if np.any(y == 0) else 0
        j = tpr - fpr
        if j > best + 1e-12 or (abs(j - best) <= 1e-12 and abs(t - 0.5) < abs(bt - 0.5)):
            best, bt = j, float(t)
    return bt


def metrics_at(score: np.ndarray, y: np.ndarray, thr: float) -> dict:
    pred = (score >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1))); tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0))); fn = int(np.sum((pred == 0) & (y == 1)))
    div = lambda a, b: None if b == 0 else round(a / b, 4)   # noqa: E731
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    f1 = None if prec is None or rec is None or (prec + rec) == 0 else round(2 * prec * rec / (prec + rec), 4)
    return {"n": int(len(y)), "threshold": round(float(thr), 4), "accuracy": div(tp + tn, len(y)), "precision": prec, "recall": rec,
            "f1": f1, "false_positive_rate": div(fp, fp + tn), "false_negative_rate": div(fn, fn + tp),
            "confusion_matrix": {"tp_fake_caught": tp, "fn_fake_missed_as_real": fn, "fp_real_flagged_fake": fp, "tn_real_ok": tn},
            "auc": None if roc_auc(y, score) is None else round(roc_auc(y, score), 4), "positive_class": "fake"}


def run_selection(data_dir: Path, out_file: Path | None = None, group_regex: str | None = None, max_windows: int = 8,
                  quick: bool = False, min_auc: float = 0.75, min_per_class: int = 10, models: list[str] | None = None,
                  log=print) -> dict:
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    layout = collect(Path(data_dir))
    notes: list[str] = []
    if "flat" in layout:
        val_f, test_f, notes = split_flat(layout["flat"], group_regex)
    else:
        val_f, test_f = layout["val"], layout["test"]
    def load(split):
        clips, ys, names = [], [], []
        for cls, y in (("real", 0), ("fake", 1)):
            for f in split[cls]:
                a = load_clip_16k(f)
                if a is None:
                    notes.append(f"unreadable: {f.name}")
                    continue
                clips.append(a); ys.append(y); names.append(f.name)
        return clips, np.array(ys), names
    vc, vy, _ = load(val_f); tc, ty, _ = load(test_f)
    log(f"validation clips: {int(np.sum(vy == 0))} real / {int(np.sum(vy == 1))} fake;  test clips: {int(np.sum(ty == 0))} real / {int(np.sum(ty == 1))} fake")
    if len(vy) < 60:
        notes.append(f"Only {len(vy)} validation clips for choosing among many candidate models/preprocessings: the choice is likely "
                     "to be noise (selection overfits small validation sets). Use a few hundred clips for a trustworthy choice.")
        log("WARNING: " + notes[-1])
    if min(np.sum(vy == 0), np.sum(vy == 1), np.sum(ty == 0), np.sum(ty == 1)) == 0:
        raise SystemExit("Need at least one real and one fake clip in BOTH validation and test splits.")
    cands = candidate_specs()
    if models:
        cands = {k: v for k, v in cands.items() if k in models}
    table = []
    for key, base in cands.items():
        mp = Path(base.model_file) if Path(base.model_file).is_absolute() else ROOT / base.model_file
        model = tf.keras.models.load_model(str(mp), compile=False)
        for sp in variants_for(base, quick):
            sc = raw_scores(model, sp, vc, max_windows)
            ok = ~np.isnan(sc)
            auc = roc_auc(vy[ok], sc[ok]) if ok.sum() else None
            if auc is None:
                continue
            hm = "fake" if auc >= 0.5 else "real"
            table.append({"model": key, "norm": sp.norm, "n_fft": sp.n_fft, "val_auc": round(max(auc, 1 - auc), 4), "high_means": hm,
                          "val_auc_raw": round(auc, 4), "_spec": sp, "_model_path": mp})
            log(f"  {key:30s} norm={sp.norm:8s} n_fft={sp.n_fft:4d}  val AUC={max(auc, 1 - auc):.3f}  (high output = {hm})")
    if not table:
        raise SystemExit("No candidate produced scores (is models/audio/ populated?).")
    table.sort(key=lambda r: -r["val_auc"])
    best = table[0]
    sp: AudioSpec = best["_spec"]
    sp = replace(sp, high_means=best["high_means"])
    model = tf.keras.models.load_model(str(best["_model_path"]), compile=False)
    vs = raw_scores(model, sp, vc, max_windows); ok = ~np.isnan(vs)
    p_or = vs[ok] if sp.high_means == "fake" else 1 - vs[ok]
    cal = platt_fit(p_or, vy[ok])
    vcal = apply_platt(p_or, cal)
    thr = youden_threshold(vcal, vy[ok])
    ts = raw_scores(model, sp, tc, max_windows); tok = ~np.isnan(ts)
    t_or = ts[tok] if sp.high_means == "fake" else 1 - ts[tok]
    tcal = apply_platt(t_or, cal)
    test_m = metrics_at(tcal, ty[tok], thr)
    val_m = metrics_at(vcal, vy[ok], thr)
    enough = min(np.sum(ty == 0), np.sum(ty == 1), np.sum(vy == 0), np.sum(vy == 1)) >= min_per_class
    reasons = []
    if not enough:
        reasons.append(f"fewer than {min_per_class} clips per class in a split")
    if test_m["auc"] is None or test_m["auc"] < min_auc:
        reasons.append(f"test AUC {test_m['auc']} < required {min_auc}")
    validated = not reasons
    sel = {"model": sp.key, "model_file": str(Path(best["_model_path"]).relative_to(ROOT)) if str(best["_model_path"]).startswith(str(ROOT)) else str(best["_model_path"]),
           "norm": sp.norm, "n_fft": sp.n_fft, "hop": sp.hop, "frames": sp.frames, "n_mels": sp.n_mels, "high_means": sp.high_means,
           "calibration": cal, "threshold": round(thr, 4), "validated": validated,
           "validation": {"validated": validated, "reasons_not_validated": reasons, "val_metrics": val_m, "test_metrics": test_m,
                          "protocol": "model/preprocessing/polarity/calibration/threshold chosen on the validation split only; test split evaluated once",
                          "split_notes": notes, "n_val": int(len(vy)), "n_test": int(len(ty))}}
    report = {"selected": sel, "candidates": [{k: v for k, v in r.items() if not k.startswith("_")} for r in table], "norms": NORMS}
    if out_file:
        Path(out_file).write_text(json.dumps(sel, indent=2))
        Path(str(out_file).replace(".json", "_report.json")).write_text(json.dumps(report, indent=2))
    return report
