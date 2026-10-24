"""Media comparison lab + benign-transformation robustness tests. Originals are never modified; transformed copies
are created in a temp dir, labelled as such, and removed afterwards."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..config import Policy
from . import pipeline
from .media import MediaError, cleanup, run, temp_dir, detect_kind, ffprobe


def _flat(d: dict, pre="") -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        elif not isinstance(v, (list,)):
            out[f"{pre}{k}"] = v
    return out


def metadata_diff(a: dict, b: dict) -> list[dict]:
    fa, fb = _flat(a), _flat(b)
    rows = []
    for k in sorted(set(fa) | set(fb)):
        if fa.get(k) != fb.get(k):
            rows.append({"field": k, "a": fa.get(k), "b": fb.get(k)})
    return rows


def image_difference(pa: Path, pb: Path) -> dict:
    A = Image.open(pa).convert("RGB"); B = Image.open(pb).convert("RGB")
    B2 = B.resize(A.size)
    a = np.asarray(A).astype(np.float32); b = np.asarray(B2).astype(np.float32)
    d = np.abs(a - b).mean(axis=2)
    norm = np.clip(d / max(d.max(), 1e-6), 0, 1)
    col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    ok, buf = cv2.imencode(".png", col)
    return {"mean_abs_diff": round(float(d.mean()), 3), "max_abs_diff": round(float(d.max()), 1),
            "resized_b_to_a": A.size != B.size, "diff_png": base64.b64encode(buf.tobytes()).decode()}


def compare(pa: Path, na: str, pb: Path, nb: str, pol: Policy, progress=None) -> dict:
    ka, kb = detect_kind(pa), detect_kind(pb)
    if ka != kb:
        raise MediaError(f"Both files must be the same kind (got {ka} and {kb}).", "kind_mismatch")
    ra = pipeline.analyze_file(pa, na, pol)
    if progress: progress(0.5, "analysing second file")
    rb = pipeline.analyze_file(pb, nb, pol)
    out = {"kind": ka, "a": pipeline.slim(ra), "b": pipeline.slim(rb), "a_id": ra["id"], "b_id": rb["id"],
           "metadata_diff": metadata_diff((ra["integrity"] or {}).get("metadata", {}), (rb["integrity"] or {}).get("metadata", {})),
           "assessment_a": ra["assessment"]["overall_label"], "assessment_b": rb["assessment"]["overall_label"]}
    sa = (ra["results"].get("video") or ra["results"].get("image") or {}).get("score")
    sb = (rb["results"].get("video") or rb["results"].get("image") or {}).get("score")
    au_a, au_b = (ra["results"].get("audio") or {}).get("score"), (rb["results"].get("audio") or {}).get("score")
    out["score_delta"] = {"visual": None if sa is None or sb is None else round(sb - sa, 4),
                          "audio": None if au_a is None or au_b is None else round(au_b - au_a, 4)}
    if ka == "image":
        try:
            out["image_difference"] = image_difference(pa, pb)
        except Exception:
            out["image_difference"] = None
        out["thumb_a"] = (ra["results"]["image"].get("frames") or [{}])[0].get("thumb") if ra["results"].get("image", {}).get("status") == "ok" else None
        out["thumb_b"] = (rb["results"]["image"].get("frames") or [{}])[0].get("thumb") if rb["results"].get("image", {}).get("status") == "ok" else None
    if ka == "audio":
        out["waveform_a"] = ra["results"]["audio"].get("waveform"); out["waveform_b"] = rb["results"]["audio"].get("waveform")
        out["spectrogram_a"] = ra["results"]["audio"].get("spectrogram_png"); out["spectrogram_b"] = rb["results"]["audio"].get("spectrogram_png")
        out["windows_a"] = ra["results"]["audio"].get("windows"); out["windows_b"] = rb["results"]["audio"].get("windows")
    if ka == "video":
        out["frames_a"] = [(f["t"], f["score"]) for f in (ra["results"].get("video", {}).get("frames") or [])]
        out["frames_b"] = [(f["t"], f["score"]) for f in (rb["results"].get("video", {}).get("frames") or [])]
        out["audio_windows_a"] = [(w["start"], w["score"]) for w in (ra["results"].get("audio", {}).get("windows") or [])]
        out["audio_windows_b"] = [(w["start"], w["score"]) for w in (rb["results"].get("audio", {}).get("windows") or [])]
        out["intervals_a"] = (ra["results"].get("video", {}) or {}).get("intervals", [])
        out["intervals_b"] = (rb["results"].get("video", {}) or {}).get("intervals", [])
    out["note"] = ("Differences in model scores between two files do not by themselves show which one is manipulated; "
                   "models can be sensitive to resolution, compression and preprocessing.")
    return out


# --------------------------- robustness ---------------------------
def _transforms(kind: str) -> list[tuple[str, str]]:
    if kind == "image":
        return [("resize_50pct", "Resize to 50% and back"), ("jpeg_q75", "JPEG recompression q=75"), ("jpeg_q40", "JPEG recompression q=40")]
    if kind == "audio":
        return [("resample_8k", "Resample to 8 kHz and back"), ("volume_0.5", "Volume ×0.5"), ("volume_1.5", "Volume ×1.5 (clipped)"),
                ("mp3_64k", "MP3 64 kbps compression")]
    return [("recode_crf32", "Video re-encode (H.264 CRF 32)"), ("scale_50pct", "Downscale video to 50%")]


def _apply(kind: str, name: str, src: Path, dst_dir: Path) -> Path:
    if kind == "image":
        im = Image.open(src).convert("RGB")
        if name == "resize_50pct":
            out = dst_dir / "t.png"; im.resize((max(im.width // 2, 1), max(im.height // 2, 1))).resize(im.size).save(out)
        else:
            out = dst_dir / "t.jpg"; im.save(out, "JPEG", quality=75 if name == "jpeg_q75" else 40)
        return out
    out = dst_dir / ("t.wav" if kind == "audio" and name != "mp3_64k" else "t.mp3" if kind == "audio" else "t.mp4")
    if kind == "audio":
        args = {"resample_8k": ["-af", "aresample=8000,aresample=16000"], "volume_0.5": ["-af", "volume=0.5"],
                "volume_1.5": ["-af", "volume=1.5,alimiter=limit=0.99"], "mp3_64k": ["-b:a", "64k", "-c:a", "libmp3lame"]}[name]
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-vn", *args, str(out)]
    else:
        args = ["-c:v", "libx264", "-crf", "32", "-preset", "veryfast", "-c:a", "aac"] if name == "recode_crf32" else \
            ["-vf", "scale=trunc(iw/4)*2:trunc(ih/4)*2", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-c:a", "aac"]
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), *args, str(out)]
    p = run(cmd)
    if p.returncode != 0 or not out.exists():
        raise MediaError("Transformation tool failed for this file/codec.", "transform_failed")
    return out


def robustness(path: Path, name: str, pol: Policy, progress=None) -> dict:
    import dataclasses
    pol = dataclasses.replace(pol, transcribe=False)   # robustness compares classifier scores only (keeps it fast)
    kind = detect_kind(path)
    base = pipeline.analyze_file(path, name, pol, do_integrity=False)
    key = "image" if kind == "image" else "video" if kind == "video" else "audio"
    bs = (base["results"].get(key) or {}).get("score")
    rows = []
    td = temp_dir()
    try:
        ts = _transforms(kind)
        for i, (tid, label) in enumerate(ts):
            if progress: progress((i + 1) / (len(ts) + 1), label)
            try:
                tp = _apply(kind, tid, path, td)
                r = pipeline.analyze_file(tp, f"[transformed] {name}", pol, do_integrity=False)
                s = (r["results"].get(key) or {}).get("score")
                rows.append({"id": tid, "label": label, "score": s, "verdict": r["assessment"]["modalities"].get(key, {}).get("verdict"),
                             "delta": None if s is None or bs is None else round(s - bs, 4)})
            except MediaError as exc:
                rows.append({"id": tid, "label": label, "error": str(exc)})
    finally:
        cleanup(td)
    deltas = [abs(r["delta"]) for r in rows if r.get("delta") is not None]
    flips = [r["label"] for r in rows if r.get("verdict") and r["verdict"] != base["assessment"]["modalities"].get(key, {}).get("verdict")]
    return {"kind": kind, "original": {"score": bs, "verdict": base["assessment"]["modalities"].get(key, {}).get("verdict")},
            "transforms": rows, "max_abs_delta": round(max(deltas), 4) if deltas else None, "verdict_changes": flips,
            "stability": None if not deltas else ("stable" if max(deltas) < 0.1 and not flips else "sensitive"),
            "note": "Score changes after benign transformations indicate sensitivity to media quality/preprocessing; "
                    "they do NOT indicate that the file is fake. Originals were not modified; transformed copies were deleted."}
