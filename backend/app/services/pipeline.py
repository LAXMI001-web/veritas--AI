"""Analysis orchestrator: file -> record (dict). Each modality is isolated so one failure never discards the other."""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .. import registry
from ..config import POLICY, Policy
from dataclasses import asdict
from . import audio as audio_svc, fusion, integrity, textcheck, transcript as transcript_svc, video as video_svc
from .explain import explain
from .media import MediaError, detect_kind, duration_of, ffprobe, has_audio, sha256_file, validate_limits


class Cancelled(Exception):
    pass


def _models_info() -> dict:
    return {"fa3": registry.get_fa3().info(), "audio": registry.get_audio().info()}


def analyze_file(path: Path, original_name: str, policy: Policy | None = None, progress=None,
                 cancel=None, do_integrity: bool = True, force_kind: str | None = None) -> dict:
    pol = policy or POLICY.get()
    prog = progress or (lambda f, m: None)
    chk = lambda: (cancel and cancel.is_set()) and (_ for _ in ()).throw(Cancelled())  # noqa: E731
    t0 = time.time()
    prog(0.02, "validating file")
    probe = ffprobe(path)
    kind = detect_kind(path, probe)
    if force_kind and force_kind != kind:
        raise MediaError(f"This page expects {force_kind} media but the file is {kind}.", "wrong_kind")
    warnings = validate_limits(path, probe, kind)
    dur = duration_of(probe)
    audio_present = has_audio(probe)
    results: dict = {}
    timings: dict = {}
    chk()
    if kind == "image":
        prog(0.15, "scoring image (FA3-CLIP)")
        ts = time.time()
        try:
            results["image"] = video_svc.analyze_image(path, pol)
        except MediaError as exc:
            results["image"] = {"status": "failed", "reason": str(exc), "modality": "image"}
        timings["image_s"] = round(time.time() - ts, 2)
    elif kind == "audio":
        prog(0.15, "scoring audio (CNN+Conformer)")
        ts = time.time()
        results["audio"] = audio_svc.analyze_audio_file(path, pol)
        timings["audio_s"] = round(time.time() - ts, 2)
    else:
        prog(0.1, "sampling frames")
        ts = time.time()
        try:
            results["video"] = video_svc.analyze_video(path, pol, dur, prog)
        except MediaError as exc:
            results["video"] = {"status": "failed", "reason": str(exc), "modality": "video"}
        except Exception as exc:
            results["video"] = {"status": "failed", "reason": f"Video analysis error: {type(exc).__name__}", "modality": "video"}
        timings["video_s"] = round(time.time() - ts, 2)
        chk()
        if audio_present:
            prog(0.6, "extracting & scoring audio track")
            ts = time.time()
            try:
                results["audio"] = audio_svc.analyze_audio_file(path, pol, from_video=True)
            except Exception as exc:
                results["audio"] = {"status": "failed", "reason": f"Audio analysis error: {type(exc).__name__}", "modality": "audio"}
            timings["audio_s"] = round(time.time() - ts, 2)
    chk()
    if pol.transcribe and kind in ("audio", "video") and audio_present:
        prog(0.7, "transcribing speech")
        ts = time.time()
        tr = transcript_svc.transcribe_file(path)
        results["transcript"] = tr
        results["text_check"] = textcheck.analyze(tr, pol)
        timings["transcript_s"] = round(time.time() - ts, 2)
    chk()
    integ = None
    if do_integrity:
        prog(0.8, "integrity inspection")
        try:
            integ = integrity.inspect(path, kind, probe)
        except Exception as exc:
            integ = {"error": f"Integrity inspection failed: {type(exc).__name__}"}
    if kind == "video" and pol.sync_check and audio_present:
        prog(0.88, "audio-visual sync check")
        try:
            from . import sync
            results["sync"] = sync.check(path)
        except Exception as exc:
            results["sync"] = {"status": "failed", "reason": f"Sync check failed: {type(exc).__name__}"}
    assessment = fusion.fuse(results.get("video") or results.get("image"), results.get("audio"), pol, kind, audio_present)
    if "transcript" in results:
        av = fusion.modality_verdict(results.get("audio"), pol, "audio")["verdict"]
        cc = textcheck.crosscheck(av, results["transcript"], results["text_check"], results.get("audio"), pol)
        results["text_crosscheck"] = cc
        assessment["text_crosscheck"] = textcheck.summary(cc)
        if cc.get("kind") in ("audio_flagged_text_ordinary", "audio_ok_text_suspicious"):
            assessment.setdefault("conflicts", []).append("Voice vs wording: " + cc["headline"])
    pdict = asdict(pol)
    rec = {
        "id": uuid.uuid4().hex[:16],
        "filename": Path(original_name).name[:200],
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": integ["metadata"]["sha256"] if integ and "metadata" in integ else sha256_file(path),
        "duration_s": dur, "has_audio": audio_present,
        "policy": pdict, "models": _models_info(), "results": results, "assessment": assessment,
        "integrity": integ, "warnings": warnings, "timings": timings,
        "total_s": round(time.time() - t0, 2),
        "disclaimer": "Machine-generated assessment from uncalibrated classifiers. Not definitive forensic proof.",
    }
    rec["explanation"] = explain(rec)
    prog(1.0, "done")
    return rec


def slim(rec: dict) -> dict:
    """Summary row for lists."""
    a = rec["assessment"]
    return {"id": rec["id"], "filename": rec["filename"], "kind": rec["kind"], "created_at": rec["created_at"],
            "overall": a["overall"], "overall_label": a["overall_label"], "scenario": a["scenario"],
            "video_score": (rec["results"].get("video") or rec["results"].get("image") or {}).get("score"),
            "audio_score": (rec["results"].get("audio") or {}).get("score")}
