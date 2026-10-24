"""Transparent decision policy: per-modality verdicts + the multimodal authenticity matrix (scenarios A–I).
Scores are uncalibrated model outputs oriented so that HIGH = more suspicious. All thresholds live in Policy."""
from __future__ import annotations

from ..config import SETTINGS, Policy

LABELS = {
    "likely_authentic": "Likely authentic",
    "suspected_manipulation": "Suspected manipulation",
    "mixed_authenticity": "Mixed-authenticity warning",
    "inconclusive": "Inconclusive",
    "analysis_incomplete": "Analysis incomplete",
}
MOD_LABELS = {
    "likely_authentic": "Likely authentic",
    "suspected": "Suspected manipulation",
    "inconclusive": "Inconclusive",
    "unavailable": "Unavailable",
    "failed": "Failed",
    "not_present": "No audio track",
}


def modality_verdict(res: dict | None, pol: Policy, modality: str) -> dict:
    """modality: 'video' | 'image' | 'audio'. res: output of the video/audio/image analyzers, or None."""
    if res is None:
        return {"modality": modality, "verdict": "not_present", "score": None, "reasons": ["No such stream in this media."]}
    st = res.get("status")
    if st == "unavailable":
        return {"modality": modality, "verdict": "unavailable", "score": None, "reasons": [res.get("reason") or "Model unavailable."]}
    if st == "failed":
        return {"modality": modality, "verdict": "failed", "score": None, "reasons": [res.get("reason") or "Analysis failed."]}
    score, reasons, warns = res.get("score"), [], []
    if score is None:
        return {"modality": modality, "verdict": "inconclusive", "score": None, "insufficient_evidence": True, "intervals": [],
                "reasons": [res.get("reason") or "No scorable evidence."]}
    insufficient = False
    if modality == "video":
        if res.get("scored", 0) < pol.min_frames:
            insufficient = True
            reasons.append(f"Only {res.get('scored', 0)} frames scored (minimum {pol.min_frames}).")
    if modality == "audio":
        if (res.get("duration_s") or 0) < pol.min_audio_seconds:
            insufficient = True
            reasons.append(f"Audio is {res.get('duration_s', 0):.1f}s long (minimum {pol.min_audio_seconds}s).")
        if res.get("scored", 0) < pol.min_scored_windows:
            insufficient = True
            reasons.append("No non-silent audio window could be scored.")
    thr, m = pol.suspicious_threshold, pol.inconclusive_margin
    if modality == "audio" and (res.get("model_validation") or {}).get("threshold") is not None:
        thr = res["model_validation"]["threshold"]      # threshold measured on validation data for this model
    if insufficient:
        verdict = "inconclusive"
    elif score >= thr + m:
        verdict = "suspected"
        reasons.append(f"{res.get('aggregation_method')} score {score:.2f} ≥ {thr + m:.2f}.")
    elif score <= thr - m:
        verdict = "likely_authentic"
        reasons.append(f"{res.get('aggregation_method')} score {score:.2f} ≤ {thr - m:.2f}.")
    else:
        verdict = "inconclusive"
        reasons.append(f"Score {score:.2f} lies within the inconclusive band ({thr - m:.2f}–{thr + m:.2f}).")
    if modality in ("video", "image") and verdict == "likely_authentic":
        # "genuine" must be positively supported: not merely an average that hides suspicious frames or thin coverage
        mx, ev = res.get("max_frame_score"), res.get("evidence") or {}
        if mx is not None and mx >= pol.authentic_max_frame:
            verdict = "inconclusive"
            reasons.append(f"Withheld 'likely authentic': one sampled frame scored {mx:.2f} (≥ {pol.authentic_max_frame:.2f}) — "
                           "suspicious evidence exists even though the aggregate is low.")
        cov = ev.get("temporal_coverage")
        if modality == "video" and cov is not None and cov < pol.min_temporal_coverage:
            verdict = "inconclusive"
            reasons.append(f"Withheld 'likely authentic': scored face frames span only {cov:.0%} of the video "
                           f"(minimum {pol.min_temporal_coverage:.0%}); the rest was not examined.")
        if modality == "video" and ev.get("face_detection_rate") is not None and ev["face_detection_rate"] < 0.3 and verdict == "likely_authentic":
            verdict = "inconclusive"
            reasons.append(f"Withheld 'likely authentic': a face was found in only {ev['face_detection_rate']:.0%} of candidate frames.")
    fh = res.get("model_health") if modality in ("video", "image") else None
    if fh and fh.get("status") == "suspect":
        reasons.insert(0, "FA3-CLIP health check failed: " + fh["message"])
        return {"modality": modality, "verdict": "inconclusive", "score": score, "reasons": reasons, "model_degenerate": True,
                "warnings": ["The raw score is shown for transparency but carries no evidence."], "insufficient_evidence": True, "intervals": []}
    health = res.get("model_health") if modality == "audio" else None
    if health and health.get("status") == "suspect" and not SETTINGS.audio_allow_degenerate:
        # a constant-output model must never produce "likely authentic" (or "suspected"): force inconclusive
        reasons.insert(0, "Audio model health check failed: " + health["message"])
        return {"modality": modality, "verdict": "inconclusive", "score": score, "reasons": reasons,
                "warnings": ["The raw score is shown for transparency but carries no evidence (model gives the same answer to any input)."],
                "insufficient_evidence": True, "model_degenerate": True, "intervals": []}
    mv = res.get("model_validation") if modality == "audio" else None
    if mv is not None and not mv.get("validated", True) and not SETTINGS.audio_allow_unvalidated:
        reasons = ["Audio model is not validated on labelled data (preprocessing, class polarity and threshold unconfirmed): "
                   "the score is shown but not interpreted. Run scripts/select_audio_model.py."]
        return {"modality": modality, "verdict": "inconclusive", "score": score, "reasons": reasons,
                "warnings": ["Raw score shown for transparency only."], "insufficient_evidence": True, "model_unvalidated": True, "intervals": []}
    if mv is not None and mv.get("external") and mv.get("note"):
        warns.append(mv["note"])
    disp = res.get("dispersion") or {}
    if disp.get("std") is not None and disp["std"] > 0.3:
        warns.append(f"Scores vary strongly across samples (std {disp['std']:.2f}); evidence is internally inconsistent.")
    ivs = res.get("intervals") or []
    if ivs and verdict == "likely_authentic":
        warns.append("Contiguous high-score samples exist even though the aggregate is low (possible localized finding).")
    return {"modality": modality, "verdict": verdict, "score": score, "reasons": reasons, "warnings": warns,
            "insufficient_evidence": insufficient, "intervals": ivs}


def _scenario(v: dict, a: dict, has_audio_track: bool, pol: Policy) -> tuple[str, str, str, list[str]]:
    """-> (scenario_code, overall_key, headline, conflicts)"""
    vv, av = v["verdict"], a["verdict"]
    conflicts: list[str] = []
    if vv in ("failed", "unavailable") or av == "failed" or (av == "unavailable" and has_audio_track):
        return "G", "analysis_incomplete", "One or more analyses could not be completed.", conflicts
    if av == "not_present":
        return "F", {"likely_authentic": "likely_authentic", "suspected": "suspected_manipulation"}.get(vv, "inconclusive"), \
            "No audio track: only the video was assessed.", conflicts
    if vv == "inconclusive" or av == "inconclusive":
        return "E", "inconclusive", "At least one modality is inconclusive.", conflicts
    vs, as_ = v["score"], a["score"]
    if vv == "likely_authentic" and av == "likely_authentic":
        code, key, head = "A", "likely_authentic", "Neither modality shows strong suspicious patterns."
    elif vv == "likely_authentic" and av == "suspected":
        code, key, head = "B", "mixed_authenticity", "Video looks likely authentic but audio is suspected."
    elif vv == "suspected" and av == "likely_authentic":
        code, key, head = "C", "mixed_authenticity", "Video is suspected but audio looks likely authentic."
    else:
        code, key, head = "D", "suspected_manipulation", "Both video and audio show suspicious patterns (assessed independently)."
    if code in ("B", "C") and abs(vs - as_) >= pol.conflict_gap:
        conflicts.append(f"Strong disagreement between modalities (video {vs:.2f} vs audio {as_:.2f}).")
    for m, r in (("video", v), ("audio", a)):
        for w in r.get("warnings", []):
            if "inconsistent" in w:
                conflicts.append(f"{m.capitalize()}: {w}")
    if conflicts and code in ("B", "C"):
        head += " Evidence conflicts (scenario H)."
    return code, key, head, conflicts


def fuse(video: dict | None, audio: dict | None, pol: Policy, kind: str, has_audio_track: bool) -> dict:
    """kind: 'image' | 'audio' | 'video'. Returns the full assessment object stored in the analysis record."""
    if kind == "image":
        v = modality_verdict(video, pol, "image")
        key = {"likely_authentic": "likely_authentic", "suspected": "suspected_manipulation",
               "inconclusive": "inconclusive"}.get(v["verdict"], "analysis_incomplete")
        return {"kind": kind, "scenario": "IMG", "overall": key, "overall_label": LABELS[key],
                "headline": "Single image assessed by FA3-CLIP.", "modalities": {"image": v}, "conflicts": [],
                "localized_intervals": False}
    if kind == "audio":
        a = modality_verdict(audio, pol, "audio")
        key = {"likely_authentic": "likely_authentic", "suspected": "suspected_manipulation",
               "inconclusive": "inconclusive"}.get(a["verdict"], "analysis_incomplete")
        return {"kind": kind, "scenario": "AUD", "overall": key, "overall_label": LABELS[key],
                "headline": "Audio assessed by CNN+Conformer.", "modalities": {"audio": a}, "conflicts": [],
                "localized_intervals": bool(a.get("intervals"))}
    v = modality_verdict(video, pol, "video")
    a = modality_verdict(audio if has_audio_track else None, pol, "audio")
    code, key, head, conflicts = _scenario(v, a, has_audio_track, pol)
    localized = bool(v.get("intervals") or a.get("intervals"))
    out = {"kind": kind, "scenario": code, "overall": key, "overall_label": LABELS[key], "headline": head,
           "modalities": {"video": v, "audio": a}, "conflicts": conflicts, "localized_intervals": localized}
    if localized and code in ("A", "B", "C", "D", "E", "F"):
        out["secondary_scenario"] = "I"
    return out
