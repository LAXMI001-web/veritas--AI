"""Transcript content check + transcript-vs-audio cross-check.

IMPORTANT: this is a transparent phrase-matching heuristic over ASR text, NOT a trained or validated model.
  * "Suspicious wording" means the transcript contains phrases typical of scams/pressure tactics (urgency,
    credential or payment requests, impersonation of authorities, secrecy, threats ...). Genuine calls can contain
    them too, and scammers can avoid them.
  * It says nothing about whether the VOICE is synthetic. The cross-check only reports whether the audio model's
    verdict and the wording look consistent or contradictory, so a person can review the recording.
English only. Lexicon and thresholds are listed below and in the policy so they can be audited and changed."""
from __future__ import annotations

import math
import re

from ..config import Policy

# category -> (weight, phrases). Matching is case-insensitive on word boundaries.
LEXICON: dict[str, tuple[float, list[str]]] = {
    "urgency / pressure": (0.5, ["immediately", "right now", "urgent", "urgently", "act now", "act fast",
                                 "within 24 hours", "last chance", "as soon as possible", "today only",
                                 "before it's too late", "without delay", "right away", "don't delay"]),
    "credential request": (1.0, ["one time password", "one-time password", "otp", "pin number", "your pin", "password",
                                 "cvv", "card number", "verification code", "security code", "login details",
                                 "account number", "social security", "aadhaar", "pan card", "bank details"]),
    "payment / transfer request": (1.0, ["transfer the money", "wire transfer", "wire the money", "send money", "gift card",
                                         "gift cards", "bitcoin", "cryptocurrency", "western union", "processing fee",
                                         "pay a fee", "safe account", "secure account", "payment link", "send the money"]),
    "authority impersonation": (0.6, ["security department", "fraud department", "bank security", "customs", "police",
                                      "tax department", "income tax", "internal revenue", "tech support",
                                      "technical support", "microsoft support", "social security administration",
                                      "your account has been", "compromised", "suspended"]),
    "threat": (0.8, ["arrest", "arrested", "warrant", "legal action", "lawsuit", "jail", "block your account",
                     "freeze your account", "deactivated", "penalty", "prosecuted"]),
    "secrecy / isolation": (0.9, ["do not tell anyone", "don't tell anyone", "do not tell", "keep this confidential",
                                  "keep it secret", "between us", "do not hang up", "don't hang up", "do not disconnect"]),
    "prize / reward": (0.8, ["you have won", "you've won", "lottery", "claim your prize", "claim your reward",
                             "free gift", "you are selected", "you have been selected"]),
    "remote access": (0.9, ["anydesk", "teamviewer", "remote access", "install this app", "download the app",
                            "share your screen"]),
}
_RX = {c: [(p, re.compile(r"(?<!\w)" + re.escape(p) + r"(?!\w)", re.I)) for p in ph] for c, (_, ph) in LEXICON.items()}


def find_indicators(text: str) -> list[dict]:
    out = []
    for cat, pats in _RX.items():
        for phrase, rx in pats:
            for m in rx.finditer(text):
                out.append({"category": cat, "phrase": phrase, "pos": m.start()})
    # a phrase contained in a longer matched phrase of the same category counts once ("one time password" vs "password")
    keep = []
    for i in out:
        if not any(j is not i and j["category"] == i["category"] and j["pos"] <= i["pos"] and
                   i["pos"] + len(i["phrase"]) <= j["pos"] + len(j["phrase"]) and len(j["phrase"]) > len(i["phrase"]) for j in out):
            keep.append(i)
    return sorted(keep, key=lambda x: x["pos"])


def risk_score(inds: list[dict]) -> tuple[float, int]:
    """Saturating score in [0,1) from distinct categories (weight each, +25 % per repeated phrase up to x1.5)."""
    by: dict[str, int] = {}
    for i in inds:
        by[i["category"]] = by.get(i["category"], 0) + 1
    raw = sum(LEXICON[c][0] * min(1.5, 1 + 0.25 * (n - 1)) for c, n in by.items())
    return 1 - math.exp(-raw / 1.5), len(by)


def classify_text(text: str, pol: Policy) -> dict:
    inds = find_indicators(text)
    score, ncat = risk_score(inds)
    if ncat >= 2 and score >= pol.text_suspicious_cut:
        cls = "suspicious"
    elif inds:
        cls = "mild"
    else:
        cls = "ordinary"
    return {"class": cls, "score": round(score, 3), "categories": ncat, "indicators": inds}


def analyze(tr: dict | None, pol: Policy) -> dict:
    """Whole-transcript content check."""
    if tr is None:
        return {"status": "not_applicable", "reason": "No audio to transcribe."}
    if tr.get("status") != "ok":
        return {"status": tr.get("status", "failed"), "reason": tr.get("reason")}
    words, conf = tr.get("word_count", 0), tr.get("mean_confidence")
    base = {"status": "ok", "method": "phrase-matching heuristic (not a trained model)", "word_count": words,
            "asr_confidence": conf}
    if words < pol.text_min_words:
        return {**base, "verdict": "inconclusive", "score": None, "indicators": [],
                "reasons": [f"Only {words} word(s) transcribed (minimum {pol.text_min_words})."]}
    if conf is not None and conf < pol.asr_min_confidence:
        return {**base, "verdict": "inconclusive", "score": None, "indicators": [],
                "reasons": [f"Transcript confidence is low (mean {conf:.2f} < {pol.asr_min_confidence}); wording cannot be judged reliably."]}
    c = classify_text(tr["text"], pol)
    verdict = {"suspicious": "suspicious_wording", "mild": "mild_indicators", "ordinary": "ordinary_wording"}[c["class"]]
    reasons = {"suspicious_wording": [f"{c['categories']} indicator categories matched (risk score {c['score']:.2f} ≥ {pol.text_suspicious_cut})."],
               "mild_indicators": [f"{len(c['indicators'])} indicator phrase(s) in {c['categories']} categor{'y' if c['categories'] == 1 else 'ies'}; below the suspicious-wording rule (needs ≥2 categories and score ≥ {pol.text_suspicious_cut})."],
               "ordinary_wording": ["No scam/pressure phrases from the lexicon were found."]}[verdict]
    return {**base, "verdict": verdict, "score": c["score"], "indicators": c["indicators"], "reasons": reasons}


def _audio_class(score: float | None, pol: Policy) -> str:
    if score is None:
        return "unknown"
    t, m = pol.suspicious_threshold, pol.inconclusive_margin
    return "suspected" if score >= t + m else "authentic" if score <= t - m else "inconclusive"


def _seg_audio_score(a: dict | None, s: float, e: float) -> float | None:
    if not a or a.get("status") != "ok":
        return None
    vals = []
    for w in a.get("windows", []):
        if w.get("score") is None:
            continue
        ov = min(e, w["end"]) - max(s, w["start"])
        if ov > 0.25 * min(e - s, w["end"] - w["start"]):
            vals.append(w["score"])
    return round(sum(vals) / len(vals), 4) if vals else None


def crosscheck(audio_verdict: str, tr: dict | None, tc: dict, audio_res: dict | None, pol: Policy) -> dict:
    """Compare the audio model's verdict with what the transcript says. audio_verdict is from fusion.modality_verdict."""
    out = {"status": "ok", "audio_verdict": audio_verdict, "text_verdict": tc.get("verdict"), "segments": []}
    if tr is None or tr.get("status") != "ok":
        return {**out, "status": "unavailable", "kind": "not_comparable", "severity": "info",
                "headline": "Transcript unavailable — no text/voice comparison.", "explanation": [
                    (tr or {}).get("reason") or "No transcript was produced."]}
    for sg in tr["segments"]:
        sc = _seg_audio_score(audio_res, sg["start"], sg["end"])
        ac = _audio_class(sc, pol)
        words = len(sg["text"].split())
        tcls = classify_text(sg["text"], pol) if words >= 3 else {"class": "ordinary", "score": 0.0, "indicators": [], "categories": 0}
        mism = (ac == "suspected" and tcls["class"] == "ordinary") or (ac == "authentic" and tcls["class"] == "suspicious")
        out["segments"].append({**sg, "audio_score": sc, "audio_class": ac, "text_class": tcls["class"],
                                "indicators": tcls["indicators"], "mismatch": mism,
                                "low_confidence": sg.get("confidence") is not None and sg["confidence"] < pol.asr_min_confidence})
    tv = tc.get("verdict")
    if tv in (None, "inconclusive") or audio_verdict not in ("suspected", "likely_authentic"):
        why = []
        if tv in (None, "inconclusive"):
            why += tc.get("reasons") or ["Text check inconclusive."]
        if audio_verdict not in ("suspected", "likely_authentic"):
            why.append(f"Audio verdict is '{audio_verdict}', so there is nothing firm to compare the wording against.")
        return {**out, "kind": "not_comparable", "severity": "info",
                "headline": "Not enough firm evidence on both sides to compare voice and wording.", "explanation": why}
    text_susp, text_ok = tv == "suspicious_wording", tv in ("ordinary_wording", "mild_indicators")
    if audio_verdict == "suspected" and text_ok:
        return {**out, "kind": "audio_flagged_text_ordinary", "severity": "warn",
                "headline": "Voice looks suspicious, but the spoken words look ordinary.",
                "explanation": ["The audio model flagged suspicious voice patterns, while the transcript contains no scam/pressure wording.",
                                "Possible readings (not established): a synthetic or altered voice reading harmless text, "
                                "a false positive of the audio model on this recording, or poor recording conditions.",
                                "Normal-sounding words do not make a voice authentic, and the audio model score is uncalibrated."]}
    if audio_verdict == "likely_authentic" and text_susp:
        return {**out, "kind": "audio_ok_text_suspicious", "severity": "warn",
                "headline": "Voice looks natural to the model, but the spoken words look suspicious.",
                "explanation": ["The audio model found no strong suspicious voice patterns, while the transcript contains scam/pressure wording.",
                                "A real human can say suspicious things (and a fake voice can pass the audio model), so this is a content warning, "
                                "not a finding about whether the voice is synthetic.",
                                "Treat the content with caution and verify the sender through an independent channel."]}
    if audio_verdict == "suspected" and text_susp:
        return {**out, "kind": "both_flagged", "severity": "bad",
                "headline": "Both the voice and the wording look suspicious.",
                "explanation": ["Suspicious voice patterns and scam/pressure wording were found independently; neither proves the other.",
                                "Verify the speaker and the request through a trusted channel before acting."]}
    return {**out, "kind": "both_ordinary", "severity": "ok", "headline": "Voice and wording both look ordinary.",
            "explanation": ["No strong suspicious voice patterns and no scam/pressure wording were found. This does not prove the recording is genuine."]}


def summary(cc: dict) -> dict:
    """Compact copy (no segments) stored in the assessment."""
    return {k: cc.get(k) for k in ("kind", "severity", "headline", "explanation", "audio_verdict", "text_verdict")}
