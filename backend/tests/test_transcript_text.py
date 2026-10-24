"""Transcript wording check + voice-vs-wording cross-check.
MOCKED: audio verdicts/scores and transcripts are supplied by the tests (pure-function tests of the policy logic).
REAL: see test_real_asr.py."""
import pytest

from app.config import Policy
from app.services import textcheck as tx

SCAM = ("Hello this is the bank security department. Your account has been compromised. You must share the one time "
        "password right now and transfer the money to a safe account immediately. Do not tell anyone.")
BENIGN = "Hello, thanks for calling. I wanted to confirm our meeting tomorrow at ten in the morning in the main office."
POL = Policy()


def tr(text, conf=0.9, start=0.0, end=10.0):
    return {"status": "ok", "engine": "x", "note": "n", "segments": [{"start": start, "end": end, "text": text, "confidence": conf}],
            "text": text, "word_count": len(text.split()), "mean_confidence": conf}


def audio_res(score, n=5):
    return {"status": "ok", "windows": [{"start": i * 2.0, "end": i * 2.0 + 2.048, "score": score, "silent": False} for i in range(n)]}


def test_scam_text_flagged_benign_ordinary():
    assert tx.classify_text(SCAM, POL)["class"] == "suspicious"
    assert tx.classify_text(BENIGN, POL)["class"] == "ordinary"


def test_single_category_is_only_mild():
    c = tx.classify_text("Please do not share your password with anybody at the office today.", POL)
    assert c["class"] == "mild" and c["categories"] == 1


def test_longer_phrase_not_double_counted():
    inds = tx.find_indicators("send me the one time password")
    assert [i["phrase"] for i in inds if i["category"] == "credential request"] == ["one time password"]


def test_analyze_too_few_words_inconclusive():
    r = tx.analyze(tr("hello there"), POL)
    assert r["verdict"] == "inconclusive" and "word" in r["reasons"][0]


def test_analyze_low_asr_confidence_inconclusive():
    r = tx.analyze(tr(SCAM, conf=0.1), POL)
    assert r["verdict"] == "inconclusive" and "confidence" in r["reasons"][0]


def test_analyze_unavailable_passthrough():
    r = tx.analyze({"status": "unavailable", "reason": "no model"}, POL)
    assert r["status"] == "unavailable" and r["reason"] == "no model"


def _cc(av, text, score):
    t = tr(text)
    return tx.crosscheck(av, t, tx.analyze(t, POL), audio_res(score), POL)


def test_audio_fake_but_text_ordinary():
    cc = _cc("suspected", BENIGN, 0.95)
    assert cc["kind"] == "audio_flagged_text_ordinary" and cc["severity"] == "warn"
    assert cc["segments"][0]["mismatch"] and cc["segments"][0]["audio_class"] == "suspected"


def test_audio_real_but_text_suspicious_vice_versa():
    cc = _cc("likely_authentic", SCAM, 0.02)
    assert cc["kind"] == "audio_ok_text_suspicious" and cc["segments"][0]["mismatch"]


def test_both_flagged_and_both_ordinary():
    assert _cc("suspected", SCAM, 0.95)["kind"] == "both_flagged"
    ok = _cc("likely_authentic", BENIGN, 0.02)
    assert ok["kind"] == "both_ordinary" and not ok["segments"][0]["mismatch"]


def test_inconclusive_audio_or_text_not_comparable():
    assert _cc("inconclusive", SCAM, 0.5)["kind"] == "not_comparable"
    t = tr("hi there")
    assert tx.crosscheck("suspected", t, tx.analyze(t, POL), audio_res(0.9), POL)["kind"] == "not_comparable"


def test_transcript_unavailable_not_comparable():
    u = {"status": "unavailable", "reason": "no model"}
    cc = tx.crosscheck("suspected", u, tx.analyze(u, POL), audio_res(0.9), POL)
    assert cc["kind"] == "not_comparable" and "no model" in cc["explanation"][0]


def test_per_segment_mismatch_localised():
    t = {"status": "ok", "engine": "x", "note": "n", "word_count": 40, "mean_confidence": 0.9, "text": BENIGN + " " + SCAM,
         "segments": [{"start": 0.0, "end": 4.0, "text": BENIGN, "confidence": .9}, {"start": 6.0, "end": 10.0, "text": SCAM, "confidence": .9}]}
    a = {"status": "ok", "windows": [{"start": 0, "end": 2.048, "score": 0.95, "silent": False}, {"start": 2.0, "end": 4.048, "score": 0.95, "silent": False},
                                      {"start": 6, "end": 8.048, "score": 0.05, "silent": False}, {"start": 8, "end": 10.048, "score": 0.05, "silent": False}]}
    cc = tx.crosscheck("inconclusive", t, tx.analyze(t, POL), a, POL)
    s0, s1 = cc["segments"]
    assert s0["audio_class"] == "suspected" and s0["text_class"] == "ordinary" and s0["mismatch"]     # fake-sounding voice, ordinary words
    assert s1["audio_class"] == "authentic" and s1["text_class"] == "suspicious" and s1["mismatch"]   # natural voice, scam words


def test_segment_without_audio_windows_has_unknown_audio():
    t = tr(BENIGN, start=50.0, end=55.0)
    cc = tx.crosscheck("suspected", t, tx.analyze(t, POL), audio_res(0.9), POL)
    assert cc["segments"][0]["audio_class"] == "unknown" and not cc["segments"][0]["mismatch"]
