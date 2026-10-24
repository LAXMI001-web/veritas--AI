"""Decision-engine unit tests. All inputs are MOCKED model outputs (not real model performance)."""
from app.config import Policy
from app.services import fusion


def vres(score, scored=16, intervals=None, std=0.05):
    return {"status": "ok", "score": score, "scored": scored, "aggregation_method": "mean", "intervals": intervals or [],
            "dispersion": {"std": std}}


def ares(score, dur=6.0, scored=5, intervals=None, std=0.05):
    return {"status": "ok", "score": score, "scored": scored, "duration_s": dur, "aggregation_method": "mean",
            "intervals": intervals or [], "dispersion": {"std": std}}


P = Policy()


def f(v, a, has=True, kind="video"):
    return fusion.fuse(v, a, P, kind, has)


def test_A_neither_flagged():
    r = f(vres(0.1), ares(0.1)); assert r["scenario"] == "A" and r["overall"] == "likely_authentic"


def test_B_real_video_fake_audio_mixed():
    r = f(vres(0.1), ares(0.9)); assert r["scenario"] == "B" and r["overall"] == "mixed_authenticity"
    assert any("Strong disagreement" in c for c in r["conflicts"])


def test_C_fake_video_real_audio_mixed():
    r = f(vres(0.9), ares(0.1)); assert r["scenario"] == "C" and r["overall"] == "mixed_authenticity"


def test_D_both_flagged():
    r = f(vres(0.9), ares(0.9)); assert r["scenario"] == "D" and r["overall"] == "suspected_manipulation"


def test_E_inconclusive_band():
    r = f(vres(0.52), ares(0.1)); assert r["scenario"] == "E" and r["overall"] == "inconclusive"


def test_E_insufficient_frames():
    r = f(vres(0.9, scored=2), ares(0.1)); assert r["scenario"] == "E"


def test_E_short_audio_inconclusive():
    r = f(vres(0.1), ares(0.9, dur=0.5)); assert r["scenario"] == "E"


def test_F_no_audio_not_penalized():
    r = f(vres(0.1), None, has=False); assert r["scenario"] == "F" and r["overall"] == "likely_authentic"
    assert r["modalities"]["audio"]["verdict"] == "not_present"


def test_G_one_modality_failed_keeps_other():
    r = f(vres(0.1), {"status": "failed", "reason": "x"}); assert r["scenario"] == "G" and r["overall"] == "analysis_incomplete"
    assert r["modalities"]["video"]["score"] == 0.1


def test_G_model_unavailable():
    r = f({"status": "unavailable", "reason": "no ckpt"}, ares(0.1)); assert r["scenario"] == "G"


def test_H_internal_conflict_flagged():
    r = f(vres(0.1, std=0.45), ares(0.9)); assert any("inconsistent" in c for c in r["conflicts"])


def test_I_localized_interval_secondary():
    iv = [{"start": 1, "end": 2, "n": 3, "max": 0.95, "mean": 0.9}]
    r = f(vres(0.2, intervals=iv), ares(0.1)); assert r["secondary_scenario"] == "I" and r["localized_intervals"]


def test_threshold_configurable():
    p = Policy(suspicious_threshold=0.3, inconclusive_margin=0.05)
    r = fusion.fuse(vres(0.5), ares(0.1), p, "video", True); assert r["modalities"]["video"]["verdict"] == "suspected"


def test_image_only_and_audio_only():
    assert fusion.fuse(vres(0.9), None, P, "image", False)["overall"] == "suspected_manipulation"
    assert fusion.fuse(None, ares(0.1), P, "audio", True)["overall"] == "likely_authentic"


def test_policy_validation():
    import pytest
    from app.config import POLICY
    with pytest.raises(AssertionError):
        POLICY.update({"suspicious_threshold": 1.5})
    with pytest.raises(ValueError):
        POLICY.update({"nonsense": 1})
