"""False-genuine video fixes. FA3 weights are unavailable, so FA3 outputs are MOCKED (scripted scores) — these tests verify the
pipeline/decision logic (face gating, aggregation, withholding 'authentic'), NOT detection accuracy. Face detection is REAL
(OpenCV Haar on a real public-domain photo)."""
import numpy as np
import pytest
from PIL import Image

from app import registry
from app.config import Policy
from app.models.fa3 import FA3Detector
from app.services import faces as faces_svc, fusion, video as v
from app.services.aggregate import aggregate, effective_k
from tests.conftest import MockFA3

pytestmark = pytest.mark.filterwarnings("ignore")


def vres(path, scores, pol=None):
    registry.set_detectors(fa3=MockFA3(scores))
    pol = pol or Policy(max_faces_per_frame=1)       # one crop per frame so scripted scores map 1:1 onto frames
    r = v.analyze_video(path, pol, 6.0)
    return r, fusion.modality_verdict(r, pol, "video")


def test_real_face_detected_noise_not():
    d = faces_svc.FaceDetector()
    d.load()
    assert d.available, d.reason
    face = np.array(Image.open("tests/assets/face_astronaut.jpg").convert("RGB"))
    assert len(d.detect(face)) == 1
    assert d.detect((np.random.default_rng(0).random((300, 400, 3)) * 255).astype("uint8")) == []


def test_no_face_video_is_insufficient_evidence_not_genuine(media):
    r, vd = vres(media["video_av"], 0.05)                  # testsrc pattern: no faces; model would say "real" if it were asked
    assert r["status"] == "ok" and r["scored"] == 0 and r["score"] is None and r["crop_modes"]["full_frame"] == 0
    assert vd["verdict"] == "inconclusive" and vd["insufficient_evidence"] and "NOT evidence" in vd["reasons"][0]
    assert registry.get_fa3().calls == 0                   # whole frames were never sent to the face-trained model


def test_full_frame_fallback_is_explicit_opt_in(media):
    r, _ = vres(media["video_av"], 0.05, Policy(video_full_frame_fallback=True, max_faces_per_frame=1))
    assert r["scored"] > 0 and r["crop_modes"]["full_frame"] > 0


def test_face_video_scores_face_crops_across_timeline(media):
    r, vd = vres(media["video_face"], 0.1)
    assert r["scored"] == 16 and r["crop_modes"]["face"] == 16 and r["evidence"]["temporal_coverage"] > 0.9
    assert vd["verdict"] == "likely_authentic"             # supported: many face frames, low scores, full coverage
    ts = [f["t"] for f in r["frames"] if f["score"] is not None]
    assert ts == sorted(ts) and ts[-1] - ts[0] > 4        # spread over the clip, not the first frames


def test_manipulated_minority_not_averaged_away(media):
    """6 of 16 frames strongly fake (37%): mean says authentic, the justified (top-k) aggregation does not."""
    sc = lambda i: 0.95 if i % 16 in (2, 5, 7, 9, 12, 14) else 0.05
    r, vd = vres(media["video_face"], sc)
    assert r["aggregates"]["mean"] < 0.5 and r["aggregates"]["topk"] > 0.6
    assert vd["verdict"] == "suspected"
    _, mean_vd = vres(media["video_face"], sc, Policy(frame_aggregation="mean", max_faces_per_frame=1))
    assert mean_vd["verdict"] != "likely_authentic"        # even with the old mean rule the high frames block "authentic"


def test_single_extreme_frame_cannot_decide_but_blocks_authentic(media):
    sc = lambda i: 0.97 if i % 16 == 3 else 0.03
    r, vd = vres(media["video_face"], sc)
    assert r["topk_used"] >= 4 and r["score"] < 0.4        # one frame cannot make it "suspected"...
    assert vd["verdict"] == "inconclusive" and "one sampled frame scored" in " ".join(vd["reasons"])   # ...nor "authentic"


def test_face_detector_missing_is_unavailable_not_silent_whole_frame(media):
    bad = faces_svc.FaceDetector(cascade_dir="/nonexistent")
    faces_svc.set_detector(bad)
    try:
        r, vd = vres(media["video_face"], 0.1)
        assert r["status"] == "unavailable" and "Face detector unavailable" in r["reason"] and vd["verdict"] == "unavailable"
        assert registry.get_fa3().calls == 0
    finally:
        faces_svc.set_detector(None)


def test_low_temporal_coverage_withholds_authentic():
    pol = Policy()
    res = {"status": "ok", "score": 0.1, "scored": 10, "aggregation_method": "topk", "dispersion": {"std": 0.0}, "max_frame_score": 0.1,
           "intervals": [], "evidence": {"temporal_coverage": 0.15, "face_detection_rate": 0.9}}
    out = fusion.modality_verdict(res, pol, "video")
    assert out["verdict"] == "inconclusive" and "span only 15%" in out["reasons"][-1]
    res["evidence"]["temporal_coverage"] = 0.9
    assert fusion.modality_verdict(res, pol, "video")["verdict"] == "likely_authentic"


def test_constant_fa3_health_blocks_verdict():
    res = {"status": "ok", "score": 0.02, "scored": 10, "aggregation_method": "topk", "dispersion": {"std": 0.0}, "max_frame_score": 0.02,
           "intervals": [], "model_health": {"status": "suspect", "message": "constant"}}
    out = fusion.modality_verdict(res, Policy(), "video")
    assert out["verdict"] == "inconclusive" and out["model_degenerate"]


def test_effective_k_and_topk():
    assert effective_k(16, 3, 0.25) == 4 and effective_k(2, 3, 0.25) == 2 and effective_k(40, 3, 0.25) == 10
    assert aggregate([0.9, 0.1, 0.1, 0.1], "topk", 0.5, 2) == pytest.approx(0.5)


class _Cfg:
    def __init__(self, l2i): self.label2id = l2i


def test_fa3_probe_flags_constant_model_and_passes_varying():
    d = FA3Detector()
    d._score = lambda ims, batch=8: [{"fake_prob": 0.01} for _ in ims]
    assert d._probe_health()["status"] == "suspect"
    seq = iter(np.linspace(0.1, 0.9, 4))
    d._score = lambda ims, batch=8: [{"fake_prob": float(next(seq))} for _ in ims]
    assert d._probe_health()["status"] == "ok"


def test_fa3_label_mapping_comes_from_config_and_unexpected_refused(tmp_path, monkeypatch):
    import app.vendor.fa3_model as vm
    import torch

    class M(torch.nn.Module):
        def __init__(self, l2i):
            super().__init__(); self.config = _Cfg(l2i); self.fusion_logit = torch.nn.Parameter(torch.zeros(1))
        def forward(self, **k): raise RuntimeError("no")
    import json
    (tmp_path / "fa3_model.pt").write_bytes(b"x")
    for l2i, ok, fake_idx in (({"real": 0, "fake": 1}, True, 1), ({"fake": 0, "real": 1}, True, 0), ({"a": 0, "b": 1}, False, None)):
        (tmp_path / "fa3_config.json").write_text(json.dumps({"label2id": l2i}))     # the checkpoint's own record of its class order
        monkeypatch.setattr(vm, "load_model", lambda d, l2i=l2i: (M({"fake": 0, "real": 1}), object()))   # vendored default is ignored
        d = FA3Detector(str(tmp_path)); d._probe_health = lambda: {"status": "ok"}
        d.load()
        assert d.available is ok
        if ok:
            assert d.fake_idx == fake_idx and d.class_mapping["fake"] == fake_idx   # NOT assumed to be index 0
        else:
            assert "unexpected class mapping" in d.reason


def test_live_frame_without_face_is_not_scored(client):
    sid = client.post("/api/live/start", json={}).json()["session_id"] if hasattr(client, "post") else None
    import io
    buf = io.BytesIO(); Image.fromarray((np.random.default_rng(1).random((240, 320, 3)) * 255).astype("uint8")).save(buf, "JPEG")
    r = client.post(f"/api/live/{sid}/frame", files={"file": ("f.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["frame"]["status"] == "no_face" and r.json()["frame"]["score"] is None