"""Transcript in file analysis + live detection API.
MOCKED: FA3, audio model scores and ASR output (see conftest). Real models: test_real_asr.py."""
import os
import subprocess
import time
from pathlib import Path

import pytest

from tests.conftest import MockAudio, MockFA3
from app import registry
from app.config import SETTINGS

SCAM = ("Hello this is the bank security department. Your account has been compromised. You must share the one time "
        "password right now and transfer the money to a safe account immediately. Do not tell anyone.")


def run(client, path, name, expect="", policy=None):
    with open(path, "rb") as f:
        data = {"expect": expect}
        r = client.post("/api/analyze", files={"file": (name, f, "application/octet-stream")}, data=data)
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    for _ in range(300):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.2)
    assert j["status"] == "done", j
    return client.get(f"/api/analyses/{j['analysis_id']}").json()


def test_video_audio_fake_text_ordinary_is_shown(client, media):
    registry.set_detectors(fa3=MockFA3(0.05), audio=MockAudio(0.95))
    rec = run(client, media["video_face_av"], "v.mp4", "video")
    assert rec["results"]["transcript"]["status"] == "ok"
    assert rec["results"]["text_check"]["verdict"] == "ordinary_wording"
    assert rec["assessment"]["text_crosscheck"]["kind"] == "audio_flagged_text_ordinary"
    assert any("Voice vs wording" in c for c in rec["assessment"]["conflicts"])
    assert rec["assessment"]["scenario"] == "B"                          # text check is advisory: matrix unchanged
    assert "segments" not in rec["assessment"]["text_crosscheck"]
    assert rec["results"]["text_crosscheck"]["segments"][0]["mismatch"]


def test_video_audio_real_text_suspicious_vice_versa(client, media, asr):
    asr(text=SCAM)
    registry.set_detectors(fa3=MockFA3(0.05), audio=MockAudio(0.02))
    rec = run(client, media["video_face_av"], "v.mp4", "video")
    assert rec["assessment"]["text_crosscheck"]["kind"] == "audio_ok_text_suspicious"
    assert rec["assessment"]["scenario"] == "A" and rec["assessment"]["overall"] == "likely_authentic"


def test_audio_file_has_transcript(client, media):
    rec = run(client, media["audio_wav"], "a.wav", "audio")
    assert rec["results"]["transcript"]["segments"][0]["text"].startswith("Hello")


def test_video_without_audio_has_no_transcript(client, media):
    rec = run(client, media["video_face"], "n.mp4", "video")
    assert "transcript" not in rec["results"] and "text_crosscheck" not in rec["assessment"]


def test_image_has_no_transcript(client, media):
    assert "transcript" not in run(client, media["image"], "i.png", "image")["results"]


def test_transcript_unavailable_does_not_break_analysis(client, media, asr):
    asr(status="unavailable")
    rec = run(client, media["video_face_av"], "v.mp4", "video")
    assert rec["results"]["transcript"]["status"] == "unavailable"
    assert rec["assessment"]["text_crosscheck"]["kind"] == "not_comparable"
    assert rec["assessment"]["modalities"]["audio"]["verdict"] != "failed"


def test_transcript_failure_does_not_break_analysis(client, media, asr):
    asr(status="failed")
    rec = run(client, media["audio_wav"], "a.wav", "audio")
    assert rec["results"]["transcript"]["status"] == "failed" and rec["results"]["audio"]["status"] == "ok"


def test_policy_can_disable_transcription(client, media):
    r = client.put("/api/policy", json={"transcribe": False}); assert r.status_code == 200
    try:
        rec = run(client, media["audio_wav"], "a.wav", "audio")
        assert "transcript" not in rec["results"]
    finally:
        client.put("/api/policy", json={"transcribe": True})


def test_reports_include_transcript(client, media, asr):
    asr(text=SCAM)
    rec = run(client, media["video_face_av"], "v.mp4", "video")
    pdf = client.get(f"/api/analyses/{rec['id']}/report.pdf"); assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    js = client.get(f"/api/analyses/{rec['id']}/report.json").json()
    assert js["analysis"]["results"]["transcript"]["text"].startswith("Hello this is the bank")


def test_models_endpoint_reports_asr(client):
    assert client.get("/api/models").json()["asr"]["available"] is True


# ----------------------------- live -----------------------------
@pytest.fixture
def chunk(tmp_path):
    p = tmp_path / "c.webm"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=4", "-c:a", "libopus", str(p)], check=True)
    return p


def post_chunk(client, sid, p):
    with open(p, "rb") as f:
        return client.post(f"/api/live/{sid}/audio", files={"file": ("c.webm", f, "audio/webm")})


def test_live_chunk_flow_and_rolling_mismatch(client, chunk, asr):
    registry.set_detectors(audio=MockAudio(0.95))
    s = client.post("/api/live/start").json(); sid = s["session_id"]
    r = post_chunk(client, sid, chunk); assert r.status_code == 200, r.text
    j = r.json()
    assert j["chunk"]["start"] == 0.0 and abs(j["chunk"]["end"] - 4.0) < 0.2 and j["chunk"]["score"] == 0.95
    seg = j["transcript"]["segments"][0]
    assert seg["audio_class"] == "suspected" and seg["text_class"] == "ordinary" and seg["mismatch"]
    j2 = post_chunk(client, sid, chunk).json()
    assert j2["chunk"]["start"] > 3.5                                       # session clock advances
    assert j2["assessment"]["modalities"]["audio"]["verdict"] == "suspected"
    assert j2["assessment"]["text_crosscheck"]["kind"] == "audio_flagged_text_ordinary"
    assert client.delete(f"/api/live/{sid}").json() == {"ended": True}
    assert post_chunk(client, sid, chunk).status_code == 404


def test_live_scam_words_with_authentic_voice(client, chunk, asr):
    asr(text=SCAM)
    registry.set_detectors(audio=MockAudio(0.02))
    sid = client.post("/api/live/start").json()["session_id"]
    j = post_chunk(client, sid, chunk).json(); j = post_chunk(client, sid, chunk).json()
    assert j["assessment"]["text_crosscheck"]["kind"] == "audio_ok_text_suspicious"
    assert j["transcript"]["segments"][0]["indicators"]


def test_live_transcript_unavailable_still_scores(client, chunk, asr):
    asr(status="unavailable")
    sid = client.post("/api/live/start").json()["session_id"]
    j = post_chunk(client, sid, chunk).json()
    assert j["chunk"]["score"] is not None and j["transcript"]["status"] == "unavailable" and j["transcript"]["segments"] == []


def test_live_bad_inputs(client, tmp_path):
    sid = client.post("/api/live/start").json()["session_id"]
    assert client.post("/api/live/nope/audio", files={"file": ("c.webm", b"x" * 500, "audio/webm")}).status_code == 404
    assert client.post(f"/api/live/{sid}/audio", files={"file": ("c.webm", b"x" * 500, "audio/webm")}).status_code == 422   # not audio
    assert client.post(f"/api/live/{sid}/audio", files={"file": ("c.webm", b"", "audio/webm")}).status_code in (400, 422)
    assert client.post(f"/api/live/{sid}/frame", files={"file": ("f.jpg", b"notajpeg", "image/jpeg")}).status_code == 422


def test_live_size_limit(client, monkeypatch):
    monkeypatch.setattr(SETTINGS, "live_max_chunk_mb", 0)
    sid = client.post("/api/live/start").json()["session_id"]
    assert client.post(f"/api/live/{sid}/audio", files={"file": ("c.webm", b"x" * 500, "audio/webm")}).status_code == 413


def test_live_frame_without_face_is_not_scored(client, tmp_path):
    from PIL import Image
    import numpy as np
    jp = tmp_path / "n.jpg"; Image.fromarray((np.random.default_rng(1).random((120, 160, 3)) * 255).astype("uint8")).save(jp)
    fa3 = MockFA3(0.1); registry.set_detectors(fa3=fa3, audio=MockAudio(0.1))
    sid = client.post("/api/live/start").json()["session_id"]
    with open(jp, "rb") as f:
        j = client.post(f"/api/live/{sid}/frame", files={"file": ("f.jpg", f, "image/jpeg")}).json()
    assert j["frame"]["status"] == "no_face" and j["frame"]["score"] is None and fa3.calls == 0


def test_live_frame_scores_and_video_fusion(client, chunk, tmp_path):
    from PIL import Image
    import numpy as np
    from pathlib import Path
    jp = Path(__file__).parent / "assets" / "face_astronaut.jpg"          # real face: live frames without a face are not scored
    registry.set_detectors(fa3=MockFA3(0.9), audio=MockAudio(0.1))
    sid = client.post("/api/live/start").json()["session_id"]
    for _ in range(6):
        with open(jp, "rb") as f:
            j = client.post(f"/api/live/{sid}/frame", files={"file": ("f.jpg", f, "image/jpeg")}).json()
        assert j["frame"]["status"] == "ok" and j["frame"]["score"] == 0.9
    j = post_chunk(client, sid, chunk).json()
    assert j["assessment"]["modalities"]["video"]["verdict"] == "suspected" and j["assessment"]["modalities"]["audio"]["verdict"] in ("likely_authentic", "inconclusive")
    assert j["frames_total"] == 6


def test_live_frame_unavailable_is_reported(client, tmp_path):
    from PIL import Image
    jp = tmp_path / "f.jpg"; Image.new("RGB", (64, 64), "gray").save(jp)

    class Down(MockFA3):
        available = False
        reason = "weights missing"
        def load(self): pass
    registry.set_detectors(fa3=Down())
    sid = client.post("/api/live/start").json()["session_id"]
    with open(jp, "rb") as f:
        j = client.post(f"/api/live/{sid}/frame", files={"file": ("f.jpg", f, "image/jpeg")}).json()
    assert j["frame"]["status"] == "unavailable" and j["frame"]["score"] is None


def test_live_leaves_no_temp_files(client, chunk):
    tmpdir = Path(SETTINGS.data_dir) / "tmp"
    before = set(tmpdir.glob("*")) if tmpdir.exists() else set()
    sid = client.post("/api/live/start").json()["session_id"]
    post_chunk(client, sid, chunk)
    assert set(tmpdir.glob("*")) == before


def test_live_session_cap(client, monkeypatch):
    from app.services import live
    monkeypatch.setattr(live, "MAX_SESSIONS", len(live._SESSIONS))
    assert client.post("/api/live/start").status_code == 503
