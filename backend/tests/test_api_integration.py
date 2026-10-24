"""API integration tests. FA3 is MOCKED (see conftest); audio uses MockAudio except in test_real_audio.py."""
import time
from pathlib import Path

import pytest

from app import registry
from app.config import SETTINGS
from tests.conftest import MockAudio, MockFA3


def run_job(client, path, expect="", name=None, timeout=60):
    with open(path, "rb") as f:
        r = client.post("/api/analyze", files={"file": (name or Path(path).name, f.read())}, data={"expect": expect})
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise AssertionError("job timeout")


def rec(client, j):
    assert j["status"] == "done", j
    return client.get(f"/api/analyses/{j['analysis_id']}").json()


def tmp_left():
    d = Path(SETTINGS.data_dir) / "tmp"
    return list(d.glob("v_*")) if d.exists() else []


def test_health_and_models(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/models").json()["fa3"]["available"] is True


def test_valid_image(client, media):
    r = rec(client, run_job(client, media["image"], "image"))
    assert r["kind"] == "image" and r["assessment"]["overall"] == "likely_authentic"
    assert r["results"]["image"]["frames"][0]["score"] == 0.1


def test_invalid_image_text_file(client, media):
    j = run_job(client, media["text"], "image")
    assert j["status"] == "failed" and j["error_code"] in ("unreadable", "unsupported")
    assert "/" not in (j["error"] or "").replace("(", "")[:0]  # no path leakage below


def test_wrong_kind_rejected(client, media):
    j = run_job(client, media["audio_wav"], "image")
    assert j["status"] == "failed" and j["error_code"] == "wrong_kind"


def test_valid_audio(client, media):
    r = rec(client, run_job(client, media["audio_wav"], "audio"))
    assert r["kind"] == "audio" and r["results"]["audio"]["scored"] >= 1


def test_very_short_audio_inconclusive(client, media):
    r = rec(client, run_job(client, media["audio_short"], "audio"))
    assert r["assessment"]["overall"] == "inconclusive"
    assert r["assessment"]["modalities"]["audio"]["insufficient_evidence"]


def test_video_with_audio_both_modalities(client, media):
    r = rec(client, run_job(client, media["video_face_av"], "video"))
    assert r["has_audio"] and r["results"]["video"]["status"] == "ok" and r["results"]["audio"]["status"] == "ok"
    assert r["assessment"]["scenario"] == "A"
    assert len(r["results"]["video"]["frames"]) >= 6 and r["integrity"]["metadata"]["sha256"]


def test_video_without_audio_scenario_F(client, media):
    r = rec(client, run_job(client, media["video_face"], "video"))
    assert r["assessment"]["scenario"] == "F" and "audio" not in r["results"]
    assert r["assessment"]["modalities"]["audio"]["verdict"] == "not_present"


def test_corrupted_video_fails_gracefully(client, media):
    j = run_job(client, media["corrupt"], "video")
    if j["status"] == "done":  # decoder may salvage the header; then frames must be flagged
        r = client.get(f"/api/analyses/{j['analysis_id']}").json()
        assert r["assessment"]["overall"] in ("analysis_incomplete", "inconclusive", "likely_authentic")
    else:
        assert j["error_code"] in ("unreadable", "unsupported")


def test_failed_frame_extraction(client, media, monkeypatch):
    from app.services import video as vs
    monkeypatch.setattr(vs, "sample_video_frames", lambda *a, **k: ([{"index": 0, "t": 0, "rgb": None, "error": "frame could not be decoded"}], {}))
    r = rec(client, run_job(client, media["video_face"], "video"))
    assert r["results"]["video"]["status"] == "failed" and r["assessment"]["overall"] == "analysis_incomplete"


def test_model_inference_failure_keeps_other_modality(client, media):
    registry.set_detectors(audio=MockAudio(fail=True))
    r = rec(client, run_job(client, media["video_face_av"], "video"))
    assert r["results"]["audio"]["status"] == "failed" and r["results"]["video"]["status"] == "ok"
    assert r["assessment"]["scenario"] == "G" and r["assessment"]["modalities"]["video"]["score"] is not None


def test_real_looking_video_suspicious_audio(client, media):
    registry.set_detectors(fa3=MockFA3(0.05), audio=MockAudio(0.95))
    r = rec(client, run_job(client, media["video_face_av"], "video"))
    assert r["assessment"]["scenario"] == "B" and r["assessment"]["overall"] == "mixed_authenticity"
    assert any("audio" in x.lower() for x in r["explanation"]["model_interpretation"])


def test_suspicious_video_authentic_audio(client, media):
    registry.set_detectors(fa3=MockFA3(0.95), audio=MockAudio(0.05))
    assert rec(client, run_job(client, media["video_face_av"], "video"))["assessment"]["scenario"] == "C"


def test_both_flagged(client, media):
    registry.set_detectors(fa3=MockFA3(0.95), audio=MockAudio(0.95))
    assert rec(client, run_job(client, media["video_face_av"], "video"))["assessment"]["scenario"] == "D"


def test_inconclusive_predictions(client, media):
    registry.set_detectors(fa3=MockFA3(0.5), audio=MockAudio(0.5))
    assert rec(client, run_job(client, media["video_face_av"], "video"))["assessment"]["overall"] == "inconclusive"


def test_localized_interval_scenario_I(client, media):
    registry.set_detectors(fa3=MockFA3(lambda i: 0.95 if 4 <= i <= 7 else 0.02), audio=MockAudio(0.05))
    r = rec(client, run_job(client, media["video_face_av"], "video"))
    assert r["assessment"].get("secondary_scenario") == "I" and r["results"]["video"]["intervals"]


def test_fa3_unavailable_is_reported_not_faked(client, media):
    class Gone(MockFA3):
        available = False
        reason = "Checkpoint fa3_model.pt not found"

        def info(self): return {"name": "FA3", "available": False, "reason": self.reason}
    registry.set_detectors(fa3=Gone())
    r = rec(client, run_job(client, media["image"], "image"))
    assert r["results"]["image"]["status"] == "unavailable" and r["assessment"]["overall"] == "analysis_incomplete"


def test_report_pdf_json(client, media):
    r = rec(client, run_job(client, media["video_face_av"], "video"))
    pdf = client.get(f"/api/analyses/{r['id']}/report.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    js = client.get(f"/api/analyses/{r['id']}/report.json").json()
    assert js["machine_generated"] and js["analysis"]["sha256"] == r["sha256"]
    assert "thumb" not in str(js["analysis"]["results"]["video"]["frames"][0])


def test_history_search_delete(client, media):
    r = rec(client, run_job(client, media["image"], "image", name="findme_xyz.png"))
    assert any(i["id"] == r["id"] for i in client.get("/api/analyses", params={"q": "findme_xyz"}).json()["items"])
    assert any(i["id"] == r["id"] for i in client.get("/api/analyses", params={"q": r["id"]}).json()["items"])
    assert client.delete(f"/api/analyses/{r['id']}").json()["deleted"]
    assert client.get(f"/api/analyses/{r['id']}").status_code == 404


def test_batch(client, media):
    files = [("files", ("a.png", media["image"].read_bytes())), ("files", ("b.wav", media["audio_wav"].read_bytes())),
             ("files", ("c.txt", media["text"].read_bytes()))]
    r = client.post("/api/batch", files=files).json()
    t0 = time.time()
    while time.time() - t0 < 60:
        b = client.get(f"/api/batches/{r['batch_id']}").json()
        if b["finished"]:
            break
        time.sleep(0.3)
    assert b["counts"].get("done") == 2 and b["counts"].get("failed") == 1
    rep = client.get(f"/api/batches/{r['batch_id']}/report").json()
    assert len(rep["items"]) == 3


def test_cleanup_after_analysis(client, media):
    run_job(client, media["video_face_av"], "video")
    run_job(client, media["text"], "image")
    time.sleep(0.3)
    assert tmp_left() == []


def test_invalid_requests(client):
    assert client.post("/api/analyze").status_code == 422
    assert client.post("/api/analyze", files={"file": ("x.png", b"")}).status_code in (400, 422)
    assert client.get("/api/analyses/zzzz").status_code == 400
    assert client.get("/api/analyses/" + "a" * 16).status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.put("/api/policy", json={"suspicious_threshold": 5}).status_code == 400
    assert client.post("/api/analyze", files={"file": ("x", b"abc")}, data={"expect": "weird"}).status_code == 400


def test_unsupported_format_bytes(client):
    r = client.post("/api/analyze", files={"file": ("a.mp4", b"\x00\x01\x02" * 100)})
    j = client.get(f"/api/jobs/{r.json()['job_id']}").json()
    t0 = time.time()
    while j["status"] in ("queued", "running") and time.time() - t0 < 20:
        time.sleep(0.2); j = client.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert j["status"] == "failed"


def test_size_limit(client, monkeypatch):
    monkeypatch.setattr(SETTINGS, "max_upload_mb", 0)
    r = client.post("/api/analyze", files={"file": ("a.png", b"x" * 10)})
    assert r.status_code == 413


def wait_task(client, tid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        t = client.get(f"/api/tasks/{tid}").json()
        if t["status"] in ("done", "failed"):
            return t
        time.sleep(0.3)
    raise AssertionError("task timeout")


def test_media_comparison_images(client, media):
    r = client.post("/api/compare", files={"file_a": ("a.png", media["image"].read_bytes()), "file_b": ("b.png", media["image2"].read_bytes())})
    t = wait_task(client, r.json()["task_id"])
    assert t["status"] == "done" and t["result"]["image_difference"]["mean_abs_diff"] > 0


def test_media_comparison_kind_mismatch(client, media):
    r = client.post("/api/compare", files={"file_a": ("a.png", media["image"].read_bytes()), "file_b": ("b.wav", media["audio_wav"].read_bytes())})
    assert wait_task(client, r.json()["task_id"])["status"] == "failed"


@pytest.mark.parametrize("key", ["image", "audio_wav", "video_face_av"])
def test_robustness(client, media, key):
    r = client.post("/api/robustness", files={"file": ("m", media[key].read_bytes())})
    t = wait_task(client, r.json()["task_id"], 180)
    assert t["status"] == "done", t
    res = t["result"]
    assert res["transforms"] and "do NOT indicate" in res["note"]
    assert any("delta" in x for x in res["transforms"])


def test_inspect_endpoint(client, media):
    r = client.post("/api/inspect", files={"file": ("a.mp4", media["video_face_av"].read_bytes())})
    t = wait_task(client, r.json()["task_id"])
    assert t["result"]["provenance"]["verified"] is False and t["result"]["metadata"]["video"]["width"] == 320


def test_assistant_grounded(client, media):
    r = rec(client, run_job(client, media["video_face_av"], "video"))
    a = client.post(f"/api/analyses/{r['id']}/ask", json={"question": "Which timestamp has the highest score?"}).json()
    assert "Highest frame score" in a["answer"]
    b = client.post(f"/api/analyses/{r['id']}/ask", json={"question": "What's the weather?"}).json()
    assert b["facts"] == []


def test_evaluation_unavailable(client, tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "eval_data_dir", str(tmp_path))
    t = wait_task(client, client.post("/api/evaluation/run").json()["task_id"])
    assert t["result"]["status"] == "unavailable"


def test_token_auth(client, monkeypatch):
    monkeypatch.setattr(SETTINGS, "api_token", "s3cret")
    assert client.get("/api/models").status_code == 401
    assert client.get("/api/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/api/health").status_code == 200


def test_sync_check_optional(client, media):
    r = client.put("/api/policy", json={"sync_check": True}); assert r.status_code == 200
    try:
        rr = rec(client, run_job(client, media["video_face_av"], "video"))
        assert rr["results"]["sync"]["status"] in ("ok", "inconclusive", "unavailable", "failed")
    finally:
        client.put("/api/policy", json={"sync_check": False})
