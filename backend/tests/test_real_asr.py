"""REAL inference: Whisper (sherpa-onnx) + the real cnn_conformer.h5, on synthetic speech made with espeak-ng.
Skipped automatically if the ASR model files or the espeak helper package are missing.
NOTE: espeak speech is robotic; this tests the transcript/cross-check plumbing, NOT detection accuracy."""
import ctypes
import time
import wave

import numpy as np
import pytest

from app.services import transcript as transcript_svc

espeakng_loader = pytest.importorskip("espeakng_loader")
pytestmark = pytest.mark.real_asr

SCAM = ("Hello this is the bank security department. Your account has been compromised. You must share the one time "
        "password right now and transfer the money to a safe account immediately. Do not tell anyone.")
BENIGN = "Hello, thanks for calling. I wanted to confirm our meeting tomorrow at ten in the morning in the main office."


def tts(path, text):
    lib = ctypes.CDLL(espeakng_loader.get_library_path())
    cb_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)
    buf = []

    def cb(w, n, ev):
        if w and n > 0:
            buf.append(np.ctypeslib.as_array(w, (n,)).copy())
        return 0
    keep = cb_t(cb)
    sr = lib.espeak_Initialize(1, 0, espeakng_loader.get_data_path().encode(), 0)
    lib.espeak_SetSynthCallback(keep)
    lib.espeak_ng_SetVoiceByName.argtypes = [ctypes.c_char_p]
    lib.espeak_ng_SetVoiceByName(b"en-us")
    t = text.encode()
    lib.espeak_Synth(t, len(t) + 1, 0, 0, 0, 0x1, None, None)
    lib.espeak_Synchronize()
    a = np.concatenate(buf).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(a.tobytes())


@pytest.fixture(scope="module")
def real_asr():
    t = transcript_svc.Transcriber()
    t.load()
    if not t.available:
        pytest.skip(t.reason)
    return t


def test_real_whisper_transcribes_and_flags(real_asr, tmp_path):
    from app.config import Policy
    from app.services import textcheck
    p = tmp_path / "s.wav"
    tts(p, SCAM)
    transcript_svc.set_transcriber(real_asr)
    tr = transcript_svc.transcribe_file(p)
    low = tr["text"].lower().replace("-", " ")      # Whisper may write "one-time" or "one time"
    assert tr["status"] == "ok" and "bank security" in low and "one time password" in low
    assert textcheck.analyze(tr, Policy())["verdict"] == "suspicious_wording"
    p2 = tmp_path / "b.wav"
    tts(p2, BENIGN)
    tb = transcript_svc.transcribe_file(p2)
    assert textcheck.analyze(tb, Policy())["verdict"] == "ordinary_wording"


def test_real_models_end_to_end_via_api(real_asr, tmp_path):
    """Real audio model + real Whisper through /api/analyze. Asserts structure, not the (out-of-domain) audio score."""
    from fastapi.testclient import TestClient
    from app import registry
    from app.main import app
    from app.models.audio_model import AudioDetector
    registry.set_detectors(audio=AudioDetector())
    transcript_svc.set_transcriber(real_asr)
    p = tmp_path / "s.wav"
    tts(p, SCAM)
    c = TestClient(app)
    with open(p, "rb") as f:
        jid = c.post("/api/analyze", files={"file": ("s.wav", f, "audio/wav")}, data={"expect": "audio"}).json()["job_id"]
    for _ in range(600):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert j["status"] == "done", j
    rec = c.get(f"/api/analyses/{j['analysis_id']}").json()
    assert rec["results"]["audio"]["status"] == "ok" and rec["results"]["transcript"]["status"] == "ok"
    assert rec["assessment"]["text_crosscheck"]["kind"] in (
        "audio_flagged_text_ordinary", "audio_ok_text_suspicious", "both_flagged", "not_comparable")
    assert rec["results"]["text_check"]["verdict"] == "suspicious_wording"
