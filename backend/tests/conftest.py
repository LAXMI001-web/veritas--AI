"""Test fixtures. NOTE: FA3-CLIP is replaced by MockFA3 (deterministic fake scores) in these tests because the real
checkpoint is not available in CI. The audio model tests marked `real_audio` use the REAL cnn_conformer.h5."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_tmp = tempfile.mkdtemp(prefix="veritas_test_")
os.environ["VERITAS_DATA_DIR"] = _tmp
os.environ["RATE_LIMIT_PER_MIN"] = "100000"
os.environ["AUDIO_SELECTION_FILE"] = os.path.join(_tmp, "no_selection.json")   # tests must not depend on a local selection file
os.environ.pop("AUDIO_MODEL", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import registry  # noqa: E402


class MockFA3:
    """Returns scores supplied by the test (callable or constant). MOCKED OUTPUT — not model performance."""
    name = "MockFA3"
    available = True
    reason = None

    def __init__(self, score=0.1):
        self.score = score
        self.calls = 0

    def load(self): pass

    def info(self):
        return {"name": self.name, "available": True, "reason": None, "mock": True}

    def score_images(self, images, batch=8):
        out = []
        for im in images:
            self.calls += 1
            s = self.score(self.calls - 1) if callable(self.score) else self.score
            out.append({"fake_prob": s, "real_prob": 1 - s, "logits": [s, 1 - s]})
        return out


class MockAudio:
    name = "MockAudio"
    available = True
    reason = None

    def __init__(self, score=0.1, fail=False):
        self.score, self.fail = score, fail

    def load(self): pass

    def info(self):
        return {"name": self.name, "available": True, "reason": None, "mock": True}

    def score_windows(self, audio, hop_s=1.024, batch=32):
        if self.fail:
            raise RuntimeError("boom")
        from app.models.audio_model import WINDOW_SAMPLES, SAMPLE_RATE
        n = len(audio)
        starts = list(range(0, max(n - WINDOW_SAMPLES, 0) + 1, int(hop_s * SAMPLE_RATE))) or [0]
        return [{"start": s / SAMPLE_RATE, "end": min(s + WINDOW_SAMPLES, n) / SAMPLE_RATE,
                 "score": self.score(i) if callable(self.score) else self.score, "raw": 0.0, "silent": False,
                 "padded": n < WINDOW_SAMPLES, "rms_dbfs": -20.0} for i, s in enumerate(starts)]


class MockASR:
    """Returns the supplied text as one segment spanning the audio. MOCKED OUTPUT — not transcription accuracy."""
    name = "MockASR"
    available = True
    reason = None

    def __init__(self, text="Hello, thanks for calling. I wanted to confirm our meeting tomorrow morning in the main office.",
                 confidence=0.9, status="ok"):
        self.text, self.confidence, self.status = text, confidence, status

    def load(self): pass

    def info(self):
        return {"name": "MockASR", "available": self.status != "unavailable", "reason": None, "mock": True}

    def transcribe_audio(self, audio, offset_s=0.0):
        if self.status == "unavailable":
            return {"status": "unavailable", "reason": "ASR model not installed (mock).", "modality": "transcript"}
        if self.status == "failed":
            return {"status": "failed", "reason": "Transcription failed: boom", "modality": "transcript"}
        dur = len(audio) / 16000
        segs = [{"start": round(offset_s, 2), "end": round(offset_s + dur, 2), "text": self.text, "confidence": self.confidence}] if self.text else []
        return {"status": "ok", "modality": "transcript", "engine": "MockASR", "language": "en", "segments": segs,
                "text": self.text, "word_count": len(self.text.split()), "mean_confidence": self.confidence if segs else None,
                "note": "MOCK transcript for tests."}


def _ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="session")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    m = {}
    m["video_av"] = d / "av.mp4"
    _ff("-f", "lavfi", "-i", "testsrc=duration=6:size=320x240:rate=15", "-f", "lavfi", "-i", "sine=frequency=220:duration=6",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(m["video_av"]))
    m["video_noaudio"] = d / "na.mp4"
    _ff("-f", "lavfi", "-i", "testsrc=duration=6:size=320x240:rate=15", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(m["video_noaudio"]))
    # real face (public-domain NASA photo of astronaut Eileen Collins, from scikit-image's data set) panning slowly so frames differ
    face = Path(__file__).parent / "assets" / "face_astronaut.jpg"
    vf = "crop=w=iw*0.8:h=ih*0.8:x='(iw-ow)*t/6':y='(ih-oh)*t/6',scale=320:320,format=yuv420p"
    m["video_face"] = d / "face.mp4"
    _ff("-loop", "1", "-framerate", "15", "-t", "6", "-i", str(face), "-vf", vf, "-c:v", "libx264", str(m["video_face"]))
    m["video_face_av"] = d / "face_av.mp4"
    _ff("-loop", "1", "-framerate", "15", "-t", "6", "-i", str(face), "-f", "lavfi", "-i", "sine=frequency=220:duration=6", "-vf", vf,
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(m["video_face_av"]))
    m["audio_wav"] = d / "a.wav"
    _ff("-f", "lavfi", "-i", "sine=frequency=300:duration=5", str(m["audio_wav"]))
    m["audio_short"] = d / "short.wav"
    _ff("-f", "lavfi", "-i", "sine=frequency=300:duration=0.5", str(m["audio_short"]))
    m["image"] = d / "i.png"
    from PIL import Image
    rng = np.random.default_rng(0)
    Image.fromarray((rng.random((120, 160, 3)) * 255).astype("uint8")).save(m["image"])
    m["image2"] = d / "i2.png"
    Image.fromarray((rng.random((120, 160, 3)) * 255).astype("uint8")).save(m["image2"])
    m["corrupt"] = d / "bad.mp4"
    m["corrupt"].write_bytes(m["video_av"].read_bytes()[:3000] + b"\x00" * 500)
    m["text"] = d / "x.txt"
    m["text"].write_text("not media")
    return m


@pytest.fixture(autouse=True)
def mocks():
    from app.services import transcript as transcript_svc
    fa3, au = MockFA3(0.1), MockAudio(0.1)
    registry.set_detectors(fa3=fa3, audio=au)
    transcript_svc.set_transcriber(MockASR())
    return fa3, au


@pytest.fixture
def asr():
    """Replace the mock transcriber with a configured one: asr(text=..., status=...)."""
    from app.services import transcript as transcript_svc

    def make(**kw):
        t = MockASR(**kw)
        transcript_svc.set_transcriber(t)
        return t
    return make


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)
