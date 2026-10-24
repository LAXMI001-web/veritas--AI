"""Model health probe: a collapsed (constant-output) audio model must not yield 'likely authentic'.
REAL: the shipped cnn_conformer.h5 is probed (it is in fact constant). MOCKED: fusion/API cases with injected health."""
import time

import numpy as np
import pytest

from app import registry
from app.config import SETTINGS, Policy
from app.services import fusion
from tests.conftest import MockAudio, MockFA3

SUSPECT = {"status": "suspect", "message": "constant classifier (test)"}


def audio_res(score=0.0, health=None):
    return {"status": "ok", "score": score, "scored": 5, "duration_s": 6.0, "aggregation_method": "mean",
            "dispersion": {"std": 0.0}, "intervals": [], "model_health": health}


def test_suspect_health_forces_inconclusive():
    v = fusion.modality_verdict(audio_res(0.0, SUSPECT), Policy(), "audio")
    assert v["verdict"] == "inconclusive" and v["model_degenerate"] and "health check failed" in v["reasons"][0]
    v2 = fusion.modality_verdict(audio_res(0.99, SUSPECT), Policy(), "audio")
    assert v2["verdict"] == "inconclusive"              # also never "suspected"


def test_override_env_shows_raw_verdict(monkeypatch):
    monkeypatch.setattr(SETTINGS, "audio_allow_degenerate", True)
    assert fusion.modality_verdict(audio_res(0.0, SUSPECT), Policy(), "audio")["verdict"] == "likely_authentic"


def test_ok_health_does_not_change_verdict():
    assert fusion.modality_verdict(audio_res(0.0, {"status": "ok"}), Policy(), "audio")["verdict"] == "likely_authentic"


def test_api_audio_with_degenerate_model_is_inconclusive(client, media):
    class Bad(MockAudio):
        health = SUSPECT

        def score_windows(self, audio, hop_s=1.024, batch=32):
            return super().score_windows(audio, hop_s, batch)
    registry.set_detectors(audio=Bad(0.0))
    with open(media["audio_wav"], "rb") as f:
        jid = client.post("/api/analyze", files={"file": ("a.wav", f, "audio/wav")}, data={"expect": "audio"}).json()["job_id"]
    for _ in range(300):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.2)
    rec = client.get(f"/api/analyses/{j['analysis_id']}").json()
    assert rec["assessment"]["overall"] == "inconclusive"
    assert rec["assessment"]["text_crosscheck"]["kind"] == "not_comparable"


@pytest.mark.real_audio
def test_real_checkpoint_is_flagged_as_constant():
    from app.config import ROOT
    from app.models.audio_model import AudioDetector
    d = AudioDetector(model_path=str(ROOT / "models" / "cnn_conformer.h5"))     # the ORIGINAL ScamGuard checkpoint
    d.load()
    assert d.available and d.health["status"] == "suspect"
    assert all(p < 1e-3 for p in d.health["probe_outputs"].values())


def test_probe_does_not_flag_a_discriminative_model():
    """A tiny model whose output really depends on the input must be 'ok' (probe has no false positive here)."""
    import tensorflow as tf
    from app.models.audio_model import AudioDetector
    inp = tf.keras.Input((128, 128, 1))
    x = tf.keras.layers.GlobalAveragePooling2D()(inp)
    out = tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer=tf.keras.initializers.Constant(12.0),
                                bias_initializer=tf.keras.initializers.Constant(-6.0))(x)
    d = AudioDetector(model_path="x.h5")
    d._model = tf.keras.Model(inp, out)
    assert d._probe_health()["status"] == "ok"


def test_import_order_guard_prevents_tf_torch_segfault():
    """Regression: TensorFlow imported before transformers' CLIP code segfaults this environment. `import app` must make
    the later order (TF first, then FA3 libs) safe. Runs in a subprocess so a crash fails the test instead of pytest."""
    import subprocess
    import sys
    from pathlib import Path
    code = ("import app\nimport tensorflow\nfrom transformers import CLIPConfig, CLIPModel\n"
            "m = CLIPModel(CLIPConfig())\nprint('ORDER_OK')")
    r = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=240)
    assert r.returncode == 0 and "ORDER_OK" in r.stdout, (r.returncode, r.stderr[-400:])
