"""REAL audio models (no mocks): the original ScamGuard checkpoint and the default FakeAVCeleb model.
Checks loading, shapes, determinism, silence handling and the validation gate — NOT detection accuracy (no labelled data here)."""
import time

import numpy as np
import pytest

from app import registry
from app.config import ROOT
from app.models.audio_model import AudioDetector, WINDOW_SAMPLES, preprocess_window
from app.models.audio_specs import load_spec, window_features

pytestmark = pytest.mark.real_audio


@pytest.fixture(scope="module")
def legacy():
    d = AudioDetector(model_path=str(ROOT / "models" / "cnn_conformer.h5"))
    d.load()
    if not d.available:
        pytest.skip(d.reason)
    return d


@pytest.fixture(scope="module")
def det():
    d = AudioDetector()           # default spec: FakeAVCeleb cnn_conformer, unvalidated (no selection file in tests)
    d.load()
    if not d.available:
        pytest.skip(d.reason)
    return d


def test_legacy_model_loads_exactly(legacy):
    assert legacy.param_count == 114433 and legacy.spec.input_shape == (128, 128, 1)


def test_legacy_preprocess_matches_original_shape_range():
    x = preprocess_window(np.random.default_rng(0).standard_normal(WINDOW_SAMPLES).astype("float32") * 0.1)
    assert x.shape == (1, 128, 128, 1) and 0 <= x.min() and x.max() <= 1


def test_spec_driven_legacy_features_equal_original_function():
    """The generic feature extractor, configured as the legacy spec, must reproduce the original preprocessing exactly."""
    from app.models.audio_specs import LEGACY
    a = np.random.default_rng(1).standard_normal(WINDOW_SAMPLES).astype("float32") * 0.1
    np.testing.assert_allclose(window_features(a, LEGACY)[None], preprocess_window(a), atol=1e-5)


def test_default_spec_is_fakeavceleb_and_unvalidated():
    sp = load_spec()
    assert sp.key.endswith("fakeavceleb") and sp.input_shape == (128, 94, 1) and not sp.validated


def test_new_model_loads_and_shapes(det):
    assert det._model.input_shape[1:] == (128, 94, 1) and det.window_samples == 94 * 512 and det.info()["validated"] is False
    f = window_features(np.random.default_rng(0).standard_normal(det.window_samples).astype("float32") * .1, det.spec)
    assert f.shape == (128, 94, 1) and np.isfinite(f).all()


@pytest.mark.parametrize("which", ["legacy", "det"])
def test_windows_deterministic_and_bounded(which, request):
    d = request.getfixturevalue(which)
    t = np.arange(16000 * 7) / 16000
    a = (0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype("float32")
    w1, w2 = d.score_windows(a), d.score_windows(a)
    assert [w["score"] for w in w1] == [w["score"] for w in w2]
    assert all(0 <= w["score"] <= 1 for w in w1 if w["score"] is not None)
    assert w1[-1]["end"] == pytest.approx(7.0, abs=0.01)  # tail window covers the end


@pytest.mark.parametrize("which", ["legacy", "det"])
def test_silence_skipped(which, request):
    d = request.getfixturevalue(which)
    w = d.score_windows(np.zeros(16000 * 4, dtype="float32"))
    assert all(x["silent"] and x["score"] is None for x in w)


def test_end_to_end_unvalidated_model_gives_inconclusive_not_authentic(client, media, det):
    registry.set_detectors(audio=det)
    with open(media["audio_wav"], "rb") as f:
        jid = client.post("/api/analyze", files={"file": ("a.wav", f.read())}).json()["job_id"]
    for _ in range(200):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.3)
    assert j["status"] == "done"
    r = client.get(f"/api/analyses/{j['analysis_id']}").json()
    a = r["results"]["audio"]
    assert a["model"].startswith("cnn_conformer_fakeavceleb") and a["spectrogram_png"] and a["score"] is not None
    assert r["assessment"]["modalities"]["audio"]["verdict"] == "inconclusive"
    assert "not validated" in r["assessment"]["modalities"]["audio"]["reasons"][0]
    assert r["assessment"]["overall"] == "inconclusive"
