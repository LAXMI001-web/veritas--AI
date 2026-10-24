"""Audio model selection + validation gate. Pure-function tests use made-up numbers (MOCKED); the smoke test runs the REAL
models on SYNTHETIC tones/noise to prove the pipeline writes a valid selection file — it says nothing about accuracy."""
import json
import subprocess

import numpy as np
import pytest
import soundfile as sf

from app.config import Policy
from app.services import audio_selection as sel, fusion


def test_platt_flat_for_uninformative_scores():
    rng = np.random.default_rng(0)
    p = np.full(200, 0.999)                     # a saturated/constant model
    y = (rng.random(200) < 0.3).astype(float)
    cal = sel.platt_fit(p, y)
    out = sel.apply_platt(p, cal)
    assert abs(out.mean() - y.mean()) < 0.05    # calibrated to the base rate, not overconfident


def test_platt_monotonic_and_calibrates_separable_scores():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.5).astype(float)
    p = np.clip(np.where(y == 1, rng.beta(5, 2, 400), rng.beta(2, 5, 400)), 0.001, 0.999)
    cal = sel.platt_fit(p, y)
    assert cal["a"] > 0
    s = sel.apply_platt(np.array([0.2, 0.5, 0.8]), cal)
    assert s[0] < s[1] < s[2]


def test_metrics_and_confusion_matrix_exact():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    s = np.array([.9, .8, .2, .1, .3, .6, .2, .1])
    m = sel.metrics_at(s, y, 0.5)
    assert m["confusion_matrix"] == {"tp_fake_caught": 2, "fn_fake_missed_as_real": 1, "fp_real_flagged_fake": 1, "tn_real_ok": 4}
    assert m["accuracy"] == 0.75 and m["recall"] == pytest.approx(0.6667, abs=1e-3) and m["false_positive_rate"] == 0.2
    assert m["false_negative_rate"] == pytest.approx(0.3333, abs=1e-3) and m["auc"] is not None


def test_youden_threshold_separates():
    y = np.array([0] * 10 + [1] * 10)
    s = np.concatenate([np.linspace(0.05, 0.35, 10), np.linspace(0.65, 0.95, 10)])
    t = sel.youden_threshold(s, y)
    assert 0.35 < t <= 0.65


def test_split_keeps_groups_together_and_is_deterministic(tmp_path):
    from pathlib import Path
    files = {"real": [Path(f"spk{i % 5}_r{i}.wav") for i in range(20)], "fake": [Path(f"spk{i % 5}_f{i}.wav") for i in range(20)]}
    v1, t1, _ = sel.split_flat(files, r"(spk\d)")
    v2, t2, _ = sel.split_flat(files, r"(spk\d)")
    assert v1 == v2 and t1 == t2
    vs = {f.name[:4] for c in v1.values() for f in c}; ts = {f.name[:4] for c in t1.values() for f in c}
    assert vs.isdisjoint(ts)                    # no speaker in both splits


def test_split_warns_without_group_regex():
    from pathlib import Path
    _, _, notes = sel.split_flat({"real": [Path("a.wav")], "fake": [Path("b.wav")]}, None)
    assert notes and "optimistic" in notes[0]


def test_validation_gate_in_fusion(monkeypatch):
    from app.config import SETTINGS
    res = {"status": "ok", "score": 0.95, "scored": 5, "duration_s": 9.0, "aggregation_method": "mean", "dispersion": {"std": 0.0}, "intervals": []}
    unval = fusion.modality_verdict({**res, "model_validation": {"validated": False}}, Policy(), "audio")
    assert unval["verdict"] == "inconclusive" and unval["model_unvalidated"]
    ok = fusion.modality_verdict({**res, "model_validation": {"validated": True, "threshold": 0.5}}, Policy(), "audio")
    assert ok["verdict"] == "suspected"
    monkeypatch.setattr(SETTINGS, "audio_allow_unvalidated", True)
    assert fusion.modality_verdict({**res, "model_validation": {"validated": False}}, Policy(), "audio")["verdict"] == "suspected"


def test_validated_model_uses_its_own_measured_threshold():
    res = {"status": "ok", "score": 0.7, "scored": 5, "duration_s": 9.0, "aggregation_method": "mean", "dispersion": {"std": 0.0}, "intervals": []}
    hi = fusion.modality_verdict({**res, "model_validation": {"validated": True, "threshold": 0.9}}, Policy(), "audio")
    lo = fusion.modality_verdict({**res, "model_validation": {"validated": True, "threshold": 0.3}}, Policy(), "audio")
    assert hi["verdict"] == "likely_authentic" and lo["verdict"] == "suspected"


@pytest.mark.real_audio
def test_selection_smoke_real_models_synthetic_clips(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for split in ("val", "test"):
        for cls in ("real", "fake"):
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(4):
                t = np.arange(16000 * 4) / 16000
                a = (0.3 * np.sin(2 * np.pi * (150 + 20 * i) * t) * (1 + .5 * np.sin(2 * np.pi * 3 * t))) if cls == "real" else 0.2 * rng.standard_normal(len(t))
                sf.write(d / f"{cls}_{i}.wav", a.astype("float32"), 16000)
    out = tmp_path / "selection.json"
    rep = sel.run_selection(tmp_path, out, quick=True, models=["cnn_lstm_fakeavceleb"], log=lambda *_: None)
    s = json.loads(out.read_text())
    assert s["model"] == "cnn_lstm_fakeavceleb" and s["high_means"] in ("fake", "real") and "calibration" in s
    assert s["validated"] is False and "fewer than 10 clips" in s["validation"]["reasons_not_validated"][0]   # too small to validate
    assert set(s["validation"]["test_metrics"]) >= {"accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate", "confusion_matrix", "auc"}
    assert len(rep["candidates"]) == 6
    # the app picks the file up
    monkeypatch.setenv("AUDIO_SELECTION_FILE", str(out))
    from app.config import SETTINGS
    monkeypatch.setattr(SETTINGS, "audio_selection_file", str(out))
    from app.models.audio_specs import load_spec
    sp = load_spec()
    assert sp.key == "cnn_lstm_fakeavceleb" and sp.norm == s["norm"] and sp.high_means == s["high_means"] and sp.calibration == s["calibration"]
