"""HF audio backend plumbing. Uses a TINY RANDOMLY-INITIALISED wav2vec2 built locally (no download) — verifies loading, label
resolution, scoring mechanics and pipeline wiring, NOT detection quality."""
import json

import numpy as np
import pytest

pytest.importorskip("transformers")
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification  # noqa: E402

from app import registry  # noqa: E402
from app.models.hf_audio import HFAudioDetector, resolve_fake_index  # noqa: E402


def tiny(path, id2label, meta=None):
    cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, conv_dim=(16, 16, 16),
                         conv_stride=(5, 4, 4), conv_kernel=(10, 4, 4), num_feat_extract_layers=3, num_labels=2,
                         id2label=id2label, label2id={v: k for k, v in id2label.items()}, classifier_proj_size=16, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2)
    import torch
    torch.manual_seed(0)
    Wav2Vec2ForSequenceClassification(cfg).save_pretrained(path)
    Wav2Vec2FeatureExtractor(sampling_rate=16000).save_pretrained(path)
    if meta:
        (path / "veritas_meta.json").write_text(json.dumps(meta))
    return path


def test_resolve_labels():
    assert resolve_fake_index({0: "real", 1: "fake"}, None, None)[0] == 1
    assert resolve_fake_index({0: "fake", 1: "bonafide"}, None, None)[0] == 0
    assert resolve_fake_index({0: "spoof", 1: "bonafide"}, None, None)[0] == 0
    assert resolve_fake_index({0: "human", 1: "AI"}, None, None)[0] == 1
    i, _, err = resolve_fake_index({0: "LABEL_0", 1: "LABEL_1"}, None, None)
    assert i is None and "Cannot tell" in err                       # ambiguous -> refuse
    assert resolve_fake_index({0: "LABEL_0", 1: "LABEL_1"}, 1, None)[0] == 1
    i, _, err = resolve_fake_index({0: "real", 1: "fake"}, 0, None)
    assert i is None and "disagree" in err                           # conflicting sources -> refuse


def test_missing_model_is_unavailable(tmp_path):
    d = HFAudioDetector(str(tmp_path / "none")); d.load()
    assert not d.available and "fetch_audio_model.py" in d.reason


def test_loads_scores_and_skips_silence(tmp_path):
    d = HFAudioDetector(str(tiny(tmp_path / "m", {0: "real", 1: "fake"}))); d.load()
    assert d.available and d.fake_idx == 1 and d.class_mapping["source"] == "config id2label"
    t = np.arange(16000 * 9) / 16000
    a = (0.3 * np.sin(2 * np.pi * 180 * t)).astype("float32")
    w = d.score_windows(a)
    assert all(0 <= x["score"] <= 1 for x in w if x["score"] is not None) and w[-1]["end"] == pytest.approx(9.0, abs=0.01)
    assert w == d.score_windows(a)                                  # deterministic
    assert all(x["silent"] and x["score"] is None for x in d.score_windows(np.zeros(16000 * 5, dtype="float32")))
    assert d.info()["validation"]["external"] is True


def test_label_flip_inverts_score(tmp_path):
    m = tiny(tmp_path / "m", {0: "real", 1: "fake"})
    a = (0.3 * np.sin(2 * np.pi * 180 * np.arange(16000 * 4) / 16000)).astype("float32")
    d1 = HFAudioDetector(str(m)); d1.load(); s1 = d1.score_windows(a)[0]["score"]
    cfg = json.loads((m / "config.json").read_text()); cfg["id2label"] = {"0": "fake", "1": "real"}; cfg["label2id"] = {"fake": 0, "real": 1}
    (m / "config.json").write_text(json.dumps(cfg))
    d2 = HFAudioDetector(str(m)); d2.load(); s2 = d2.score_windows(a)[0]["score"]
    assert d2.fake_idx == 0 and s1 + s2 == pytest.approx(1.0, abs=2e-5)


def test_ambiguous_labels_refused_without_meta(tmp_path):
    m = tiny(tmp_path / "m", {0: "LABEL_0", 1: "LABEL_1"})
    d = HFAudioDetector(str(m)); d.load()
    assert not d.available and "Cannot tell" in d.reason
    (m / "veritas_meta.json").write_text(json.dumps({"fake_index": 1}))
    d2 = HFAudioDetector(str(m)); d2.load()
    assert d2.available and d2.fake_idx == 1


def test_registry_auto_and_end_to_end(tmp_path, monkeypatch, client, media):
    import time
    from app.config import SETTINGS
    m = tiny(tmp_path / "default", {0: "real", 1: "fake"})
    monkeypatch.setattr(SETTINGS, "audio_hf_dir", str(m)); monkeypatch.setattr(SETTINGS, "audio_backend", "auto")
    monkeypatch.setattr(registry, "_audio", None)
    assert isinstance(registry.get_audio(), HFAudioDetector)
    with open(media["audio_wav"], "rb") as f:
        jid = client.post("/api/analyze", files={"file": ("a.wav", f.read())}).json()["job_id"]
    for _ in range(200):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.3)
    assert j["status"] == "done"
    au = client.get(f"/api/analyses/{j['analysis_id']}").json()
    r = au["results"]["audio"]; v = au["assessment"]["modalities"]["audio"]
    assert r["model"].startswith("HF-audio") and r["model_validation"]["external"] and r["score"] is not None
    if r["model_health"]["status"] == "suspect":          # a randomly initialised tiny net is ~constant: the probe must catch it
        assert v["verdict"] == "inconclusive" and v.get("model_degenerate")
    else:
        assert any("own authors" in w for w in v.get("warnings", []))
    monkeypatch.setattr(registry, "_audio", None)


def test_invert_switch_swaps_fake_output(tmp_path, monkeypatch):
    from app.config import SETTINGS
    m = tiny(tmp_path / "m", {0: "real", 1: "fake"})
    a = (0.3 * np.sin(2 * np.pi * 180 * np.arange(16000 * 4) / 16000)).astype("float32")
    d1 = HFAudioDetector(str(m)); d1.load(); s1 = d1.score_windows(a)[0]["score"]
    monkeypatch.setattr(SETTINGS, "audio_hf_invert", True)
    d2 = HFAudioDetector(str(m)); d2.load(); s2 = d2.score_windows(a)[0]["score"]
    assert d1.fake_idx == 1 and d2.fake_idx == 0 and "INVERTED" in d2.label_source and s1 + s2 == pytest.approx(1.0, abs=2e-5)
    assert d2.class_mapping["source"].endswith("INVERTED by AUDIO_HF_INVERT")
