"""Loader MECHANICS test with RANDOM weights (not your trained checkpoint): verifies the offline CLIP build,
state_dict round-trip, preprocessing and output format. Says nothing about detection quality."""
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from PIL import Image  # noqa: E402

from app.models.fa3 import FA3Detector, _offline_clip  # noqa: E402

CK = Path(__file__).resolve().parents[2] / "checkpoints" / "fa3_artifact_faces"


def test_missing_checkpoint_reports_unavailable(tmp_path):
    d = FA3Detector(str(tmp_path)); d.load()
    assert not d.available and "fa3_model.pt" in d.reason
    with pytest.raises(RuntimeError):
        d.score_images([Image.new("RGB", (8, 8))])


def test_random_weight_roundtrip(tmp_path):
    from app.vendor.fa3_model import load_model, build_model
    for f in CK.iterdir():
        shutil.copy(f, tmp_path / f.name)
    with _offline_clip(tmp_path):
        model, _ = build_model()
    model.save_pretrained(tmp_path)
    d = FA3Detector(str(tmp_path)); d.load()
    assert d.available, d.reason
    out = d.score_images([Image.new("RGB", (300, 200), (90, 60, 40)), Image.new("RGB", (64, 64))])
    assert len(out) == 2 and abs(out[0]["fake_prob"] + out[0]["real_prob"] - 1) < 1e-4 and len(out[0]["logits"]) == 2
    assert d.high_pass_map(Image.new("RGB", (64, 64))).shape == (224, 224)


def _random_ckpt(tmp_path, label2id):
    import json
    from app.vendor.fa3_model import build_model
    for f in CK.iterdir():
        shutil.copy(f, tmp_path / f.name)
    with _offline_clip(tmp_path):
        model, _ = build_model()
    model.save_pretrained(tmp_path)
    cfg = json.loads((tmp_path / "fa3_config.json").read_text())
    cfg["label2id"] = label2id
    cfg["id2label"] = {str(v): k for k, v in label2id.items()}
    (tmp_path / "fa3_config.json").write_text(json.dumps(cfg))


def test_class_mapping_is_read_from_checkpoint_config(tmp_path):
    _random_ckpt(tmp_path, {"real": 0, "fake": 1})              # reversed vs. the original FA3 order
    d = FA3Detector(str(tmp_path)); d.load()
    assert d.available, d.reason
    assert d.fake_idx == 1 and d.real_idx == 0 and d.info()["class_mapping"]["fake"] == 1
    out = d.score_images([Image.new("RGB", (64, 64), (10, 20, 30))])[0]
    assert out["fake_prob"] == pytest.approx(__import__("math").exp(out["logits"][1]) / sum(__import__("math").exp(x) for x in out["logits"]), abs=1e-3)


def test_unexpected_class_labels_are_refused(tmp_path):
    _random_ckpt(tmp_path, {"cat": 0, "dog": 1})
    d = FA3Detector(str(tmp_path)); d.load()
    assert not d.available and "class mapping" in d.reason


def test_health_probe_flags_constant_output_and_passes_varying():
    d = FA3Detector("/nonexistent")
    d._score = lambda imgs, batch=8: [{"fake_prob": 0.01} for _ in imgs]
    assert d._probe_health()["status"] == "suspect"
    vals = iter([0.1, 0.7, 0.3, 0.9])
    d._score = lambda imgs, batch=8: [{"fake_prob": next(vals)} for _ in imgs]
    assert d._probe_health()["status"] == "ok"
