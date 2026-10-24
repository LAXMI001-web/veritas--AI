import numpy as np
from app.services import aggregate as ag, evaluation as ev


def test_aggregates():
    s = [0.1, 0.2, 0.9, 0.95]
    assert ag.aggregate(s, "mean") == np.mean(s)
    assert ag.aggregate(s, "median") == np.median(s)
    assert ag.aggregate(s, "fraction", 0.5) == 0.5
    assert abs(ag.aggregate(s, "topk", topk=2) - 0.925) < 1e-9
    assert ag.aggregate([None], "mean") is None


def test_intervals():
    items = [{"start": i, "end": i, "score": s} for i, s in enumerate([0.1, 0.9, 0.95, 0.2, 0.85, 0.1])]
    iv = ag.suspicious_intervals(items, 0.8, 2)
    assert len(iv) == 1 and iv[0]["start"] == 1 and iv[0]["end"] == 2


def test_metrics_and_auc_known_values():
    y = [0, 0, 1, 1]; s = [0.1, 0.4, 0.35, 0.8]
    m = ev.metrics(y, s, 0.5)
    assert m["confusion"] == {"tp": 1, "fp": 0, "tn": 2, "fn": 1}
    assert abs(m["roc_auc"] - 0.75) < 1e-9 and m["accuracy"] == 0.75


def test_auc_ties_and_perfect():
    assert ev.roc_auc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5
    assert ev.roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_abstain_reduces_coverage():
    m = ev.metrics([0, 1, 0, 1], [0.1, 0.9, 0.52, 0.48], 0.5, abstain_margin=0.1)
    assert m["coverage"] == 0.5


def test_evaluation_unavailable_without_data(tmp_path):
    from app.config import Policy
    r = ev.run(Policy(), root=str(tmp_path))
    assert r["status"] == "unavailable"


def test_temperature_fit_needs_data():
    assert ev.fit_temperature(np.array([1.0, -1.0]), np.array([1, 0])) is None
