#!/usr/bin/env python3
"""Choose the audio model, preprocessing, polarity, calibration and threshold from LABELLED clips.

Usage (from the backend folder):
  python scripts/select_audio_model.py --data D:\\fakeavceleb_eval
where D:\\fakeavceleb_eval contains either
  real/ fake/                      (flat; split 50/50 into validation/test by file or by --group-regex), or
  val/real val/fake test/real test/fake   (your own disjoint splits; preferred)
Clips may be audio or video (audio is extracted). Writes models/audio_selection.json (+ _report.json). The API reads that file
at start-up: restart the backend afterwards. Never put TRAINING clips in the evaluation folders."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import ROOT  # noqa: E402
from app.services.audio_selection import run_selection  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default=str(ROOT / "models" / "audio_selection.json"))
ap.add_argument("--group-regex", default=None, help="regex whose first group identifies speaker/source (keeps a group in one split)")
ap.add_argument("--max-windows", type=int, default=8, help="3 s windows scored per clip")
ap.add_argument("--quick", action="store_true", help="only n_fft=2048")
ap.add_argument("--min-auc", type=float, default=0.75)
ap.add_argument("--models", nargs="*", default=None, help="subset of model keys")
a = ap.parse_args()
r = run_selection(Path(a.data), Path(a.out), a.group_regex, a.max_windows, a.quick, a.min_auc, models=a.models)
s = r["selected"]
print("\n=== SELECTED ===")
print(f"model {s['model']}  norm={s['norm']}  n_fft={s['n_fft']}  high output means {s['high_means']}  threshold={s['threshold']}")
print("validated:", s["validated"], s["validation"]["reasons_not_validated"])
print("TEST metrics (held out, evaluated once):")
for k, v in s["validation"]["test_metrics"].items():
    print(f"  {k}: {v}")
print("written:", a.out)
