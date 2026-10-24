# Evaluation & research protocol
No metrics ship with this project. Provide labeled data:
```
eval_data/<image|audio|video>/<val|test>/<real|fake>/files...
```
Keep train/val/test disjoint (also by identity/source video). `val` is for choosing thresholds, aggregation and temperature scaling; `test` is touched once for reporting. Run from **Model Evaluation** (or `POST /api/evaluation/run`). Positive class = fake.

Reported: accuracy, precision, recall, F1, FPR, FNR, confusion matrix, ROC/AUC, latency, per-modality; video frame-aggregation comparison (mean/median/fraction/top-k); ablation on videos that have audio: video-only, audio-only, combined mean, combined max, combined+abstention (coverage reported). The "temporal consistency features" ablation row is marked unavailable: no validated feature exists.
Not yet implemented: stratification by compression/resolution/manipulation type (needs metadata-labelled data), calibration reports (`fit_temperature` exists, unused in policy).

## Research contribution — honest status
| Candidate | Status |
|---|---|
| Engineering integration of two existing models | Done (not novel) |
| Explainable cross-modal conflict reporting (scenarios A–I) | Application feature; unvalidated |
| Uncertainty-aware abstention | Implemented; benefit must be shown by the ablation on your data |
| Benign-transformation robustness | Implemented; no results yet |
| Time-aligned evidence intervals | Implemented as sampled evidence; localization accuracy unmeasured |
No literature review was performed (web access was limited); do not claim novelty before one. No improvement claim is supported until the ablation shows it on a held-out test split.
