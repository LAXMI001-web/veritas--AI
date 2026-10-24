# Decision policy (backend/app/services/fusion.py, config.Policy)
Score orientation: high = more suspicious. Modality verdict: `suspected` if score ≥ threshold+margin, `likely_authentic` if ≤ threshold−margin, else `inconclusive`; also `inconclusive` if evidence is insufficient (< min_frames scored frames; audio < min_audio_seconds or no non-silent window). Defaults: threshold 0.5, margin 0.10, mean aggregation — placeholders, **not calibrated**.
Scenarios: G (failure/unavailable) > F (no audio) > E (inconclusive) > A/B/C/D. H is added as a conflict flag (large video–audio gap, or high within-modality dispersion). I is a secondary flag when ≥2 consecutive samples ≥ interval_min_score.
Overall labels: Likely authentic · Suspected manipulation · Mixed-authenticity warning · Inconclusive · Analysis incomplete.


## Video: how false-genuine verdicts are prevented (v2)
- **Faces only.** FA3-CLIP is face-trained. Up to `n_frames × video_candidate_factor` candidate frames are decoded, faces detected (vendored OpenCV Haar cascades, frontal + profile, up to `max_faces_per_frame`), and `n_frames` face frames spread evenly over the timeline are scored. A frame's score is its highest face score. Frames with no face are **not scored** (whole-frame fallback is opt-in: `video_full_frame_fallback`). If the detector cannot load, the video analysis is *unavailable* rather than silently scoring whole frames.
- **Aggregation.** Default `topk`: mean of the highest `max(3, 25 %)` frames, so a manipulated minority is not averaged away and a single frame cannot decide alone. Mean/median/fraction remain selectable and are compared by `/api/evaluate` on labelled data.
- **"Likely authentic" must be supported.** It is withheld (→ inconclusive) if any frame scores ≥ `authentic_max_frame` (0.85), if scored face frames span < `min_temporal_coverage` (40 %) of the video, if a face was found in < 30 % of candidate frames, if < `min_frames` frames scored, or if the FA3 health probe shows a constant-output model.
- **Class order** is read from the checkpoint's `fa3_config.json` (`label2id`) and refused if it is not exactly fake/real on outputs 0/1.
- All thresholds are conventional placeholders until measured with `/api/evaluate` on your labelled data.


## Video: why "genuine" is hard to earn (false-genuine fix)
* FA3-CLIP is face-trained: only frames with a detected face are scored (frontal+profile Haar cascades, vendored). No face / no detector ⇒ **inconclusive / unavailable**, never a whole-frame "real". Whole-frame scoring is an explicit opt-in (`video_full_frame_fallback`).
* 3× candidate frames are decoded; up to `n_frames` face frames are kept evenly across the timeline. Every detected face (≤ `max_faces_per_frame`) is scored; a frame is as suspicious as its worst face.
* Default aggregation is **top-k** (k = max(3, 25 % of frames)): a manipulated minority is not averaged away, and no single frame can decide alone.
* "Likely authentic" is withheld (→ inconclusive) if any frame ≥ `authentic_max_frame` (0.85), if scored face frames span < `min_temporal_coverage` (40 %) of the video, if faces were found in < 30 % of candidates, or if the FA3 health probe shows a constant-output model.
* Class order is read from the checkpoint's `fa3_config.json` (`label2id`), not assumed; unexpected mappings are refused.
All thresholds are conventional placeholders — **not calibrated**; use the evaluation page on labelled data to choose them.
