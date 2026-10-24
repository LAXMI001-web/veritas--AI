# TruthLens AI — Phase 1: Repository Audit

Audited: `btech-project-main` (FA3-CLIP image site) and `ScamGuard-Ai-main` (CNN+Conformer audio site).

## 1. Existing architecture

### FA3-CLIP image detector (`btech-project-main`)
- FastAPI + Jinja2 + a single-page vanilla-JS UI (`static/js/app.js`, 807 lines). Endpoints: `/api/predict-image`, `/api/predict-batch`, `/api/predict-url`, `/api/predict-frame`, `/api/model-info`, `/api/health`, `/metrics`.
- Model: `model/fa3_model.py` — CLIP ViT-B/32 + FFT high-pass branch + learnable spatial/frequency fusion (`fusion_logit`) + instance-conditioned live/fake prompts. Output: **2 logits, `id2label = {0: fake, 1: real}`**, scaled by `logit_scale.exp()` (≤100). Softmax over them is **not calibrated**.
- Preprocessing: `CLIPProcessor` (resize shortest edge 224 bicubic → center crop 224 → CLIP mean/std). Center crop means faces off-centre can be cropped out of the frame.
- Checkpoint format: `fa3_model.pt` (a `state_dict`) + `fa3_config.json` + tokenizer/processor JSON. Prompt IDs (`"a fake face"`, `"a live real face"`) are rebuilt from the tokenizer.
- `model/inference.py` has a **DEMO MODE**: when the checkpoint is missing it returns a hash-seeded pseudo-random "prediction". That is fabricated output and is **not carried over**.
- Datasets names are placeholders ("Dataset A/B/C") — no evaluation numbers exist in the repo.

### CNN + Conformer audio detector (`ScamGuard-Ai-main`)
- FastAPI + MongoDB (motor) + JWT auth + React frontend. **The React `src/` and `public/` folders are missing from the archive** (only `package.json`, lock file, tailwind config), so no audio UI could be reused; only the backend logic is reusable.
- `cnn_conformer.h5` is a **full Keras 3.12.1 model** (Functional; Conv2D32→Pool→Conv2D64→Pool→Reshape→MHA(4×64)+LN→FFN+LN→Conv1D64→GAP→Dense(1,sigmoid)); ~100k parameters. Input `(128,128,1)`; binary cross-entropy, so output is a sigmoid score. The repo treats **high = fake/scam** (`is_scam = score >= 0.5`). The training label order is not recorded in the repo — see risks.
- Preprocessing (`utils/audio_processor.py`): 16 kHz mono → Butterworth bandpass 80–8000 Hz → peak normalise → mel (n_fft 2048, hop 256, 128 mels, 80–8000 Hz, power 2) → `power_to_db(ref=max)` → pad/trim to **128 frames (~2.05 s)** → `(dB+80)/80` clipped to [0,1]. Only the **first** 128 frames are ever scored, so a long recording is judged on its first ~2 s.
- Problems found in `predictor.py` / `audio_processor.py`:
  1. Final score = `0.8 × model + 0.2 × handcrafted heuristic` — the displayed score is not purely the model's.
  2. `load_audio` **substitutes a synthetic sine wave when decoding fails** and then classifies it. Fabricated input.
  3. Without TensorFlow it silently falls back to the heuristic alone.
  4. `requirements.txt` pins `tensorflow==2.16.1` (Keras 3.0–3.3) but the `.h5` was written by Keras **3.12.1**; the `load_model` failure then falls to a rebuilt graph + `load_weights(by_name, skip_mismatch=True)` which can **silently skip layers**. Verified below.
  5. Uploads are saved to a public `StaticFiles` mount (`/uploads`).
- Neither segment-level scoring nor attention maps exist; both would be new code.

## 2. Preserve vs. drop
| Preserve | Drop / fix |
|---|---|
| FA3 architecture file (copied byte-for-byte to `backend/app/vendor/fa3_model.py`) | Demo-mode pseudo-predictions |
| FA3 CLIP preprocessing (`CLIPProcessor`) | 0.8/0.2 heuristic blending |
| Audio preprocessing constants + 128-frame input | Dummy-audio fallback |
| Original `cnn_conformer.h5` weights, unchanged | Public `/uploads` mount |
| Face-crop (Haar) idea for FA3 | Mongo/JWT dependency (replaced by local SQLite) |

## 3. Unified architecture
One FastAPI service (`backend/app`) serving a static SPA (`frontend/`), SQLite for results, no persistent media by default.
`models/` adapters (FA3, Audio) → `services/` (video sampling & aggregation, audio windows, integrity, sync, fusion/decision policy, explanation, comparison, robustness, report, jobs, history, evaluation). Models are loaded lazily and report `available/unavailable` honestly.

## 4. Blockers and risks (verified in this environment)
1. **`fa3_model.pt` (~580 MB) is not in the zip.** The Drive link you sent is blocked by the sandbox network policy (HTTP 403 on CONNECT), so I cannot fetch it. FA3 image/video inference therefore cannot be run or tested here. The code path is written against your real checkpoint format; until the file is at `checkpoints/fa3_artifact_faces/fa3_model.pt` the platform reports FA3 as *unavailable* — never demo scores.
2. HuggingFace is blocked too, so `openai/clip-vit-base-patch32` cannot be downloaded. The adapter builds the CLIP architecture from the default ViT-B/32 config offline and loads your state_dict (which contains the CLIP weights) — to be confirmed once you supply the file.
3. Audio label polarity (is sigmoid=fake?) and the model's training data are undocumented. Default follows ScamGuard; configurable.
4. No labeled evaluation data is in either repo → the Evaluation page shows *unavailable* until you provide a labeled test set (`docs/EVALUATION.md`).
5. Hardware: FA3 on CPU ≈ seconds/frame batch; Keras audio is tiny. Free hosting tiers cannot hold the 580 MB checkpoint + torch.

## 5. Implementable now vs. needs more
- **Now:** audio windows scoring, video frame sampling/aggregation (pipeline), audio extraction, fusion matrix A–I, integrity inspection, comparison, robustness, reports, batch, history, retention, tests (mocked FA3, real audio).
- **Needs your input:** FA3 weights; labeled data for metrics/calibration/aggregation selection.
- **Needs research/models:** lip-sync (optional module, not implemented), Grad-CAM validation for FA3, C2PA verification, LLM assistant.
