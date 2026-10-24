# TruthLens AI — multimodal deepfake intelligence & media forensics
One FastAPI service + one web UI combining your **FA3-CLIP** (image / video frames) and **CNN+Conformer** (audio) models. Audit: `docs/AUDIT.md`. Policy: `docs/DECISION_POLICY.md`. Evaluation/research: `docs/EVALUATION.md`.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate     # Python 3.11–3.13
pip install -r backend/requirements.txt                 # + ffmpeg/ffprobe on PATH
# FA3 weights (NOT in the repo, ~580 MB): put your file at
cp /path/to/fa3_model.pt checkpoints/fa3_artifact_faces/fa3_model.pt
python backend/scripts/fetch_asr_model.py                 # one-time ~200 MB Whisper model for transcripts (from GitHub releases)
cd backend && python -m uvicorn app.main:app --port 8000   # open http://localhost:8000
```
The audio model `models/cnn_conformer.h5` is already included. Until `fa3_model.pt` exists, the header shows FA3 as *unavailable* and image/video results say so — there is no demo mode. CLIP ViT-B/32 is rebuilt offline from its default config; your state_dict supplies all weights.

## Transcript + voice-vs-wording check (new)
For every audio file / video with sound, the audio track is transcribed (Whisper via sherpa-onnx, **fully offline**, English) and shown under the audio evidence, split into speech segments with timestamps. Each segment shows the audio model's score for that stretch and a wording class (*ordinary* / *mild indicators* / *suspicious*). The cross-check then states plainly:
* **voice flagged, words ordinary** – audio model says suspicious but the text has no scam/pressure wording;
* **voice natural, words suspicious** – the reverse;
* both flagged / both ordinary / not comparable (inconclusive audio, too few words, transcript unavailable).
It is **advisory**: it does not change the A–I authenticity matrix, but adds a conflict line. The wording check is a transparent phrase-matching lexicon (`services/textcheck.py`, thresholds in the policy: `text_suspicious_cut`, `text_min_words`, `asr_min_confidence`), **not a trained model**, and it says nothing about whether a voice is synthetic. Turn it off with the policy field `transcribe=false`. If the Whisper model is not installed the transcript shows "unavailable" and everything else works. Whisper can hallucinate on noise/music; non-speech tags like "(static)" are dropped; timestamps are VAD-segment boundaries, not word-exact; sherpa-onnx's Whisper does not expose token probabilities here, so "transcription confidence" is usually n/a and `asr_min_confidence` is then inactive.

## Live detection (new) — page "Live Detection"
Microphone (and optional webcam) in the browser → chunks of 4–8 s (MediaRecorder, each a self-contained file) + a webcam JPEG every 3 s → `/api/live/*` → same CNN+Conformer windows, Whisper transcript and wording check, FA3-CLIP on webcam frames (face crop per policy) → rolling 30 s assessment through the same fusion policy, a live transcript with per-segment voice/wording badges, and score charts. Chunks/frames are processed in temp files deleted immediately; only scores and text are kept **in server memory** for the session (not in history, not on disk); "Download session log" saves them locally. Needs `http://localhost` or HTTPS for mic/camera access. Webcam scoring needs `fa3_model.pt`. Limits: a chunk can cut a word in half; short windows are noisier than file analysis; per-chunk processing must be faster than the chunk length (≈1–3 s for a 5 s chunk on 2 CPU cores with `base.en`; use `ASR_MODEL=tiny.en` on slower machines).
API: `POST /api/live/start` → `{session_id}` · `POST /api/live/{id}/audio` (file) · `POST /api/live/{id}/frame` (JPEG) · `DELETE /api/live/{id}`.

## Tests
`cd backend && python -m pytest tests -q` (needs ffmpeg).
* Decision/aggregation/metric tests: **mocked** scores.
* API/pipeline tests: **mocked FA3**, mocked or real audio (`test_real_audio.py` uses the real `.h5`).
* `test_fa3_loader.py`: **random weights**, checks loader mechanics only.
* `test_transcript_text.py`, `test_transcript_live_api.py`: **mocked** ASR/audio/FA3 outputs (policy logic, API, live flow).
* `test_real_asr.py`: **real** Whisper + real `.h5` on espeak-generated speech (skipped if the ASR model is missing). Robotic TTS tests plumbing, not detection accuracy.
* Not tested: real FA3 inference (checkpoint missing).
UI demo without the checkpoint (mock FA3, testing only): `python -m tests.ui_mock_server 8766`.

## Architecture
```
browser SPA ──/api──> FastAPI ─ jobs (thread pool) ─ pipeline ─┬ video: sample frames → (face crop) → FA3 → aggregate/intervals
                                                              ├ audio: ffmpeg extract → 16k → 2.05 s windows → CNN+Conformer
                                                              ├ integrity (ffprobe, repeated frames, cuts, provenance marker)
                                                              ├ optional sync heuristic
                                                              └ fusion policy (A–I) → explanation → SQLite (results only)
```
Uploads go to a temp dir, are deleted when the job ends; only results are stored (retention `RETENTION_DAYS`).

## API (all under /api)
`GET health|models|config|stats` · `POST live/start|live/{id}/audio|live/{id}/frame` · `DELETE live/{id}` · · `PUT policy` · `POST analyze` (file, expect, policy) → job · `GET jobs/{id}` · `POST jobs/{id}/cancel|retry` · `POST batch` · `GET batches/{id}[/report]` · `GET analyses[?q&kind&overall]` · `GET|DELETE analyses/{id}` · `GET analyses/{id}/report.pdf|report.json` · `POST analyses/{id}/ask` · `POST inspect|compare|robustness` → task · `GET tasks/{id}` · `GET evaluation` · `POST evaluation/run`.

## Differences from the original apps (deliberate)
Removed: FA3 demo-mode pseudo-scores; audio 0.8/0.2 heuristic blend; synthetic-sine fallback on decode failure; public `/uploads`; silent partial-weight loading. Audio is now scored over the whole recording, not just its first 2 s. Preprocessing constants and the FA3 architecture file are unchanged.

## Known limitations / not done
* FA3 inference unverified here (weights unavailable). Face-crop for video frames (`video_face_crop=auto`) deviates from whole-image scoring; set `off` to match the original app exactly. Which is better must be decided on validation data.
* Audio score polarity (high = fake) is taken from ScamGuard; confirm and set `AUDIO_HIGH_MEANS`. The model saturates to 0/1 on out-of-domain input (e.g. pure tones).
* No Grad-CAM/attribution for FA3 (not validated); no audio attribution. Lip-sync model not implemented; only an experimental speech-activity/mouth-motion heuristic (off by default).
* Transcript/live: English only; wording lexicon is unvalidated; webcam live scoring untested with real FA3 weights; live sessions are in-memory and single-process.
* C2PA verification not implemented (marker presence only). LLM assistant not implemented (rule-based only).
* No playback (media is not retained). Single-user, in-process queue; use a real queue/DB for multi-user deployment.
* Hosting: the FA3 checkpoint (~580 MB) + torch exceed free tiers; use a VM with ≥4 GB RAM (GPU optional).
* Don't commit `*.pt`, `*.h5` weights or `data/` (see .gitignore).

## Audio model selection & video fixes (latest update)
* **Audio**: four FakeAVCeleb models live in `models/audio/`. Their training preprocessing/class polarity was not recorded, so they are treated as *unvalidated* and the app reports audio as **Inconclusive** until you measure them:
  `cd backend && python scripts/select_audio_model.py --data <dir with real/ and fake/ (or val|test/real|fake)> --group-regex "(id\d+)"` — it compares every model x preprocessing x polarity on a validation split, fits calibration + threshold there, evaluates the held-out test split once, and writes `models/audio_selection.json`. Only then does the app interpret audio scores.
* **Video**: only frames with a detected face are scored (vendored Haar cascades; `opencv-python-headless<5` required), top-k aggregation, "likely authentic" withheld on any extreme frame / thin coverage / constant model. See `docs/DECISION_POLICY.md`.
* **Run (Windows, from the project folder)**:
  `py -3.11 -m venv venv && venv\Scripts\activate && pip install -r backend\requirements.txt && python backend\scripts\fetch_asr_model.py && cd backend && python -m uvicorn app.main:app --port 8000` then open http://localhost:8000 (the backend serves the frontend). Needs ffmpeg on PATH.

### Better audio model (Hugging Face) — one command
`python backend\scripts\fetch_audio_model.py` downloads `garystafford/wav2vec2-deepfake-voice-detector` (Apache-2.0, ~1.2 GB) to `models\audio_hf\default`; restart the backend and it is used automatically (`AUDIO_BACKEND=auto`). Delete that folder (or set `AUDIO_BACKEND=keras`) to return to the FakeAVCeleb models. Its published accuracy is on its authors' data (TTS services), not yours — the app shows this as a warning.
