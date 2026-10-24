"""Environment-driven configuration + runtime-editable decision policy."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    if isinstance(default, bool):
        return v.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v


@dataclass
class Settings:
    data_dir: str = _env("VERITAS_DATA_DIR", str(ROOT / "data"))
    fa3_checkpoint_dir: str = _env("FA3_CHECKPOINT_DIR", str(ROOT / "checkpoints" / "fa3_artifact_faces"))
    audio_model_path: str = _env("AUDIO_MODEL_PATH", str(ROOT / "models" / "cnn_conformer.h5"))
    # What a HIGH sigmoid output of the audio model means. ScamGuard treats high as fake/scam.
    audio_high_means: str = _env("AUDIO_HIGH_MEANS", "fake")
    # If the audio model gives the same extreme answer to every probe input (a collapsed/constant classifier) its verdicts
    # are forced to "inconclusive". Set AUDIO_ALLOW_DEGENERATE=true to show its raw verdicts anyway.
    audio_allow_degenerate: bool = _env("AUDIO_ALLOW_DEGENERATE", False)
    audio_model: str = _env("AUDIO_MODEL", "")                    # e.g. cnn_lstm_fakeavceleb (only if no selection file)
    audio_selection_file: str = _env("AUDIO_SELECTION_FILE", "")  # default models/audio_selection.json
    # Verdicts from a model that has not been validated on labelled data are shown as inconclusive unless this is set.
    audio_backend: str = _env("AUDIO_BACKEND", "auto")            # auto | keras | hf   (auto: use the Hugging Face model if downloaded)
    audio_hf_dir: str = _env("AUDIO_HF_DIR", str(ROOT / "models" / "audio_hf" / "default"))
    audio_hf_invert: bool = _env("AUDIO_HF_INVERT", False)        # swap which output means "fake" (use only after checking on known clips)
    audio_hf_fake_index: str = _env("AUDIO_HF_FAKE_INDEX", "")    # only needed if the model's labels are ambiguous
    audio_allow_unvalidated: bool = _env("AUDIO_ALLOW_UNVALIDATED", False)
    max_upload_mb: int = _env("MAX_UPLOAD_MB", 200)
    max_media_seconds: int = _env("MAX_MEDIA_SECONDS", 600)
    request_timeout_s: int = _env("SUBPROCESS_TIMEOUT_S", 120)
    rate_limit_per_min: int = _env("RATE_LIMIT_PER_MIN", 60)
    api_token: str = _env("VERITAS_API_TOKEN", "")  # optional; if set, /api requires it
    retention_days: int = _env("RETENTION_DAYS", 30)
    store_thumbnails: bool = _env("STORE_THUMBNAILS", True)
    eval_data_dir: str = _env("EVAL_DATA_DIR", str(ROOT / "eval_data"))
    workers: int = _env("WORKERS", 2)
    # --- speech-to-text (sherpa-onnx Whisper). Model files are NOT shipped: run scripts/fetch_asr_model.py
    asr_model_name: str = _env("ASR_MODEL", "base.en")            # tiny.en | base.en | small.en (English only)
    asr_model_dir: str = _env("ASR_MODEL_DIR", "")                # default: <repo>/models/asr/sherpa-onnx-whisper-<name>
    asr_threads: int = _env("ASR_THREADS", 2)
    live_rate_limit_per_min: int = _env("LIVE_RATE_LIMIT_PER_MIN", 300)
    live_max_chunk_mb: int = _env("LIVE_MAX_CHUNK_MB", 4)
    live_max_chunk_s: int = _env("LIVE_MAX_CHUNK_S", 20)


SETTINGS = Settings()


@dataclass
class Policy:
    """Transparent, configurable decision policy (see docs/DECISION_POLICY.md).
    NOTHING here has been calibrated on labeled data; defaults are conventional placeholders."""
    suspicious_threshold: float = 0.5      # score >= this (+margin) => suspected
    inconclusive_margin: float = 0.10      # |score - threshold| < margin => inconclusive
    frame_aggregation: str = "topk"        # mean | median | fraction | topk  (topk: mean of the highest-scoring frames, so a
                                           # manipulated minority of frames is not averaged away; needs >=3 such frames)
    audio_aggregation: str = "mean"        # mean | median | fraction | topk
    frame_suspicious_cut: float = 0.5      # per-frame/window cut for "suspicious" counts
    topk: int = 3
    topk_frac: float = 0.25                # top-k size = max(topk, ceil(topk_frac * scored frames)) -> one frame can't decide
    n_frames: int = 16                     # sampled frames per video
    max_frames: int = 64
    min_frames: int = 6                    # below => inconclusive (insufficient evidence)
    audio_window_hop_s: float = 1.024
    min_audio_seconds: float = 2.0
    min_scored_windows: int = 1
    video_face_crop: str = "auto"          # auto | off  (auto: score detected face crops; off: whole frames)
    video_candidate_factor: int = 3        # decode n_frames*factor candidate frames, keep those containing a face
    max_faces_per_frame: int = 3           # each face is scored; the frame's score is the highest face score
    video_full_frame_fallback: bool = False  # False: frames without a detectable face are NOT scored (the model is face-trained)
    authentic_max_frame: float = 0.85      # "likely authentic" is withheld if any single frame scores >= this
    min_temporal_coverage: float = 0.4     # "likely authentic" is withheld if scored face frames span < this share of the video
    interval_min_score: float = 0.8        # contiguous samples above this form a "suspicious interval"
    interval_min_len: int = 2
    sync_check: bool = False               # optional experimental audio/visual speech-activity check
    conflict_gap: float = 0.5              # |video-audio| score gap flagged as strong disagreement
    transcribe: bool = True                # speech-to-text + transcript-vs-audio cross-check (needs the ASR model)
    text_suspicious_cut: float = 0.5       # text-risk score >= this AND >=2 indicator categories => suspicious wording
    text_min_words: int = 8                # fewer transcribed words => text check inconclusive
    asr_min_confidence: float = 0.30       # mean ASR token confidence below this => text check inconclusive

    def validate(self):
        assert 0 < self.suspicious_threshold < 1
        assert 0 <= self.inconclusive_margin < 0.5
        assert self.frame_aggregation in ("mean", "median", "fraction", "topk")
        assert self.audio_aggregation in ("mean", "median", "fraction", "topk")
        assert 1 <= self.n_frames <= self.max_frames <= 256
        assert self.video_face_crop in ("auto", "off")
        assert 1 <= self.video_candidate_factor <= 8 and 1 <= self.max_faces_per_frame <= 8
        assert 0 <= self.topk_frac <= 1 and 0 < self.authentic_max_frame <= 1 and 0 <= self.min_temporal_coverage <= 1
        assert 0 < self.text_suspicious_cut <= 1 and self.text_min_words >= 1 and 0 <= self.asr_min_confidence < 1
        return self


class PolicyStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._policy = Policy()

    def get(self) -> Policy:
        with self._lock:
            return Policy(**asdict(self._policy))

    def update(self, patch: dict) -> Policy:
        with self._lock:
            d = asdict(self._policy)
            for k, v in patch.items():
                if k not in d:
                    raise ValueError(f"unknown policy field: {k}")
                d[k] = type(d[k])(v) if not isinstance(d[k], bool) else bool(v)
            new = Policy(**d).validate()
            self._policy = new
            return Policy(**asdict(new))


POLICY = PolicyStore()


def data_path(*parts: str) -> Path:
    p = Path(SETTINGS.data_dir).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
