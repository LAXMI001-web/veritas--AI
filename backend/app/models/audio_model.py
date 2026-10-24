"""CNN+Conformer audio adapter.

Preprocessing is a faithful copy of ScamGuard's `utils/audio_processor.preprocess_audio` (constants and order of
operations unchanged). Differences from ScamGuard, all deliberate:
  * score = raw model sigmoid only (no 0.2 heuristic blend),
  * decoding failures raise instead of substituting a synthetic sine wave,
  * the model is scored on *windows* (one 128-frame input each) instead of only the first 128 frames.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from dataclasses import asdict

from ..config import SETTINGS
from .audio_specs import LEGACY, AudioSpec, load_spec, oriented_fake_score, window_features

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 256
TARGET_FRAMES = 128
WINDOW_SAMPLES = (TARGET_FRAMES - 1) * HOP_LENGTH + HOP_LENGTH  # 32768 samples = 2.048 s -> 129 mel frames
WINDOW_SECONDS = WINDOW_SAMPLES / SAMPLE_RATE
SILENCE_RMS_DBFS = -55.0


def _bandpass(audio: np.ndarray, sr: int, lowcut=80, highcut=8000) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    nyq = sr / 2.0
    b, a = butter(4, [max(lowcut / nyq, 0.001), min(highcut / nyq, 0.999)], btype="band")
    return filtfilt(b, a, audio).astype(np.float32)


def preprocess_window(audio: np.ndarray) -> np.ndarray:
    """Identical to ScamGuard preprocess_audio(frame_selection='head') for 16 kHz input. -> (1,128,128,1)"""
    import librosa
    audio = _bandpass(audio, SAMPLE_RATE)
    m = np.max(np.abs(audio))
    if m > 0:
        audio = audio / m
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                         n_mels=N_MELS, fmin=80, fmax=8000, power=2.0)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    if log_mel.shape[1] < TARGET_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, TARGET_FRAMES - log_mel.shape[1])), mode="constant")
    else:
        log_mel = log_mel[:, :TARGET_FRAMES]
    log_mel = np.clip((log_mel + 80) / 80, 0, 1)
    return log_mel[None, ..., None].astype(np.float32)


def load_audio_16k(wav_path: Path) -> np.ndarray:
    """Same loader as the original (librosa, mono, 16 kHz). Raises on failure; never fabricates audio."""
    import librosa
    audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    if audio.size == 0:
        raise ValueError("decoded audio is empty")
    return audio.astype(np.float32)


class AudioDetector:
    """Spec-driven Keras audio detector (see audio_specs.py). Default spec = models/audio_selection.json or the default
    FakeAVCeleb model (unvalidated). Passing `model_path` builds a legacy-style spec for that file (old ScamGuard model)."""

    def __init__(self, model_path: str | None = None, spec: AudioSpec | None = None):
        if spec is None:
            if model_path:
                spec = AudioSpec(**{**asdict(LEGACY), "model_file": str(model_path)})
            else:
                spec = load_spec()
        self.spec = spec
        self.model_path = Path(spec.model_file)
        self.name = f"{spec.key} ({self.model_path.name})"
        self._model = None
        self._lock = threading.Lock()
        self.available = False
        self.reason: str | None = None
        self.param_count: int | None = None
        self.keras_version: str | None = None
        self.health: dict | None = None
        self._loaded = False

    # convenience for the rest of the app
    @property
    def window_samples(self) -> int:
        return self.spec.window_samples

    @property
    def window_seconds(self) -> float:
        return self.spec.window_seconds

    @property
    def decision_threshold(self) -> float:
        return self.spec.threshold

    @property
    def validated(self) -> bool:
        return bool(self.spec.validated)

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            if not self.model_path.exists():
                self.reason = f"Model file not found ({self.model_path.name})."
                return
            try:
                import tensorflow as tf
                self._model = tf.keras.models.load_model(str(self.model_path), compile=False)
                self.param_count = int(self._model.count_params())
                self.keras_version = tf.keras.__version__
                shp = tuple(self._model.input_shape[1:])
                if shp != self.spec.input_shape:
                    raise ValueError(f"model expects input {shp} but the preprocessing spec produces {self.spec.input_shape}")
                self.available = True
                self.health = self._probe_health()
            except Exception as exc:  # no silent partial-weight fallback (unlike ScamGuard)
                self._model = None
                self.reason = f"Model failed to load: {type(exc).__name__}: {str(exc)[:160]}"

    def _probe_health(self) -> dict:
        """Constant-output check. Feeds very different synthetic inputs (silence-like, all-ones, noise, gradients) and looks
        at the logits. A working real-vs-fake classifier should not give the SAME extreme answer to all of them.
        Result 'suspect' = the model looks collapsed/non-discriminative; it is a heuristic, not proof either way."""
        try:
            rng = np.random.default_rng(0)
            h, w = self.spec.n_mels, self.spec.frames
            probes = {"zeros": np.zeros((h, w, 1)), "ones": np.ones((h, w, 1)), "half": np.full((h, w, 1), .5),
                      "uniform noise": rng.random((h, w, 1)), "gaussian noise": np.clip(rng.normal(.5, .2, (h, w, 1)), 0, 1),
                      "gradient": np.tile(np.linspace(0, 1, h)[:, None, None], (1, w, 1))}
            x = np.stack(list(probes.values())).astype(np.float32)
            p = np.clip(np.asarray(self._model.predict(x, verbose=0)).reshape(-1).astype(np.float64), 1e-12, 1 - 1e-12)
            logit = np.log(p / (1 - p))
            same_side = bool(np.all(p < 0.5) or np.all(p > 0.5))
            saturated = bool(np.all((p < 1e-3) | (p > 1 - 1e-3)))
            suspect = same_side and saturated
            return {"status": "suspect" if suspect else "ok", "probe_outputs": {k: float(f"{v:.3g}") for k, v in zip(probes, p)},
                    "logit_range": [round(float(logit.min()), 1), round(float(logit.max()), 1)],
                    "message": ("The model returns the same extreme answer (all ≈%s) for every probe input — silence-like, noise, tones "
                                "and gradients — so it behaves like a constant classifier. Its output cannot separate real from fake audio. "
                                "Check the training run (class balance, learning rate, dead ReLUs) or retrain." % ("0" if p[0] < .5 else "1")) if suspect
                    else "Model responds differently to different probe inputs (this is not a proof that it is accurate)."}
        except Exception as exc:
            return {"status": "unknown", "message": f"Health probe failed: {type(exc).__name__}"}

    def info(self) -> dict:
        self.load()
        sp = self.spec
        return {"name": self.name, "model": sp.key, "available": self.available, "reason": self.reason, "health": self.health,
                "params": self.param_count, "keras_version": self.keras_version,
                "input": f"{sp.n_mels} mel bins x {sp.frames} frames (≈{sp.window_seconds:.2f} s @{sp.sr} Hz, hop {sp.hop}, n_fft {sp.n_fft}, norm {sp.norm})",
                "output": "sigmoid score", "high_means": sp.high_means, "threshold": sp.threshold,
                "calibrated": bool(sp.calibration), "validated": bool(sp.validated), "validation": sp.validation,
                "window_seconds": round(sp.window_seconds, 3),
                "validation_note": None if sp.validated else
                    "This model has not been validated on labelled data: its preprocessing, class polarity and threshold are "
                    "unconfirmed, so verdicts are shown as inconclusive. Run scripts/select_audio_model.py with labelled clips."}

    def score_windows(self, audio: np.ndarray, hop_s: float = 1.024, batch: int = 32) -> list[dict]:
        """Score fixed windows. Returns [{start,end,score,raw,silent,padded}]. score = P(fake)-oriented (polarity and
        calibration from the spec); raw = the model's unmodified sigmoid output."""
        self.load()
        if not self.available:
            raise RuntimeError(self.reason or "audio model unavailable")
        sp, W = self.spec, self.spec.window_samples
        n = len(audio)
        hop = max(int(hop_s * sp.sr), 1)
        starts = list(range(0, max(n - W, 0) + 1, hop)) or [0]
        if n > W and starts[-1] + W < n:
            starts.append(n - W)  # tail-aligned final window so the end is covered
        out, feats, idx = [], [], []
        for s in starts:
            seg = audio[s:s + W]
            padded = len(seg) < W
            if padded:
                seg = np.pad(seg, (0, W - len(seg)))
            rms = float(np.sqrt(np.mean(np.square(audio[s:s + W]))) + 1e-12)
            dbfs = 20 * np.log10(rms)
            rec = {"start": s / sp.sr, "end": min(s + W, n) / sp.sr,
                   "score": None, "raw": None, "silent": bool(dbfs < SILENCE_RMS_DBFS), "padded": padded,
                   "rms_dbfs": round(float(dbfs), 1)}
            out.append(rec)
            if not rec["silent"]:
                feats.append(window_features(seg, sp))
                idx.append(len(out) - 1)
        for i in range(0, len(feats), batch):
            raw = np.asarray(self._model.predict(np.stack(feats[i:i + batch]), verbose=0)).reshape(-1)
            for j, r in zip(idx[i:i + batch], raw):
                out[j]["raw"] = round(float(r), 5)
                out[j]["score"] = round(oriented_fake_score(float(r), sp), 5)
        return out
