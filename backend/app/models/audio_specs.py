"""Audio model specifications: model file + the exact feature extraction it expects + how to read its output.

The FakeAVCeleb models' .h5 files record the INPUT SHAPE (128 mel bins x 94 frames, ~3 s at 16 kHz with hop 512) but NOT
the preprocessing used to train them (normalisation, n_fft, ...), nor which class index means "fake". Those cannot be
guessed reliably, so they are determined from labelled data by scripts/select_audio_model.py, which writes
models/audio_selection.json (model, preprocessing, polarity, calibration, threshold, measured test metrics).
Without such a file the default model runs UNVALIDATED and its verdicts are shown as inconclusive (raw score still shown)."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..config import ROOT, SETTINGS

NORMS = {
    "db80": "power_to_db(ref=max), (dB+80)/80 clipped to [0,1]   (same style as the original ScamGuard model)",
    "db": "power_to_db(ref=max), raw dB (negative values)",
    "minmax": "power_to_db(ref=max), min-max per clip to [0,1]",
    "zscore": "power_to_db(ref=max), z-score per clip",
    "db_ref1": "power_to_db(ref=1.0), raw dB",
    "log": "natural log of the mel power",
}


@dataclass
class AudioSpec:
    key: str                      # e.g. "cnn_conformer_fakeavceleb"
    model_file: str               # path relative to the repo root (or absolute)
    sr: int = 16000
    n_fft: int = 2048
    hop: int = 512
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float | None = None
    frames: int = 94
    bandpass: bool = False
    norm: str = "db80"
    high_means: str = "fake"      # which class a HIGH sigmoid output means: "fake" | "real"
    calibration: dict | None = None   # {"a":..,"b":..}: p_fake = sigmoid(a*logit(p_oriented)+b), fitted on validation data
    threshold: float = 0.5        # decision threshold on the (calibrated) fake score, chosen on validation data
    validated: bool = False
    validation: dict | None = None

    @property
    def window_samples(self) -> int:
        return self.frames * self.hop

    @property
    def window_seconds(self) -> float:
        return self.window_samples / self.sr

    @property
    def input_shape(self) -> tuple:
        return (self.n_mels, self.frames, 1)


LEGACY = AudioSpec(key="cnn_conformer_scamguard", model_file="models/cnn_conformer.h5", n_fft=2048, hop=256, fmin=80.0, fmax=8000.0,
                   frames=128, bandpass=True, norm="db80", high_means=SETTINGS.audio_high_means)
CANDIDATES = ("cnn_conformer", "cnn_lstm", "cnn_transformer", "cnn_bilstm_transformer")


def candidate_specs() -> dict[str, AudioSpec]:
    """All FakeAVCeleb models shipped in models/audio/ (default preprocessing, unvalidated) + the legacy ScamGuard model."""
    out = {}
    for c in CANDIDATES:
        f = ROOT / "models" / "audio" / f"{c}_fakeavceleb.h5"
        if f.exists():
            out[f"{c}_fakeavceleb"] = AudioSpec(key=f"{c}_fakeavceleb", model_file=str(f.relative_to(ROOT)))
    if (ROOT / LEGACY.model_file).exists():
        out[LEGACY.key] = LEGACY
    return out


def selection_path() -> Path:
    return Path(SETTINGS.audio_selection_file) if SETTINGS.audio_selection_file else ROOT / "models" / "audio_selection.json"


def load_spec() -> AudioSpec:
    """Selected spec = models/audio_selection.json if present (validated or not), else the default unvalidated model."""
    cands = candidate_specs()
    sp = selection_path()
    if sp.exists():
        try:
            d = json.loads(sp.read_text())
            base = cands.get(d["model"]) or AudioSpec(key=d["model"], model_file=d["model_file"])
            fields = {k: d[k] for k in asdict(base) if k in d and k not in ("key",)}
            fields["key"] = d["model"]
            spec = AudioSpec(**{**asdict(base), **fields})
            if not Path(spec.model_file).is_absolute():
                spec.model_file = str(ROOT / spec.model_file)
            return spec
        except Exception:
            pass       # corrupt selection file: fall through to the default (unvalidated)
    key = SETTINGS.audio_model or ("cnn_conformer_fakeavceleb" if "cnn_conformer_fakeavceleb" in cands else next(iter(cands), LEGACY.key))
    spec = cands.get(key, LEGACY)
    spec = AudioSpec(**asdict(spec))
    if not Path(spec.model_file).is_absolute():
        spec.model_file = str(ROOT / spec.model_file)
    return spec


# ------------------------------------------------------------------ features
def _bandpass(audio: np.ndarray, sr: int, lowcut=80, highcut=8000) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    nyq = sr / 2.0
    b, a = butter(4, [max(lowcut / nyq, 0.001), min(highcut / nyq, 0.999)], btype="band")
    return filtfilt(b, a, audio).astype(np.float32)


def window_features(seg: np.ndarray, spec: AudioSpec) -> np.ndarray:
    """One window of 16 kHz mono audio (length spec.window_samples, already padded) -> (n_mels, frames, 1) float32."""
    import librosa
    a = seg.astype(np.float32)
    if spec.bandpass:
        a = _bandpass(a, spec.sr)
        m = float(np.max(np.abs(a)))
        if m > 0:
            a = a / m
    mel = librosa.feature.melspectrogram(y=a, sr=spec.sr, n_fft=spec.n_fft, hop_length=spec.hop, n_mels=spec.n_mels,
                                         fmin=spec.fmin, fmax=spec.fmax if spec.fmax else spec.sr / 2, power=2.0)
    if spec.norm in ("db80", "db", "minmax", "zscore"):
        db = librosa.power_to_db(mel, ref=np.max)
    if spec.norm == "db80":
        x = np.clip((db + 80) / 80, 0, 1)
    elif spec.norm == "db":
        x = db
    elif spec.norm == "minmax":
        x = (db - db.min()) / (db.max() - db.min() + 1e-9)
    elif spec.norm == "zscore":
        x = (db - db.mean()) / (db.std() + 1e-9)
    elif spec.norm == "db_ref1":
        x = librosa.power_to_db(mel)
    elif spec.norm == "log":
        x = np.log(mel + 1e-9)
    else:
        raise ValueError(f"unknown normalisation {spec.norm}")
    if x.shape[1] < spec.frames:
        x = np.pad(x, ((0, 0), (0, spec.frames - x.shape[1])), mode="constant")
    return x[:, :spec.frames, None].astype(np.float32)


def oriented_fake_score(raw: float, spec: AudioSpec) -> float:
    """Raw sigmoid output -> P(fake)-oriented score (polarity from the spec; optional Platt calibration)."""
    p = float(raw) if spec.high_means == "fake" else 1.0 - float(raw)
    if spec.calibration:
        q = min(max(p, 1e-6), 1 - 1e-6)
        z = spec.calibration["a"] * math.log(q / (1 - q)) + spec.calibration["b"]
        p = 1 / (1 + math.exp(-z))
    return p
