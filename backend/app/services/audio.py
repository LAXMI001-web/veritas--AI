"""Audio analysis: decode, window-score with the CNN+Conformer, aggregate, waveform/spectrogram payloads."""
from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from .. import registry
from ..config import Policy
from ..models.audio_model import SAMPLE_RATE, WINDOW_SECONDS, load_audio_16k
from .aggregate import all_aggregates, dispersion, suspicious_intervals, aggregate
from .media import MediaError, extract_audio_wav, temp_dir, cleanup


def waveform_envelope(audio: np.ndarray, buckets: int = 800) -> dict:
    n = len(audio)
    buckets = min(buckets, max(n, 1))
    edges = np.linspace(0, n, buckets + 1).astype(int)
    mn, mx = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = audio[a:b] if b > a else audio[a:a + 1]
        mn.append(round(float(seg.min()), 4)); mx.append(round(float(seg.max()), 4))
    return {"min": mn, "max": mx, "duration": n / SAMPLE_RATE}


def spectrogram_png_b64(audio: np.ndarray, max_width: int = 900) -> str:
    import librosa
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=256, n_mels=128,
                                         fmin=80, fmax=8000, power=2.0)
    db = librosa.power_to_db(mel, ref=np.max)
    img = np.clip((db + 80) / 80, 0, 1)[::-1]
    if img.shape[1] > max_width:
        img = cv2.resize(img, (max_width, img.shape[0]), interpolation=cv2.INTER_AREA)
    col = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    ok, buf = cv2.imencode(".png", col)
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def analyze_audio_file(src: Path, pol: Policy, from_video: bool = False) -> dict:
    """src may be a standalone audio file or a video (its audio track is extracted)."""
    det = registry.get_audio()
    det.load()
    if not det.available:
        return {"status": "unavailable", "reason": det.reason, "modality": "audio"}
    tmp = temp_dir()
    try:
        wav = tmp / "a.wav"
        try:
            extract_audio_wav(src, wav)
            audio = load_audio_16k(wav)
        except MediaError as exc:
            return {"status": "failed", "reason": str(exc), "modality": "audio"}
        except Exception:
            return {"status": "failed", "reason": "Audio could not be decoded.", "modality": "audio"}
        duration = len(audio) / SAMPLE_RATE
        try:
            wins = det.score_windows(audio, hop_s=pol.audio_window_hop_s)
        except Exception as exc:
            return {"status": "failed", "reason": f"Audio inference failed: {type(exc).__name__}", "modality": "audio",
                    "duration_s": round(duration, 3)}
        scored = [w for w in wins if w["score"] is not None]
        sc = [w["score"] for w in scored]
        primary = aggregate(sc, pol.audio_aggregation, pol.frame_suspicious_cut, pol.topk)
        return {
            "status": "ok", "modality": "audio", "model": det.name, "duration_s": round(duration, 3),
            "windows": wins, "scored": len(scored), "silent_windows": sum(w["silent"] for w in wins),
            "window_seconds": round(getattr(det, "window_seconds", WINDOW_SECONDS), 3), "hop_seconds": pol.audio_window_hop_s,
            "aggregates": all_aggregates(sc, pol.frame_suspicious_cut, pol.topk),
            "aggregation_method": pol.audio_aggregation, "score": None if primary is None else round(primary, 4),
            "dispersion": dispersion(sc),
            "intervals": suspicious_intervals(wins, pol.interval_min_score, pol.interval_min_len),
            "waveform": waveform_envelope(audio), "spectrogram_png": spectrogram_png_b64(audio),
            "from_video": from_video, "model_health": getattr(det, "health", None),
            "model_validation": {"validated": getattr(det, "validated", True), "external": getattr(det, "external_validation", False), "note": getattr(det, "validation_note", None), "metrics": (getattr(det, "spec", None) and det.spec.validation),
                                 "threshold": getattr(det, "decision_threshold", None), "high_means": getattr(getattr(det, "spec", None), "high_means", None),
                                 "calibrated": bool(getattr(getattr(det, "spec", None), "calibration", None))},
            "note": "Each window is scored independently as one ~2.05 s model input; overlapping windows are not "
                    "independent evidence.",
        }
    finally:
        cleanup(tmp)
