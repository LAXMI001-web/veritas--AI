"""Speech-to-text for the audio track (or a live chunk).

Engine: Whisper (English) run through sherpa-onnx — no cloud service, audio never leaves the machine.
The model files are not shipped; install them with `python scripts/fetch_asr_model.py`.
If the model or library is missing the transcriber reports itself UNAVAILABLE; it never invents text.

Segmentation is done here with a simple energy VAD (so timestamps are VAD-segment boundaries, not word-exact):
each speech segment is decoded separately and carries the start/end of the audio it was decoded from.
Transcription accuracy is not validated on your data. Whisper can hallucinate text on noise/music; segments with
very low confidence are flagged rather than hidden."""
from __future__ import annotations

import math
import re
import threading
from pathlib import Path

import numpy as np

from ..config import ROOT, SETTINGS
from ..models.audio_model import SAMPLE_RATE, load_audio_16k
from .media import MediaError, cleanup, extract_audio_wav, temp_dir

MAX_SEG_S = 24.0          # Whisper decodes <= 30 s windows
MIN_SEG_S = 0.35
MERGE_GAP_S = 0.45
PAD_S = 0.15
MIN_SEG_DBFS = -50.0
_NON_SPEECH = re.compile(r"[\[(\*][^\])\*]*[\])\*]\.?")


def model_dir() -> Path:
    if SETTINGS.asr_model_dir:
        return Path(SETTINGS.asr_model_dir)
    return ROOT / "models" / "asr" / f"sherpa-onnx-whisper-{SETTINGS.asr_model_name}"


def _files(d: Path) -> tuple[Path, Path, Path] | None:
    n = SETTINGS.asr_model_name
    for suffix in (".int8.onnx", ".onnx"):
        enc, dec = d / f"{n}-encoder{suffix}", d / f"{n}-decoder{suffix}"
        if enc.exists() and dec.exists() and (d / f"{n}-tokens.txt").exists():
            return enc, dec, d / f"{n}-tokens.txt"
    return None


def speech_segments(audio: np.ndarray) -> list[tuple[float, float]]:
    """Energy-based speech/non-speech split -> [(start_s, end_s)]."""
    import librosa
    if audio.size < int(MIN_SEG_S * SAMPLE_RATE):
        return []
    iv = librosa.effects.split(audio, top_db=35, frame_length=1024, hop_length=256)
    segs: list[list[float]] = []
    for a, b in iv:
        s, e = a / SAMPLE_RATE, b / SAMPLE_RATE
        if segs and s - segs[-1][1] < MERGE_GAP_S:
            segs[-1][1] = e
        else:
            segs.append([s, e])
    out: list[tuple[float, float]] = []
    total = audio.size / SAMPLE_RATE
    for s, e in segs:
        if e - s < MIN_SEG_S:
            continue
        s, e = max(0.0, s - PAD_S), min(total, e + PAD_S)
        while e - s > MAX_SEG_S:           # long continuous speech: cut into <= MAX_SEG_S pieces
            out.append((s, s + MAX_SEG_S))
            s += MAX_SEG_S
        if e - s >= MIN_SEG_S:
            out.append((s, e))
    return out


class Transcriber:
    def __init__(self):
        self._rec = None
        self._lock = threading.Lock()
        self._loaded = False
        self.available = False
        self.reason: str | None = None

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            try:
                import sherpa_onnx as so
            except Exception:
                self.reason = "Speech-to-text library missing (pip install sherpa-onnx)."
                return
            files = _files(model_dir())
            if files is None:
                self.reason = (f"Whisper '{SETTINGS.asr_model_name}' model files not found in {model_dir().name}. "
                               "Run: python scripts/fetch_asr_model.py")
                return
            try:
                enc, dec, tok = files
                self._rec = so.OfflineRecognizer.from_whisper(encoder=str(enc), decoder=str(dec), tokens=str(tok),
                                                              language="en", task="transcribe",
                                                              num_threads=SETTINGS.asr_threads)
                self.available = True
            except Exception as exc:
                self._rec = None
                self.reason = f"Speech model failed to load: {type(exc).__name__}: {str(exc)[:160]}"

    def info(self) -> dict:
        self.load()
        return {"name": f"Whisper {SETTINGS.asr_model_name} (sherpa-onnx)", "available": self.available,
                "reason": self.reason, "language": "en", "local_only": True}

    def transcribe_audio(self, audio: np.ndarray, offset_s: float = 0.0) -> dict:
        """audio: float32 mono 16 kHz. -> transcript dict (status ok | unavailable | failed)."""
        self.load()
        if not self.available:
            return {"status": "unavailable", "reason": self.reason, "modality": "transcript"}
        try:
            segs_out = []
            for s, e in speech_segments(audio):
                chunk = audio[int(s * SAMPLE_RATE): int(e * SAMPLE_RATE)]
                dbfs = 20 * math.log10(float(np.sqrt(np.mean(np.square(chunk)))) + 1e-12)
                if dbfs < MIN_SEG_DBFS:
                    continue
                with self._lock:
                    st = self._rec.create_stream()
                    st.accept_waveform(SAMPLE_RATE, chunk)
                    self._rec.decode_stream(st)
                    r = st.result
                text = (r.text or "").strip()
                if not text or _NON_SPEECH.fullmatch(text):    # Whisper tags like "(static)", "[BLANK_AUDIO]", "[Music]"
                    continue
                lp = list(getattr(r, "ys_log_probs", []) or [])
                conf = float(math.exp(sum(lp) / len(lp))) if lp else None   # geometric-mean token probability
                segs_out.append({"start": round(s + offset_s, 2), "end": round(e + offset_s, 2), "text": text,
                                 "confidence": None if conf is None else round(conf, 3)})
        except Exception as exc:
            return {"status": "failed", "reason": f"Transcription failed: {type(exc).__name__}", "modality": "transcript"}
        confs = [x["confidence"] for x in segs_out if x["confidence"] is not None]
        words = sum(len(x["text"].split()) for x in segs_out)
        return {"status": "ok", "modality": "transcript", "engine": f"Whisper {SETTINGS.asr_model_name} (sherpa-onnx)",
                "language": "en", "segments": segs_out, "text": " ".join(x["text"] for x in segs_out),
                "word_count": words, "mean_confidence": round(float(np.mean(confs)), 3) if confs else None,
                "note": "Timestamps are speech-segment boundaries from a simple energy VAD, not word-exact. "
                        "Machine transcription can contain errors and may hallucinate text on noise or music."}


_T: Transcriber | None = None


def get_transcriber() -> Transcriber:
    global _T
    if _T is None:
        _T = Transcriber()
    return _T


def set_transcriber(t) -> None:      # used by tests
    global _T
    _T = t


def transcribe_file(src: Path) -> dict:
    """Extract the audio track of a media file and transcribe it. Never raises."""
    tr = get_transcriber()
    tr.load()
    if not tr.available:
        return {"status": "unavailable", "reason": tr.reason, "modality": "transcript"}
    tmp = temp_dir()
    try:
        wav = tmp / "t.wav"
        try:
            extract_audio_wav(src, wav)
            audio = load_audio_16k(wav)
        except MediaError as exc:
            return {"status": "failed", "reason": str(exc), "modality": "transcript"}
        except Exception:
            return {"status": "failed", "reason": "Audio could not be decoded for transcription.", "modality": "transcript"}
        return tr.transcribe_audio(audio)
    finally:
        cleanup(tmp)
