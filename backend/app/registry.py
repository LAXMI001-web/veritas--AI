"""Model registry: single place to obtain detectors (tests inject fakes here)."""
from __future__ import annotations

from .models.audio_model import AudioDetector
from .models.fa3 import FA3Detector

_fa3 = None
_audio = None


def get_fa3():
    global _fa3
    if _fa3 is None:
        _fa3 = FA3Detector()
    return _fa3


def get_audio():
    """Audio backend: 'hf' (downloaded Hugging Face model), 'keras' (FakeAVCeleb .h5 models), or 'auto' = hf if present."""
    global _audio
    if _audio is None:
        from pathlib import Path
        from .config import SETTINGS
        want = (SETTINGS.audio_backend or "auto").lower()
        if want == "hf" or (want == "auto" and (Path(SETTINGS.audio_hf_dir) / "config.json").exists()):
            from .models.hf_audio import HFAudioDetector
            _audio = HFAudioDetector()
        else:
            _audio = AudioDetector()
    return _audio


def set_detectors(fa3=None, audio=None):
    """Used by tests to inject mocked detectors."""
    global _fa3, _audio
    if fa3 is not None:
        _fa3 = fa3
    if audio is not None:
        _audio = audio
