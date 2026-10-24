#!/usr/bin/env python3
"""Download the Whisper speech-to-text model used for transcripts (one time, ~200 MB for base.en).
Source: k2-fsa/sherpa-onnx GitHub release 'asr-models'. After this, transcription runs fully offline.
Usage: python scripts/fetch_asr_model.py [tiny.en|base.en|small.en]"""
import sys
import tarfile
import urllib.request
from pathlib import Path

name = sys.argv[1] if len(sys.argv) > 1 else "base.en"
if name not in ("tiny.en", "base.en", "small.en"):
    sys.exit("model must be one of: tiny.en, base.en, small.en")
root = Path(__file__).resolve().parents[2] / "models" / "asr"
root.mkdir(parents=True, exist_ok=True)
url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-{name}.tar.bz2"
dest = root / f"{name}.tar.bz2"
print("downloading", url)
urllib.request.urlretrieve(url, dest)
with tarfile.open(dest, "r:bz2") as t:
    for m in t.getmembers():                       # refuse path traversal
        if m.name.startswith(("/", "..")) or ".." in Path(m.name).parts:
            sys.exit("unsafe path in archive")
    t.extractall(root)
dest.unlink()
print("installed to", root / f"sherpa-onnx-whisper-{name}", "\nSet ASR_MODEL=%s if it is not base.en." % name)
