"""Safe media helpers: subprocess wrappers (no shell), ffprobe, decoding, hashing."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..config import SETTINGS

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


class MediaError(Exception):
    """User-facing, safe-to-display error."""

    def __init__(self, message: str, code: str = "media_error"):
        super().__init__(message)
        self.code = code


def run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout or SETTINGS.request_timeout_s, check=False)
    except subprocess.TimeoutExpired:
        raise MediaError("Media processing timed out.", "timeout")
    except FileNotFoundError:
        raise MediaError("Required media tool (ffmpeg/ffprobe) is not installed on the server.", "tool_missing")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ffprobe(path: Path) -> dict:
    p = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)], 30)
    if p.returncode != 0:
        raise MediaError("File could not be read as media (corrupted or unsupported format).", "unreadable")
    try:
        return json.loads(p.stdout.decode("utf-8", "replace"))
    except Exception:
        raise MediaError("File metadata could not be parsed.", "unreadable")


def detect_kind(path: Path, probe: dict | None = None) -> str:
    """Content-based kind detection: 'image' | 'video' | 'audio'. Does not trust the extension."""
    probe = probe or ffprobe(path)
    streams = probe.get("streams", [])
    vids = [s for s in streams if s.get("codec_type") == "video"]
    auds = [s for s in streams if s.get("codec_type") == "audio"]
    fmt = (probe.get("format") or {}).get("format_name", "")
    if vids:
        v = vids[0]
        nb = v.get("nb_frames")
        is_still = (
            any(x in fmt for x in ("image2", "png_pipe", "jpeg_pipe", "webp_pipe", "bmp_pipe", "gif"))
            and not auds
            and (nb in (None, "1", "N/A"))
        )
        if is_still or (fmt.startswith(("image2", "png_pipe", "jpeg_pipe", "webp_pipe", "bmp_pipe"))):
            return "image"
        if v.get("disposition", {}).get("attached_pic") == 1 and auds:
            return "audio"  # cover-art audio file
        return "video"
    if auds:
        return "audio"
    raise MediaError("No decodable image, audio, or video stream found.", "unsupported")


def duration_of(probe: dict) -> float | None:
    try:
        d = float((probe.get("format") or {}).get("duration"))
        return d if d > 0 else None
    except (TypeError, ValueError):
        for s in probe.get("streams", []):
            try:
                return float(s.get("duration"))
            except (TypeError, ValueError):
                continue
    return None


def has_audio(probe: dict) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe.get("streams", []))


def extract_audio_wav(src: Path, dst: Path) -> None:
    """Extract the audio track to mono PCM wav at its native rate (resampling to 16 kHz is then done with
    librosa, exactly like the original ScamGuard loader)."""
    p = run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-c:a", "pcm_s16le",
             "-t", str(SETTINGS.max_media_seconds), str(dst)])
    if p.returncode != 0 or not dst.exists() or dst.stat().st_size < 64:
        raise MediaError("Audio track could not be extracted.", "audio_extract_failed")


def temp_dir() -> Path:
    base = Path(SETTINGS.data_dir) / "tmp"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="v_", dir=base))


def cleanup(path: Path | None) -> None:
    if path and path.exists():
        shutil.rmtree(path, ignore_errors=True)


def validate_limits(path: Path, probe: dict, kind: str) -> list[str]:
    warns: list[str] = []
    if path.stat().st_size > SETTINGS.max_upload_mb * 1024 * 1024:
        raise MediaError(f"File exceeds the {SETTINGS.max_upload_mb} MB limit.", "too_large")
    d = duration_of(probe)
    if kind in ("audio", "video"):
        if d is None:
            warns.append("Duration could not be determined from metadata.")
        elif d > SETTINGS.max_media_seconds:
            raise MediaError(f"Media is longer than the {SETTINGS.max_media_seconds}s limit.", "too_long")
    return warns
