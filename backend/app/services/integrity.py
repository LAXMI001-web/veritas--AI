"""Media integrity inspection. Strictly separates: metadata observations, heuristic indicators, unavailable checks.
Nothing here is a deepfake classification; missing metadata is never treated as evidence of manipulation."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import SETTINGS
from .media import MediaError, ffprobe, duration_of, sha256_file, run

C2PA_MARKERS = (b"c2pa", b"jumb", b"cai.")


def provenance_check(path: Path) -> dict:
    """Detects *presence* of a C2PA/JUMBF manifest marker only. No signature verification is implemented, so
    provenance is never reported as verified."""
    found = False
    try:
        with open(path, "rb") as f:
            head = f.read(2 << 20)
        found = any(m in head for m in C2PA_MARKERS)
    except OSError:
        pass
    return {"manifest_marker_present": found, "verified": False,
            "note": "Signature/manifest verification is not implemented, so provenance is reported as UNVERIFIED. "
                    "A matching hash only proves equality with a reference file; missing provenance is not evidence "
                    "of manipulation."}


def _video_motion_scan(path: Path, max_frames: int = 3000) -> dict:
    cap = cv2.VideoCapture(str(path))
    diffs, prev, n = [], None, 0
    while n < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (48, 48)).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.abs(g - prev).mean()))
        prev = g
        n += 1
    cap.release()
    return {"diffs": diffs, "frames_scanned": n}


def _video_heuristics(path: Path, fps: float | None) -> dict:
    scan = _video_motion_scan(path)
    d = np.asarray(scan["diffs"])
    res = {"frames_scanned": scan["frames_scanned"], "repeated_frames": None, "abrupt_changes": [], "notes": []}
    if d.size < 5:
        res["notes"].append("Too few frames for repeated-frame / abrupt-change checks.")
        return res
    rep = np.where(d < 0.05)[0]
    res["repeated_frames"] = {"count": int(rep.size), "fraction": round(float(rep.size / d.size), 4),
                              "first_indices": [int(i) + 1 for i in rep[:10]]}
    med = float(np.median(d)); mad = float(np.median(np.abs(d - med))) + 1e-6
    z = (d - med) / (1.4826 * mad)
    cuts = np.where((z > 8) & (d > 12))[0]
    for i in cuts[:25]:
        res["abrupt_changes"].append({"frame": int(i) + 1, "t": round((int(i) + 1) / fps, 3) if fps else None,
                                      "mean_abs_diff": round(float(d[i]), 2)})
    res["notes"].append("Abrupt changes include ordinary scene cuts and edits; they are review pointers only.")
    return res


def _audio_heuristics(wav_audio: np.ndarray, sr: int) -> dict:
    res = {"clipping_fraction": round(float((np.abs(wav_audio) > 0.999).mean()), 5), "digital_silence_fraction": None,
           "discontinuities": []}
    res["digital_silence_fraction"] = round(float((np.abs(wav_audio) < 1e-5).mean()), 4)
    hop = int(0.02 * sr)
    if len(wav_audio) > hop * 10:
        n = len(wav_audio) // hop
        rms = np.sqrt((wav_audio[: n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-9
        db = 20 * np.log10(rms)
        jump = np.abs(np.diff(db))
        idx = np.where((jump > 40) & (np.maximum(db[1:], db[:-1]) > -45))[0]
        for i in idx[:25]:
            res["discontinuities"].append({"t": round((int(i) + 1) * 0.02, 2), "level_jump_db": round(float(jump[i]), 1)})
    return res


def inspect(path: Path, kind: str, probe: dict | None = None) -> dict:
    probe = probe or ffprobe(path)
    fmt = probe.get("format") or {}
    streams = probe.get("streams", [])
    vs = [s for s in streams if s.get("codec_type") == "video"]
    au = [s for s in streams if s.get("codec_type") == "audio"]
    meta: dict = {
        "file_size_bytes": path.stat().st_size, "sha256": sha256_file(path), "container": fmt.get("format_name"),
        "duration_s": duration_of(probe), "bit_rate": fmt.get("bit_rate"), "tags": fmt.get("tags") or {},
    }
    if vs:
        v = vs[0]
        rate = v.get("avg_frame_rate", "0/1")
        try:
            a, b = rate.split("/"); fps = float(a) / float(b) if float(b) else None
        except Exception:
            fps = None
        meta["video"] = {"codec": v.get("codec_name"), "profile": v.get("profile"), "width": v.get("width"),
                         "height": v.get("height"), "fps": round(fps, 3) if fps else None,
                         "pix_fmt": v.get("pix_fmt"), "nb_frames": v.get("nb_frames"),
                         "duration_s": float(v["duration"]) if v.get("duration") not in (None, "N/A") else None,
                         "encoder_tag": (v.get("tags") or {}).get("encoder")}
    if au:
        a = au[0]
        meta["audio"] = {"codec": a.get("codec_name"), "sample_rate": a.get("sample_rate"), "channels": a.get("channels"),
                         "duration_s": float(a["duration"]) if a.get("duration") not in (None, "N/A") else None,
                         "bit_rate": a.get("bit_rate")}
    observations, heuristics, unavailable = [], {}, []
    if not (fmt.get("tags")):
        observations.append("No container metadata tags present (common after re-encoding or social-media upload; "
                            "not evidence of manipulation).")
    enc = (fmt.get("tags") or {}).get("encoder") or meta.get("video", {}).get("encoder_tag")
    if enc:
        observations.append(f"Encoder tag: {enc}.")
    if vs and au:
        vd, ad = meta["video"].get("duration_s"), meta["audio"].get("duration_s")
        if vd and ad:
            diff = abs(vd - ad)
            meta["av_duration_diff_s"] = round(diff, 3)
            if diff > 0.5:
                observations.append(f"Audio and video stream durations differ by {diff:.2f}s (can result from muxing, "
                                    "trimming, or editing).")
    if kind == "video":
        try:
            heuristics["video"] = _video_heuristics(path, meta["video"].get("fps") if "video" in meta else None)
        except Exception:
            unavailable.append("Video frame-level heuristics failed to run.")
    if kind in ("audio", "video") and au:
        import tempfile, librosa
        from .media import extract_audio_wav, temp_dir, cleanup
        td = temp_dir()
        try:
            extract_audio_wav(path, td / "a.wav")
            y, sr = librosa.load(str(td / "a.wav"), sr=None, mono=True)
            heuristics["audio"] = _audio_heuristics(y, sr)
        except Exception:
            unavailable.append("Audio heuristics failed to run.")
        finally:
            cleanup(td)
    if kind == "image":
        try:
            from PIL import Image
            im = Image.open(path)
            exif = im.getexif()
            meta["image"] = {"format": im.format, "size": list(im.size), "mode": im.mode, "has_exif": bool(exif),
                             "exif_software": str(exif.get(305)) if exif and exif.get(305) else None}
            if not exif:
                observations.append("No EXIF metadata (common after re-saving or web upload; not evidence of manipulation).")
        except Exception:
            unavailable.append("Image metadata could not be read.")
    unavailable += ["Cryptographic provenance / signature verification (not implemented).",
                    "Audio-video synchronization check runs only when enabled in settings."]
    return {"metadata": meta, "observations": observations, "heuristic_indicators": heuristics,
            "provenance": provenance_check(path), "unavailable_checks": unavailable,
            "disclaimer": "Heuristic indicators are review pointers, not deepfake classifications."}
