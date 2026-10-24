"""OPTIONAL, EXPERIMENTAL audio-visual speech-activity consistency check.
This is NOT the FA3 classifier and NOT a validated lip-sync model. It compares (a) an energy-based speech-activity
envelope from the audio with (b) frame-difference motion in the lower half of the largest detected face box (Haar,
detection only — no identification). Low correlation can come from dubbing, off-screen speakers, editing, or
detector failure; it is a review pointer only."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .media import extract_audio_wav, temp_dir, cleanup, MediaError

FPS_TARGET = 10
MAX_SECONDS = 90


def check(path: Path) -> dict:
    import librosa
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(fps / FPS_TARGET)), 1)
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    times, motion, face_found, prev, i = [], [], 0, None, 0
    while i / fps < MAX_SECONDS:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            boxes = casc.detectMultiScale(g, 1.1, 5, minSize=(48, 48))
            m = np.nan
            if len(boxes):
                x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
                mouth = cv2.resize(g[y + h // 2: y + h, x: x + w], (48, 24)).astype(np.float32)
                if prev is not None:
                    m = float(np.abs(mouth - prev).mean())
                prev = mouth; face_found += 1
            else:
                prev = None
            times.append(i / fps); motion.append(m)
        i += 1
    cap.release()
    if len(times) < 20:
        return {"status": "inconclusive", "reason": "Too few sampled frames for the check."}
    face_frac = face_found / len(times)
    td = temp_dir()
    try:
        extract_audio_wav(path, td / "a.wav")
        y, sr = librosa.load(str(td / "a.wav"), sr=16000, mono=True, duration=MAX_SECONDS)
    except MediaError as exc:
        return {"status": "unavailable", "reason": str(exc)}
    finally:
        cleanup(td)
    hop = 160
    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=hop)[0]
    rt = np.arange(len(rms)) * hop / 16000
    env = np.interp(times, rt, rms)
    thr = 0.25 * np.percentile(rms, 95) if rms.size else 0
    speech = (env > thr).astype(float)
    mo = np.asarray(motion, dtype=float)
    valid = ~np.isnan(mo)
    if face_frac < 0.5 or valid.sum() < 20 or speech.sum() < 10:
        return {"status": "inconclusive", "face_detected_fraction": round(face_frac, 3),
                "speech_fraction": round(float(speech.mean()), 3),
                "reason": "Insufficient visible face or speech activity for a meaningful comparison."}
    mo_n = (mo - np.nanmean(mo)) / (np.nanstd(mo) + 1e-6)
    sp_n = (speech - speech.mean()) / (speech.std() + 1e-6)
    best = (-2.0, 0.0)
    for lag in range(-4, 5):  # ±0.4 s at 10 fps
        a = mo_n[max(lag, 0):len(mo_n) + min(lag, 0)]
        b = sp_n[max(-lag, 0):len(sp_n) + min(-lag, 0)]
        v = ~np.isnan(a)
        if v.sum() > 15:
            c = float(np.corrcoef(a[v], b[v])[0, 1])
            if c > best[0]:
                best = (c, lag / FPS_TARGET)
    zero_lag = best[0]
    return {"status": "ok", "experimental": True, "face_detected_fraction": round(face_frac, 3),
            "speech_fraction": round(float(speech.mean()), 3), "best_correlation": round(best[0], 3),
            "best_lag_s": round(best[1], 2),
            "interpretation": ("Mouth-region motion tracks speech activity." if best[0] > 0.2 else
                               "Weak correspondence between mouth-region motion and speech activity — review manually "
                               "(may be dubbing, off-screen speaker, editing, or detector failure)."),
            "note": "Not a validated lip-sync detector; no identity or biometric matching is performed."}
