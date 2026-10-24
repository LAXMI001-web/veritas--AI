"""Live (near-real-time) detection from a microphone and optional webcam.

The browser records short self-contained audio chunks (a few seconds each) and occasional JPEG frames and posts them
here. Each chunk is decoded in a temp folder that is deleted immediately; ONLY small numbers/text are kept, in memory,
for the duration of the session (nothing is written to disk or to the history database).

Same models and the same decision policy as file analysis:
  audio chunk -> CNN+Conformer windows ; Whisper transcript ; wording check ; per-segment voice-vs-wording check
  webcam frame -> FA3-CLIP (face crop if found, per policy)
  rolling assessment = fusion over the last LIVE_ROLLING_S seconds of evidence.
Live scores are noisier than file analysis (short windows, few frames); the response says so."""
from __future__ import annotations

import io
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from .. import registry
from ..config import SETTINGS, Policy
from ..models.audio_model import SAMPLE_RATE, WINDOW_SECONDS, load_audio_16k
from . import fusion, textcheck, transcript as transcript_svc, video as video_svc
from .aggregate import aggregate, all_aggregates, dispersion, suspicious_intervals
from .media import MediaError, cleanup, extract_audio_wav, temp_dir

LIVE_ROLLING_S = 30.0
MAX_SESSIONS = 16
IDLE_EXPIRE_S = 1800
MAX_KEEP = 800            # items kept per list


class LiveSession:
    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.created = self.touched = time.time()
        self.lock = threading.Lock()
        self.audio_t = 0.0                  # seconds of audio received so far
        self.windows: list[dict] = []       # absolute-time windows
        self.segments: list[dict] = []      # absolute-time transcript segments (+ per-segment checks)
        self.frames: list[dict] = []
        self.frame_n = 0
        self.t0 = time.time()
        self.video_unavailable: str | None = None
        self.transcript_unavailable: str | None = None

    def _trim(self):
        for lst in (self.windows, self.segments, self.frames):
            del lst[:-MAX_KEEP]


_SESSIONS: dict[str, LiveSession] = {}
_SLOCK = threading.Lock()


def start_session() -> LiveSession:
    with _SLOCK:
        now = time.time()
        for k in [k for k, s in _SESSIONS.items() if now - s.touched > IDLE_EXPIRE_S]:
            _SESSIONS.pop(k, None)
        if len(_SESSIONS) >= MAX_SESSIONS:
            raise MediaError("Too many live sessions are open on this server.", "busy")
        s = LiveSession()
        _SESSIONS[s.id] = s
        return s


def get_session(sid: str) -> LiveSession:
    s = _SESSIONS.get(sid)
    if s is None:
        raise MediaError("Live session not found or expired — start a new one.", "no_session")
    s.touched = time.time()
    return s


def end_session(sid: str) -> bool:
    with _SLOCK:
        return _SESSIONS.pop(sid, None) is not None


def _rolling_assessment(s: LiveSession, pol: Policy) -> dict:
    now_a = s.audio_t
    lo = max(0.0, now_a - LIVE_ROLLING_S)
    wins = [w for w in s.windows if w["end"] > lo]
    sc = [w["score"] for w in wins if w["score"] is not None]
    audio_res = None
    if wins:
        span = (wins[-1]["end"] - wins[0]["start"]) if wins else 0.0
        agg = aggregate(sc, pol.audio_aggregation, pol.frame_suspicious_cut, pol.topk)
        audio_res = {"status": "ok", "modality": "audio", "windows": wins, "scored": len(sc), "duration_s": round(span, 2),
                     "aggregation_method": pol.audio_aggregation, "score": None if agg is None else round(agg, 4),
                     "dispersion": dispersion(sc), "aggregates": all_aggregates(sc, pol.frame_suspicious_cut, pol.topk),
                     "model_health": getattr(registry.get_audio(), "health", None),
                     "model_validation": {"validated": getattr(registry.get_audio(), "validated", True), "external": getattr(registry.get_audio(), "external_validation", False), "note": getattr(registry.get_audio(), "validation_note", None),
                                          "threshold": getattr(registry.get_audio(), "decision_threshold", None)},
                     "intervals": suspicious_intervals(wins, pol.interval_min_score, pol.interval_min_len)}
    flo = time.time() - s.t0 - LIVE_ROLLING_S
    fr = [f for f in s.frames if f["t"] >= flo and f["score"] is not None]
    video_res = None
    if fr and not s.video_unavailable:
        fs = [f["score"] for f in fr]
        agg = aggregate(fs, pol.frame_aggregation, pol.frame_suspicious_cut, pol.topk)
        video_res = {"status": "ok", "modality": "video", "scored": len(fs), "aggregation_method": pol.frame_aggregation,
                     "score": None if agg is None else round(agg, 4), "dispersion": dispersion(fs),
                     "aggregates": all_aggregates(fs, pol.frame_suspicious_cut, pol.topk),
                     "intervals": suspicious_intervals([{"score": f["score"], "start": f["t"], "end": f["t"]} for f in fr],
                                                       pol.interval_min_score, pol.interval_min_len)}
    if video_res is not None:
        assess = fusion.fuse(video_res, audio_res, pol, "video", True)
    else:
        assess = fusion.fuse(None, audio_res, pol, "audio", True)
        if s.frames and s.video_unavailable:
            assess["conflicts"] = assess.get("conflicts", []) + [f"Video not assessed: {s.video_unavailable}"]
    # transcript wording vs voice over the same rolling span
    segs = [g for g in s.segments if g["end"] > lo]
    text = " ".join(g["text"] for g in segs)
    confs = [g["confidence"] for g in segs if g.get("confidence") is not None]
    tr = {"status": "ok", "segments": segs, "text": text, "word_count": len(text.split()),
          "mean_confidence": round(float(np.mean(confs)), 3) if confs else None} if segs else None
    if s.transcript_unavailable:
        tr = {"status": "unavailable", "reason": s.transcript_unavailable}
    tc = textcheck.analyze(tr, pol) if tr else {"status": "not_applicable", "reason": "No speech transcribed yet."}
    cc = None
    if tr is not None:
        av = assess["modalities"]["audio"]["verdict"]
        cc = textcheck.crosscheck(av, tr, tc, audio_res, pol)
        assess["text_crosscheck"] = textcheck.summary(cc)
        if cc.get("kind") in ("audio_flagged_text_ordinary", "audio_ok_text_suspicious"):
            assess.setdefault("conflicts", []).append("Voice vs wording: " + cc["headline"])
    return {"assessment": assess, "text_check": {k: v for k, v in tc.items() if k != "indicators"} | {"indicators": tc.get("indicators", [])},
            "rolling_seconds": LIVE_ROLLING_S, "audio_seconds_total": round(s.audio_t, 1),
            "frames_total": s.frame_n,
            "audio_aggregate_score": None if audio_res is None else audio_res["score"],
            "video_aggregate_score": None if video_res is None else video_res["score"]}


def add_audio_chunk(s: LiveSession, src: Path, pol: Policy) -> dict:
    """Decode+score+transcribe one chunk (file on disk). Raises MediaError for unusable input."""
    det = registry.get_audio()
    det.load()
    if not det.available:
        raise MediaError(det.reason or "Audio model unavailable.", "model_unavailable")
    tmp = temp_dir()
    try:
        wav = tmp / "c.wav"
        try:
            extract_audio_wav(src, wav)
            audio = load_audio_16k(wav)
        except MediaError:
            raise MediaError("This audio chunk could not be decoded.", "unreadable")
        except Exception:
            raise MediaError("This audio chunk could not be decoded.", "unreadable")
        dur = len(audio) / SAMPLE_RATE
        if dur > SETTINGS.live_max_chunk_s:
            raise MediaError(f"Chunk longer than {SETTINGS.live_max_chunk_s}s.", "too_long")
        with s.lock:
            off = s.audio_t
        wins = det.score_windows(audio, hop_s=pol.audio_window_hop_s)
        for w in wins:
            w["start"] = round(w["start"] + off, 3); w["end"] = round(w["end"] + off, 3)
        tr = {"status": "unavailable", "reason": "Transcription disabled by policy."}
        if pol.transcribe:
            tr = transcript_svc.get_transcriber().transcribe_audio(audio, offset_s=off)
        new_segs = []
        if tr.get("status") == "ok":
            audio_res = {"status": "ok", "windows": wins}
            for sg in tr["segments"]:
                sc = textcheck._seg_audio_score(audio_res, sg["start"], sg["end"])
                ac = textcheck._audio_class(sc, pol)
                tcl = textcheck.classify_text(sg["text"], pol) if len(sg["text"].split()) >= 3 else \
                    {"class": "ordinary", "score": 0.0, "indicators": [], "categories": 0}
                mism = (ac == "suspected" and tcl["class"] == "ordinary") or (ac == "authentic" and tcl["class"] == "suspicious")
                new_segs.append({**sg, "audio_score": sc, "audio_class": ac, "text_class": tcl["class"],
                                 "indicators": tcl["indicators"], "mismatch": mism,
                                 "low_confidence": sg.get("confidence") is not None and sg["confidence"] < pol.asr_min_confidence})
        with s.lock:
            s.windows += wins
            s.segments += new_segs
            s.audio_t = off + dur
            if tr.get("status") != "ok":
                s.transcript_unavailable = tr.get("reason") if tr.get("status") == "unavailable" else None
            else:
                s.transcript_unavailable = None
            s._trim()
            roll = _rolling_assessment(s, pol)
        scored = [w["score"] for w in wins if w["score"] is not None]
        return {"chunk": {"start": round(off, 3), "end": round(off + dur, 3), "duration_s": round(dur, 2),
                          "windows": wins, "score": None if not scored else round(float(np.mean(scored)), 4),
                          "silent": all(w["silent"] for w in wins), "short": dur < getattr(det, "window_seconds", WINDOW_SECONDS)},
                "transcript": {"status": tr.get("status"), "reason": tr.get("reason"), "segments": new_segs},
                **roll,
                "note": "Live scores come from short windows and few samples; they are noisier than a full-file analysis."}
    finally:
        cleanup(tmp)


def add_frame(s: LiveSession, jpeg: bytes, pol: Policy) -> dict:
    try:
        im = Image.open(io.BytesIO(jpeg))
        im.load()
        if im.width * im.height > 4096 * 4096:
            raise MediaError("Frame too large.", "too_large")
        rgb = np.array(im.convert("RGB"))
    except MediaError:
        raise
    except Exception:
        raise MediaError("Frame could not be decoded.", "unreadable")
    with s.lock:
        n = s.frame_n
        s.frame_n += 1
    res = video_svc.analyze_frames([{"index": n, "t": None, "rgb": rgb, "error": None}], pol, "video")
    if res["status"] != "ok":
        reason = res.get("reason") or res["status"]
        with s.lock:
            if res["status"] == "unavailable":
                s.video_unavailable = reason
            roll = _rolling_assessment(s, pol)
        return {"frame": {"index": n, "score": None, "status": res["status"], "reason": reason}, **roll}
    f = res["frames"][0]
    if f["score"] is None:
        with s.lock:
            roll = _rolling_assessment(s, pol)
        return {"frame": {"index": n, "score": None, "status": "no_face", "reason": "No face detected in this frame, so it was not scored."}, **roll}
    item = {"index": n, "t": round(time.time() - s.t0, 2), "score": f["score"], "crop": f["crop"]}
    with s.lock:
        s.frames.append(item)
        s._trim()
        roll = _rolling_assessment(s, pol)
    return {"frame": {**item, "status": "ok"}, **roll,
            "note": "Single-frame FA3-CLIP score; live video evidence is a handful of sampled frames, not a full video analysis."}
