"""Frame sampling, optional face crop, FA3 scoring and aggregation for video and still images."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .. import registry
from ..config import SETTINGS, Policy
from . import faces as faces_svc
from .aggregate import all_aggregates, dispersion, suspicious_intervals, aggregate, effective_k
from .media import MediaError

def thumb_b64(rgb: np.ndarray, width: int = 192) -> str:
    h, w = rgb.shape[:2]
    im = Image.fromarray(rgb).resize((width, max(int(h * width / w), 1)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


MAX_SIDE = 1280


def _fit(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    if max(h, w) <= MAX_SIDE:
        return rgb
    k = MAX_SIDE / max(h, w)
    return cv2.resize(rgb, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)


def sample_video_frames(path: Path, n: int, max_seconds: float | None = None) -> tuple[list[dict], dict]:
    """Uniformly sample n CANDIDATE frames (frames wider than 1280 px are downscaled). Returns (frames, meta).
    Each frame: {index,t,rgb|None,error|None}."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise MediaError("Video could not be opened.", "unreadable")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0 or fps > 1000:
            fps = 0.0
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        meta = {"container_frame_count": total, "fps": round(fps, 3) if fps else None, "width": width,
                "height": height, "requested": n}
        if total <= 0:
            # frame count unknown -> read sequentially up to a cap and subsample
            frames_all = []
            while len(frames_all) < 2000:
                ok, fr = cap.read()
                if not ok:
                    break
                frames_all.append(_fit(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            total = len(frames_all)
            if total == 0:
                raise MediaError("No decodable video frames.", "unreadable")
            meta["note"] = "frame count unknown in container; decoded sequentially"
            picks = np.unique(np.linspace(0, total - 1, min(n, total)).round().astype(int))
            frames = [{"index": int(i), "t": round(i / fps, 3) if fps else None, "rgb": frames_all[i], "error": None}
                      for i in picks]
            return frames, meta
        limit = total if not max_seconds or not fps else min(total, int(max_seconds * fps))
        picks = np.unique(np.linspace(0, max(limit - 1, 0), min(n, limit)).round().astype(int))
        frames = []
        for i in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            t = round(float(i) / fps, 3) if fps else None
            if not ok or fr is None:
                frames.append({"index": int(i), "t": t, "rgb": None, "error": "frame could not be decoded"})
            else:
                frames.append({"index": int(i), "t": t, "rgb": _fit(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), "error": None})
        meta["sampled"] = len(frames)
        return frames, meta
    finally:
        cap.release()


def _dup_flags(frames: list[dict]) -> None:
    prev = None
    for f in frames:
        f["duplicate_of_prev"] = False
        if f["rgb"] is None:
            continue
        g = cv2.resize(cv2.cvtColor(f["rgb"], cv2.COLOR_RGB2GRAY), (32, 32)).astype(np.float32)
        if prev is not None and float(np.abs(g - prev).mean()) < 0.5:
            f["duplicate_of_prev"] = True
        prev = g


def _select_face_frames(cands: list[dict], n: int) -> list[dict]:
    """Keep n frames that contain a face, spread evenly across the timeline (temporal coverage, not the first n)."""
    withf = [f for f in cands if f.get("faces")]
    if len(withf) <= n:
        return withf
    idx = np.unique(np.linspace(0, len(withf) - 1, n).round().astype(int))
    return [withf[i] for i in idx]


def analyze_frames(frames: list[dict], pol: Policy, source: str, max_scored: int | None = None) -> dict:
    """Detect faces, score the face crops with FA3 and aggregate. `source` = 'video' | 'image'.

    video: only frames with a detectable face are scored (FA3 is face-trained); frames without one are reported as
    `no_face`, never silently scored as whole frames unless Policy.video_full_frame_fallback is set. If the face detector
    itself is missing the analysis is 'unavailable' — it does NOT degrade to whole-frame scoring.
    image: the whole image is scored exactly as in the original FA3 app."""
    det = registry.get_fa3()
    det.load()
    if not det.available:
        return {"status": "unavailable", "reason": det.reason, "modality": source}
    ok = [f for f in frames if f["rgb"] is not None]
    if not ok:
        return {"status": "failed", "reason": "No frames could be decoded.", "modality": source}
    use_faces = pol.video_face_crop == "auto" and source == "video"
    fdet = faces_svc.get_detector()
    face_info = fdet.info() if use_faces else None
    if use_faces:
        if not fdet.available and not pol.video_full_frame_fallback:
            return {"status": "unavailable", "modality": source,
                    "reason": f"Face detector unavailable ({fdet.reason}). FA3-CLIP was trained on face crops, so whole frames are "
                              "not scored silently; fix the detector or enable the whole-frame fallback explicitly."}
        for f in ok:
            try:
                f["faces"] = fdet.detect(f["rgb"], pol.max_faces_per_frame) if fdet.available else []
            except Exception:
                f["faces"] = []
    n_cand = len(ok)
    face_cands = sum(1 for f in ok if f.get("faces"))
    scored_set = ok
    if use_faces and not pol.video_full_frame_fallback:
        scored_set = _select_face_frames(ok, max_scored or pol.n_frames)
    elif use_faces and max_scored:
        withf = _select_face_frames(ok, max_scored)
        scored_set = withf + [f for f in ok if not f.get("faces")][:max(0, max_scored - len(withf))]
        scored_set.sort(key=lambda f: f["index"])
    picked = {id(f) for f in scored_set}
    _dup_flags(scored_set)
    crops, owners = [], []
    for f in scored_set:
        if use_faces and f.get("faces"):
            f["face_boxes"] = [faces_svc.pad_box(b, f["rgb"].shape[:2]) for b in f["faces"]]
            f["face_box"] = f["face_boxes"][0]; f["crop"] = "face"
            for bx in f["face_boxes"]:
                crops.append(Image.fromarray(np.ascontiguousarray(faces_svc.crop(f["rgb"], bx)))); owners.append(f)
        else:
            f["crop"] = "full_frame" if (not use_faces or pol.video_full_frame_fallback) else "no_face"
            f["face_box"] = None
            if f["crop"] == "full_frame":
                crops.append(Image.fromarray(f["rgb"])); owners.append(f)
    try:
        scores = det.score_images(crops) if crops else []
    except Exception as exc:
        return {"status": "failed", "reason": f"FA3-CLIP inference failed: {type(exc).__name__}", "modality": source}
    for f in scored_set:
        f["face_scores"] = []
    for f, s in zip(owners, scores):
        f["face_scores"].append(s["fake_prob"])
        # the frame is as suspicious as its most suspicious face
        if f.get("score") is None or s["fake_prob"] > f["score"]:
            f["score"] = s["fake_prob"]; f["logits"] = s["logits"]
    items = []
    for f in frames:
        if f["rgb"] is not None and use_faces and id(f) not in picked and not f.get("faces"):
            kind = "no_face"
        else:
            kind = f.get("crop")
        if f["rgb"] is not None and id(f) not in picked and (f.get("faces") or not use_faces):
            continue                                        # face frame not selected (surplus candidate): not part of the evidence
        rec = {"index": f["index"], "t": f["t"], "start": f["t"], "end": f["t"], "score": f.get("score"),
               "logits": f.get("logits"), "crop": kind, "face_box": f.get("face_box"), "faces": len(f.get("face_boxes") or []),
               "face_scores": f.get("face_scores") or None, "duplicate_of_prev": f.get("duplicate_of_prev", False),
               "error": f["error"]}
        if SETTINGS.store_thumbnails and f["rgb"] is not None and (f.get("score") is not None or source == "image"):
            rec["thumb"] = thumb_b64(f["rgb"])
        items.append(rec)
    scored_items = [r for r in items if r["score"] is not None]
    sc = [r["score"] for r in scored_items]
    unique_sc = [r["score"] for r in scored_items if not r["duplicate_of_prev"]]
    use = unique_sc or sc
    k = effective_k(len(use), pol.topk, pol.topk_frac) if use else pol.topk
    agg = all_aggregates(use, pol.frame_suspicious_cut, k)
    primary = aggregate(use, pol.frame_aggregation, pol.frame_suspicious_cut, k)
    ts = [r["t"] for r in scored_items if r["t"] is not None]
    res = {
        "status": "ok", "modality": source, "model": det.name, "frames": items, "scored": len(sc),
        "failed_frames": sum(1 for r in items if r["error"]), "duplicate_frames": sum(r["duplicate_of_prev"] for r in items),
        "aggregates": agg, "aggregation_method": pol.frame_aggregation, "topk_used": k,
        "score": None if primary is None else round(primary, 4), "dispersion": dispersion(sc),
        "max_frame_score": round(max(sc), 4) if sc else None,
        "suspicious_frames": [r["index"] for r in scored_items if r["score"] >= pol.frame_suspicious_cut],
        "intervals": suspicious_intervals(items, pol.interval_min_score, pol.interval_min_len),
        "crop_modes": {m: sum(1 for r in items if r.get("crop") == m) for m in ("face", "full_frame", "no_face")},
        "class_mapping": getattr(det, "class_mapping", None), "model_health": getattr(det, "health", None),
    }
    if source == "video":
        res["evidence"] = {"candidate_frames": n_cand, "candidate_frames_with_face": face_cands,
                           "face_detection_rate": round(face_cands / n_cand, 3) if n_cand else 0.0,
                           "scored_frames": len(sc), "face_detector": face_info,
                           "t_first": min(ts) if ts else None, "t_last": max(ts) if ts else None}
        if use_faces and not sc:
            res["reason"] = (f"No face was detected in any of {n_cand} candidate frames, so there is nothing FA3-CLIP can assess "
                             "(it is trained on faces). This is NOT evidence that the video is genuine.")
    return res


def analyze_image(path: Path, pol: Policy) -> dict:
    try:
        im = Image.open(path)
        im.load()
        rgb = np.array(im.convert("RGB"))
    except Exception:
        raise MediaError("Image could not be decoded.", "unreadable")
    fr = [{"index": 0, "t": None, "rgb": rgb, "error": None}]
    res = analyze_frames(fr, pol, "image")
    if res["status"] == "ok":
        res["frames"][0]["thumb"] = thumb_b64(rgb, 320) if SETTINGS.store_thumbnails else None
        res["image_size"] = [rgb.shape[1], rgb.shape[0]]
        res["note_crop"] = "Whole image preprocessed exactly as in the original FA3 app (resize + center crop 224)."
    return res


def analyze_video(path: Path, pol: Policy, duration: float | None, progress=None) -> dict:
    n_cand = min(pol.n_frames * pol.video_candidate_factor, 256)
    frames, meta = sample_video_frames(path, n_cand, SETTINGS.max_media_seconds)
    if progress:
        progress(0.35, "detecting faces and scoring frames")
    res = analyze_frames(frames, pol, "video", max_scored=pol.n_frames)
    dur = duration or (meta.get("container_frame_count") / meta["fps"] if meta.get("fps") and meta.get("container_frame_count") else None)
    if res.get("status") == "ok" and "evidence" in res and dur and res["evidence"]["t_first"] is not None:
        res["evidence"]["temporal_coverage"] = round(min(1.0, (res["evidence"]["t_last"] - res["evidence"]["t_first"]) / dur), 3)
    res["sampling"] = {**meta, "strategy": f"uniform candidates ({n_cand}), face frames kept evenly", "duration_s": duration}
    return res
