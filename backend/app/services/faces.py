"""Face DETECTION (not recognition/identification) for the video pipeline.

Why this module exists: FA3-CLIP ("fa3_artifact_faces") was trained on face crops. The previous code used one Haar cascade
loaded through `cv2.data`, wrapped in `except Exception`, and silently scored the WHOLE frame whenever detection returned
nothing or errored (e.g. OpenCV 5 has no CascadeClassifier). Scoring whole frames is out-of-domain for the model and tends
to read as "real", which is a direct route to false-genuine verdicts. Here:
  * cascade XML files are vendored (no dependency on cv2.data) and loaded explicitly;
  * if no detector can be loaded, `available` is False with a reason — callers must not pretend faces were checked;
  * frontal (default + alt2) and profile (+ mirrored profile) cascades are merged with NMS, up to `max_faces` per frame.
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

CASCADE_DIR = Path(__file__).resolve().parent.parent / "vendor" / "cascades"
FRONTAL = ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml")
PROFILE = ("haarcascade_profileface.xml",)
PAD = 0.15                      # same padding rule as the original FA3 app


def _iou(a, b) -> float:
    ax, ay, aw, ah = a[:4]; bx, by, bw, bh = b[:4]
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter + 1e-9)


def _nms(boxes: list[tuple], thr: float = 0.35) -> list[tuple]:
    keep: list[tuple] = []
    for b in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        if all(_iou(b, k) < thr for k in keep):
            keep.append(b)
    return keep


class FaceDetector:
    def __init__(self, cascade_dir: Path | None = None):
        self.dir = Path(cascade_dir or CASCADE_DIR)
        self.available = False
        self.reason: str | None = None
        self._frontal: list = []
        self._profile: list = []
        self._lock = threading.Lock()
        self._loaded = False

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            if not hasattr(cv2, "CascadeClassifier"):
                self.reason = (f"OpenCV {cv2.__version__} has no CascadeClassifier (OpenCV 5.x removed it). "
                               "Install opencv-python-headless>=4.9,<5.")
                return
            for names, bucket in ((FRONTAL, self._frontal), (PROFILE, self._profile)):
                for n in names:
                    p = self.dir / n
                    c = cv2.CascadeClassifier(str(p)) if p.exists() else None
                    if c is not None and not c.empty():
                        bucket.append(c)
            if not self._frontal:
                self.reason = f"No frontal-face cascade could be loaded from {self.dir}."
                return
            self.available = True

    def info(self) -> dict:
        self.load()
        return {"available": self.available, "reason": self.reason, "method": "OpenCV Haar cascades (frontal x2 + profile)",
                "opencv": cv2.__version__, "frontal_cascades": len(self._frontal), "profile_cascades": len(self._profile)}

    def detect(self, rgb: np.ndarray, max_faces: int = 3, min_size: int = 40) -> list[list[int]]:
        """-> [[x, y, w, h], ...] largest first. Raises RuntimeError if the detector is unavailable (never returns
        an empty list to mean 'could not check')."""
        self.load()
        if not self.available:
            raise RuntimeError(self.reason or "face detector unavailable")
        gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        ms = max(min_size, int(0.06 * min(gray.shape[:2])))
        found: list[tuple] = []
        for c in self._frontal:
            for b in c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(ms, ms)):
                found.append(tuple(int(v) for v in b))
        if not found:                                       # profile / turned heads only when no frontal face was found
            flipped = cv2.flip(gray, 1)
            for c in self._profile:
                for b in c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(ms, ms)):
                    found.append(tuple(int(v) for v in b))
                for (x, y, w, h) in c.detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=5, minSize=(ms, ms)):
                    found.append((int(gray.shape[1] - x - w), int(y), int(w), int(h)))
        return [list(b) for b in _nms(found)[:max_faces]]


def pad_box(box: list[int], shape: tuple[int, int]) -> list[int]:
    x, y, w, h = box
    pad = int(PAD * max(w, h))
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(shape[1], x + w + pad), min(shape[0], y + h + pad)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def crop(rgb: np.ndarray, padded_box: list[int]) -> np.ndarray:
    x, y, w, h = padded_box
    return rgb[y:y + h, x:x + w]


_DET: FaceDetector | None = None


def get_detector() -> FaceDetector:
    global _DET
    if _DET is None:
        _DET = FaceDetector()
    return _DET


def set_detector(d: FaceDetector | None) -> None:
    global _DET
    _DET = d