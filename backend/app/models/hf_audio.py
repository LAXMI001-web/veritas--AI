"""Hugging Face audio-classification deepfake detector (e.g. a fine-tuned wav2vec2).

Same interface as AudioDetector so the pipeline, fusion, live mode and UI are unchanged. Differences that matter:
  * preprocessing is the model's OWN feature extractor (preprocessor_config.json) — nothing is guessed;
  * which output means "fake" is taken from the model's config.json id2label (e.g. {0: real, 1: fake}); if the labels are
    ambiguous (LABEL_0/LABEL_1) it uses the fake_index recorded by scripts/fetch_audio_model.py from the model card, or the
    AUDIO_HF_FAKE_INDEX env var, and otherwise REFUSES to load rather than guess; two sources that disagree also refuse;
  * the model's published metrics are on ITS OWN data — the app says so (external validation) and never presents them as
    measured on yours.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import numpy as np

from ..config import SETTINGS

SR = 16000
SILENCE_DBFS = -55.0
FAKE_WORDS = {"fake", "spoof", "spoofed", "synthetic", "ai", "generated", "deepfake", "tts", "cloned", "bogus"}
REAL_WORDS = {"real", "bonafide", "bona", "human", "genuine", "authentic", "live", "natural"}


def _words(label: str) -> set[str]:
    return set(re.split(r"[^a-z]+", str(label).lower())) - {""}


def resolve_fake_index(id2label: dict, meta_idx: int | None, env_idx: int | None) -> tuple[int | None, str, str | None]:
    """-> (fake_index | None, source, error). Never guesses."""
    labels = {int(k): str(v) for k, v in dict(id2label).items()}
    from_cfg = None
    if len(labels) == 2:
        f = [i for i, l in labels.items() if _words(l) & FAKE_WORDS and not (_words(l) & REAL_WORDS)]
        r = [i for i, l in labels.items() if _words(l) & REAL_WORDS and not (_words(l) & FAKE_WORDS)]
        if len(f) == 1 and len(r) == 1 and f[0] != r[0]:
            from_cfg = f[0]
    cands = {"config id2label": from_cfg, "model card (fetch script)": meta_idx, "AUDIO_HF_FAKE_INDEX": env_idx}
    given = {k: v for k, v in cands.items() if v is not None}
    if not given:
        return None, "", (f"Cannot tell which output is 'fake': config labels are {labels} and no fake_index is recorded. "
                          "Set AUDIO_HF_FAKE_INDEX (0 or 1) after reading the model card.")
    if len(set(given.values())) > 1:
        return None, "", f"Sources disagree about which output is 'fake': {given}. Refusing to guess."
    src, idx = next(iter(given.items()))
    return idx, src, None


class HFAudioDetector:
    external_validation = True

    def __init__(self, model_dir: str | None = None):
        self.model_dir = Path(model_dir or SETTINGS.audio_hf_dir)
        self.name = "HF-audio:" + self.model_dir.name
        self.available = False
        self.reason: str | None = None
        self.health: dict | None = None
        self.param_count: int | None = None
        self.fake_idx: int | None = None
        self.label_source = ""
        self.class_mapping: dict | None = None
        self.window_seconds = 4.0
        self.decision_threshold = 0.5
        self.validated = True                      # label order + preprocessing are defined by the model itself
        self.spec = None
        self.meta: dict = {}
        self.validation_note = ("Published metrics for this model come from its own authors' data, not yours. Treat scores as "
                                "a screening signal; it may not generalise to voices, languages, codecs or generators it was not trained on.")
        self._model = self._extractor = self._torch = None
        self._lock = threading.Lock()
        self._loaded = False

    # ---- loading
    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            if not (self.model_dir / "config.json").exists():
                self.reason = (f"No model found in {self.model_dir}. Run: python scripts/fetch_audio_model.py "
                               "(downloads the Hugging Face model once).")
                return
            try:
                import torch
                from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
                mp = self.model_dir / "veritas_meta.json"
                self.meta = json.loads(mp.read_text()) if mp.exists() else {}
                model = AutoModelForAudioClassification.from_pretrained(str(self.model_dir)).eval()
                try:
                    ext = AutoFeatureExtractor.from_pretrained(str(self.model_dir))
                except Exception:
                    ext = None
                env = SETTINGS.audio_hf_fake_index
                idx, src, err = resolve_fake_index(model.config.id2label, self.meta.get("fake_index"), None if env in (None, "") else int(env))
                if err:
                    self.reason = err
                    return
                if SETTINGS.audio_hf_invert:
                    idx, src = 1 - idx, src + " + INVERTED by AUDIO_HF_INVERT"
                self.fake_idx, self.label_source = idx, src
                self.class_mapping = {"fake": idx, "real": 1 - idx, "labels": {int(k): v for k, v in model.config.id2label.items()}, "source": src}
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(self.device)
                self._torch, self._model, self._extractor = torch, model, ext
                self.param_count = sum(p.numel() for p in model.parameters())
                self.available = True
                self.health = self._probe_health()
            except Exception as exc:
                self.available = False
                self.reason = f"Audio model failed to load: {type(exc).__name__}: {str(exc)[:200]}"

    def _probe_health(self) -> dict:
        rng = np.random.default_rng(0)
        t = np.arange(SR * 4) / SR
        probes = [0.1 * rng.standard_normal(SR * 4), 0.3 * np.sin(2 * np.pi * 180 * t) * (1 + .5 * np.sin(2 * np.pi * 3 * t)),
                  0.2 * np.sin(2 * np.pi * 900 * t) + 0.05 * rng.standard_normal(SR * 4), 0.02 * rng.standard_normal(SR * 4)]
        try:
            ps = self._infer([p.astype("float32") for p in probes])
        except Exception as exc:
            return {"status": "unknown", "message": f"health probe could not run: {type(exc).__name__}"}
        if max(ps) - min(ps) < 1e-3:
            return {"status": "suspect", "probe_scores": [round(float(p), 5) for p in ps],
                    "message": "The audio model returns the same score for very different inputs; it behaves like a constant classifier."}
        return {"status": "ok", "probe_scores": [round(float(p), 5) for p in ps], "message": "output varies with input (this does not measure accuracy)"}

    def info(self) -> dict:
        self.load()
        return {"name": self.name, "model": self.name, "available": self.available, "reason": self.reason, "health": self.health,
                "params": self.param_count, "input": f"raw waveform, {SR} Hz mono, {self.window_seconds:.0f} s windows, the model's own feature extractor",
                "output": "softmax over 2 classes; score = P(fake)", "high_means": "fake", "threshold": self.decision_threshold,
                "calibrated": False, "validated": True, "validation": {"external": True, "note": self.validation_note},
                "class_mapping": self.class_mapping, "window_seconds": self.window_seconds, "validation_note": self.validation_note}

    # ---- inference
    def _infer(self, segs: list[np.ndarray]) -> list[float]:
        torch = self._torch
        out = []
        for i in range(0, len(segs), 4):
            chunk = segs[i:i + 4]
            if self._extractor is not None:
                inp = self._extractor(chunk, sampling_rate=SR, return_tensors="pt", padding=True)
                inp = {k: v.to(self.device) for k, v in inp.items() if k in ("input_values", "attention_mask", "input_features")}
            else:
                x = np.stack([(s - s.mean()) / (s.std() + 1e-7) for s in chunk]).astype("float32")
                inp = {"input_values": torch.from_numpy(x).to(self.device)}
            with torch.no_grad():
                p = torch.softmax(self._model(**inp).logits, dim=-1)[:, self.fake_idx]
            out.extend(float(v) for v in p.cpu().numpy())
        return out

    def score_windows(self, audio: np.ndarray, hop_s: float = 1.024, batch: int = 4) -> list[dict]:
        self.load()
        if not self.available:
            raise RuntimeError(self.reason or "audio model unavailable")
        W, n = int(self.window_seconds * SR), len(audio)
        hop = max(int(max(hop_s, 2.0) * SR), 1)          # large model: windows at least 2 s apart
        starts = list(range(0, max(n - W, 0) + 1, hop)) or [0]
        while len(starts) > 40:                           # bound CPU time on very long files
            hop *= 2
            starts = list(range(0, max(n - W, 0) + 1, hop)) or [0]
        if n > W and starts[-1] + W < n:
            starts.append(n - W)
        out, segs, idx = [], [], []
        for s in starts:
            seg = audio[s:s + W]
            padded = len(seg) < W
            dbfs = 20 * np.log10(float(np.sqrt(np.mean(np.square(seg))) + 1e-12))
            rec = {"start": s / SR, "end": min(s + W, n) / SR, "score": None, "raw": None, "silent": bool(dbfs < SILENCE_DBFS),
                   "padded": padded, "rms_dbfs": round(float(dbfs), 1)}
            out.append(rec)
            if not rec["silent"]:
                segs.append(np.asarray(seg, dtype="float32")); idx.append(len(out) - 1)
        if segs:
            for j, p in zip(idx, self._infer(segs)):
                out[j]["raw"] = out[j]["score"] = round(float(p), 5)
        return out
