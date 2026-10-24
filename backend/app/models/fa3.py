"""FA3-CLIP adapter. Uses the unmodified architecture in vendor/fa3_model.py and the original CLIPProcessor
preprocessing. If the checkpoint is absent the detector reports itself UNAVAILABLE — there is no demo mode and no
substitute heuristic score."""
from __future__ import annotations

import contextlib
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import SETTINGS


@contextlib.contextmanager
def _offline_clip(checkpoint_dir: Path):
    """If HuggingFace is unreachable, build CLIP ViT-B/32 from its default config (the state_dict in
    fa3_model.pt overwrites every CLIP weight) and load the processor from the checkpoint folder."""
    from transformers import CLIPConfig, CLIPModel, CLIPProcessor
    orig_model, orig_proc = CLIPModel.from_pretrained, CLIPProcessor.from_pretrained

    def model_fp(name, *a, **k):
        try:
            return orig_model(name, *a, **k)
        except Exception:
            return CLIPModel(CLIPConfig())

    def proc_fp(name, *a, **k):
        try:
            return orig_proc(str(checkpoint_dir), *a, **k)
        except Exception:
            return orig_proc(name, *a, **k)

    CLIPModel.from_pretrained = staticmethod(model_fp)
    CLIPProcessor.from_pretrained = staticmethod(proc_fp)
    try:
        yield
    finally:
        CLIPModel.from_pretrained, CLIPProcessor.from_pretrained = orig_model, orig_proc


class FA3Detector:
    name = "FA3-CLIP"

    def __init__(self, checkpoint_dir: str | None = None):
        self.checkpoint_dir = Path(checkpoint_dir or SETTINGS.fa3_checkpoint_dir)
        self.available = False
        self.reason: str | None = None
        self.device = "cpu"
        self.param_count: int | None = None
        self.fusion_beta: float | None = None
        self._model = self._processor = self._torch = None
        self._lock = threading.Lock()
        self._loaded = False
        self.fake_idx, self.real_idx = 0, 1
        self.class_mapping: dict | None = None
        self.health: dict | None = None

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            ck = self.checkpoint_dir / "fa3_model.pt"
            if not ck.exists():
                self.reason = ("Checkpoint fa3_model.pt not found in the checkpoint folder. Place your trained "
                               "weights there (see README).")
                return
            try:
                import torch
                from ..vendor.fa3_model import load_model
                with _offline_clip(self.checkpoint_dir):
                    model, processor = load_model(self.checkpoint_dir)
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(self.device).eval()
                self._torch, self._model, self._processor = torch, model, processor
                # the class order the checkpoint was TRAINED with is recorded in its own fa3_config.json; the vendored
                # FA3Config hard-codes {fake: 0, real: 1} and ignores that file, so read the file explicitly
                import json
                cfg = json.loads((self.checkpoint_dir / "fa3_config.json").read_text())
                raw = cfg.get("label2id") or dict(model.config.label2id)
                l2i = {str(k).lower(): int(v) for k, v in dict(raw).items()}
                if set(l2i) != {"fake", "real"} or sorted(l2i.values()) != [0, 1]:
                    self.reason = (f"Checkpoint config has an unexpected class mapping {l2i}; expected exactly the labels "
                                   "'fake' and 'real' on outputs 0 and 1. Refusing to guess which output means fake.")
                    return
                self.fake_idx, self.real_idx = l2i["fake"], l2i["real"]
                self.class_mapping = {"fake": self.fake_idx, "real": self.real_idx, "source": "fa3_config.json label2id" if cfg.get("label2id") else "built-in default (checkpoint config has no label2id)"}
                self.param_count = sum(p.numel() for p in model.parameters())
                self.fusion_beta = float(torch.sigmoid(model.fusion_logit).item())
                self.available = True
                self.health = self._probe_health()
            except Exception as exc:
                self.reason = f"Checkpoint failed to load: {type(exc).__name__}: {str(exc)[:200]}"

    def _probe_health(self) -> dict:
        """Very different synthetic inputs must not all give the same answer. A trained classifier varies; a collapsed
        one (all-real or all-fake whatever the input) cannot support a 'genuine' verdict."""
        rng = np.random.default_rng(0)
        g = np.tile(np.linspace(0, 255, 224, dtype=np.uint8), (224, 1))
        probes = [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8), np.stack([g, g.T, g[::-1]], -1),
                  np.full((224, 224, 3), 127, np.uint8), np.indices((224, 224)).sum(0).astype(np.uint8)[..., None].repeat(3, -1) * 8]
        try:
            ps = [r["fake_prob"] for r in self._score([Image.fromarray(x) for x in probes])]
        except Exception as exc:
            return {"status": "unknown", "message": f"health probe could not run: {type(exc).__name__}"}
        if max(ps) - min(ps) < 1e-3:
            return {"status": "suspect", "probe_fake_probs": ps,
                    "message": "FA3-CLIP returns the same fake probability for very different inputs; it behaves like a constant "
                               "classifier, so a 'genuine' verdict from it would carry no evidence."}
        return {"status": "ok", "probe_fake_probs": ps, "message": "output varies with input (this does not measure accuracy)"}

    def info(self) -> dict:
        self.load()
        return {"class_mapping": self.class_mapping, "health": self.health, "name": self.name, "available": self.available, "reason": self.reason, "device": self.device,
                "params": self.param_count, "fusion_beta": self.fusion_beta,
                "input": "CLIPProcessor: resize 224 (bicubic) + center crop 224",
                "output": "2 logits {0: fake, 1: real}, softmax (uncalibrated)", "calibrated": False}

    def score_images(self, images: list[Image.Image], batch: int = 8) -> list[dict]:
        """-> [{fake_prob, real_prob, logits}] per image, in order."""
        self.load()
        if not self.available:
            raise RuntimeError(self.reason or "FA3-CLIP unavailable")
        return self._score(images, batch)

    def _score(self, images: list[Image.Image], batch: int = 8) -> list[dict]:
        """Inference only (no load/lock) — also used by the health probe while load() still holds the lock."""
        torch, out = self._torch, []
        for i in range(0, len(images), batch):
            chunk = [im.convert("RGB") for im in images[i:i + batch]]
            inputs = self._processor(images=chunk, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            lg = logits.cpu().numpy()
            for p, l in zip(probs, lg):
                out.append({"fake_prob": round(float(p[self.fake_idx]), 5),
                            "real_prob": round(float(p[self.real_idx]), 5),
                            "logits": [round(float(x), 4) for x in l]})
        return out

    def high_pass_map(self, image: Image.Image) -> np.ndarray:
        """The model's own FFT high-pass input representation (NOT an attribution map). -> HxW float 0..1"""
        self.load()
        from ..vendor.fa3_model import FA3CLIPForImageClassification as M
        inputs = self._processor(images=[image.convert("RGB")], return_tensors="pt")
        pv = inputs["pixel_values"].to(self.device)
        with self._torch.no_grad():
            hp = M._high_pass_map(pv)[0].abs().mean(dim=0).cpu().numpy()
        return (hp - hp.min()) / (np.ptp(hp) + 1e-6)
