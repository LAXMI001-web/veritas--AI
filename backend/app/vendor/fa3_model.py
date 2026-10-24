"""Research-faithful FA3-CLIP components.

FA3-CLIP adds high-frequency FFT cues, spatial/frequency fusion, and
instance-conditioned live/fake prompt biases to a pretrained CLIP backbone.

This is the exact architecture/training/inference code used to produce the
checkpoint in checkpoints/fa3_artifact_faces/ (fa3_model.pt + fa3_config.json
+ processor files). Do not change the class internals without retraining —
the web app's inference wrapper (model/inference.py) imports this module
as-is and only handles preprocessing/serving around it.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor


class FA3Config:
    def __init__(self, model_name, num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.id2label = {0: "fake", 1: "real"}
        self.label2id = {"fake": 0, "real": 1}

    def to_json_string(self):
        return json.dumps(
            {
                "architecture": "FA3-CLIP",
                "model_name": self.model_name,
                "num_labels": self.num_labels,
                "id2label": self.id2label,
                "label2id": self.label2id,
            },
            indent=2,
        )


class FA3CLIPForImageClassification(nn.Module):
    """FA3-CLIP binary classifier with spatial/frequency CLIP fusion."""

    def __init__(self, model_name="openai/clip-vit-base-patch32", num_labels=2, prompt_length=6):
        super().__init__()
        self.model_name = model_name
        self.config = FA3Config(model_name, num_labels)
        self.clip = CLIPModel.from_pretrained(model_name)

        vision_hidden = self.clip.config.vision_config.hidden_size
        text_hidden = self.clip.config.text_config.hidden_size
        projection_dim = self.clip.config.projection_dim

        # Frequency feature generation: high-pass FFT map plus CNN encoder.
        self.frequency_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.frequency_projection = nn.Linear(64, projection_dim)
        self.spatial_projection = nn.Linear(vision_hidden, projection_dim)

        # Learnable spatial/frequency balance (the paper's beta parameter).
        self.fusion_logit = nn.Parameter(torch.tensor(0.0))
        self.fusion_norm = nn.LayerNorm(projection_dim)

        # Six-token generic contexts and separate live/fake conditional biases.
        self.prompt_context = nn.Parameter(torch.empty(num_labels, prompt_length, text_hidden))
        nn.init.normal_(self.prompt_context, std=0.02)
        self.bias_generators = nn.ModuleList(
            [nn.Sequential(nn.Linear(projection_dim, text_hidden), nn.Tanh()) for _ in range(num_labels)]
        )
        self.text_projection = self.clip.text_projection
        self.visual_projection = self.clip.visual_projection
        self.logit_scale = self.clip.logit_scale
        self.register_buffer("prompt_input_ids", torch.empty((0, 0), dtype=torch.long), persistent=False)

    def set_prompt_input_ids(self, input_ids):
        self.prompt_input_ids = input_ids.long()

    @staticmethod
    def _high_pass_map(pixel_values):
        height, width = pixel_values.shape[-2:]
        spectrum = torch.fft.fftshift(torch.fft.fft2(pixel_values, dim=(-2, -1)), dim=(-2, -1))
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=pixel_values.device),
            torch.linspace(-1.0, 1.0, width, device=pixel_values.device),
            indexing="ij",
        )
        radius = torch.sqrt(xx.square() + yy.square())
        mask = (radius >= 0.18).to(spectrum.dtype)
        high_frequency = torch.fft.ifft2(
            torch.fft.ifftshift(spectrum * mask, dim=(-2, -1)), dim=(-2, -1)
        ).real
        return high_frequency

    def _fused_image_features(self, pixel_values):
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values, return_dict=True)
        spatial = self.spatial_projection(vision_outputs.pooler_output)
        high_frequency = self._high_pass_map(pixel_values)
        frequency = self.frequency_encoder(high_frequency).flatten(1)
        frequency = self.frequency_projection(frequency)
        beta = torch.sigmoid(self.fusion_logit)
        return self.fusion_norm(beta * spatial + (1.0 - beta) * frequency)

    def _prompt_features(self, fused_features):
        if self.prompt_input_ids is None:
            raise RuntimeError("Prompt token IDs were not initialized. Use build_model().")
        prompt_ids = self.prompt_input_ids.to(fused_features.device)
        text_outputs = self.clip.text_model(input_ids=prompt_ids, return_dict=True)
        generic_text = text_outputs.pooler_output
        class_features = []
        for class_idx in range(self.config.num_labels):
            context = self.prompt_context[class_idx].mean(dim=0)
            bias = self.bias_generators[class_idx](fused_features)
            conditioned = generic_text[class_idx].unsqueeze(0) + context + bias
            class_features.append(self.text_projection(conditioned))
        return torch.stack(class_features, dim=1)

    def forward(self, pixel_values=None, labels=None, **kwargs):
        if pixel_values is None:
            raise ValueError("FA3-CLIP requires pixel_values from CLIPProcessor.")
        fused_features = self._fused_image_features(pixel_values)
        # Spatial and frequency streams are already projected to CLIP's joint
        # embedding dimension before fusion.
        image_features = fused_features / fused_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        text_features = self._prompt_features(fused_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        logits = (image_features.unsqueeze(1) * text_features).sum(dim=-1)
        logits = logits * self.logit_scale.exp().clamp(max=100.0)

        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_dir / "fa3_model.pt")
        (output_dir / "fa3_config.json").write_text(self.config.to_json_string())


def build_model(model_name="openai/clip-vit-base-patch32", num_labels=2):
    """Build FA3-CLIP and its CLIP processor."""
    model = FA3CLIPForImageClassification(model_name=model_name, num_labels=num_labels)
    processor = CLIPProcessor.from_pretrained(model_name)
    prompt_inputs = processor(
        text=["a fake face", "a live real face"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77,
    )
    model.set_prompt_input_ids(prompt_inputs["input_ids"])
    return model, processor


def load_model(checkpoint_dir, model_name=None):
    """Load a saved FA3-CLIP checkpoint and its processor."""
    checkpoint_dir = Path(checkpoint_dir)
    config = json.loads((checkpoint_dir / "fa3_config.json").read_text())
    model_name = model_name or config["model_name"]
    model, processor = build_model(model_name=model_name, num_labels=config.get("num_labels", 2))
    state = torch.load(checkpoint_dir / "fa3_model.pt", map_location="cpu")
    model.load_state_dict(state)
    return model, processor
