"""Import-order guard for native libraries.

In this environment, importing TensorFlow BEFORE transformers' CLIP code (with PyTorch present) crashes the whole
process with a segmentation fault; importing torch + transformers first and TensorFlow afterwards is stable. The app
uses both (FA3-CLIP = torch, audio model = TensorFlow, each loaded lazily), so analysing audio first and an image
second would otherwise kill the server. We therefore import the torch/transformers side eagerly, before anything can
import TensorFlow. If torch/transformers are not installed this is a no-op (audio-only installs keep working)."""
try:
    import torch  # noqa: F401
    from transformers import CLIPConfig, CLIPModel, CLIPProcessor  # noqa: F401
except Exception:      # not installed / broken: FA3 will report itself unavailable on its own
    pass
