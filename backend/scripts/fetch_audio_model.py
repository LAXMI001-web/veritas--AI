"""Download a Hugging Face audio deepfake classifier into models/audio_hf/default (run once, needs internet).
Default: garystafford/wav2vec2-deepfake-voice-detector (Apache-2.0, ~1.2 GB; labels per model card: 0=real, 1=fake).
  python scripts/fetch_audio_model.py
  python scripts/fetch_audio_model.py --repo OWNER/NAME --fake-index 1     # another audio-classification model
The app then uses it automatically (AUDIO_BACKEND=auto). Delete models/audio_hf to go back to the Keras models."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="garystafford/wav2vec2-deepfake-voice-detector")
    ap.add_argument("--fake-index", type=int, default=1, choices=[0, 1], help="output index meaning FAKE, from the model card")
    ap.add_argument("--dir", default=str(ROOT / "models" / "audio_hf" / "default"))
    a = ap.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("Run: pip install huggingface_hub")
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {a.repo} to {d} (about 1.2 GB) ...")
    snapshot_download(repo_id=a.repo, local_dir=str(d), allow_patterns=["config.json", "preprocessor_config.json", "*.safetensors", "model.safetensors.index.json"])
    (d / "veritas_meta.json").write_text(json.dumps({"repo": a.repo, "fake_index": a.fake_index, "source": "model card"}, indent=2))
    print("Done. Restart the backend; the audio pill/model card will show HF-audio:default.")


if __name__ == "__main__":
    main()
