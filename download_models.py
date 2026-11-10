"""Download pinned official weights. No audio is uploaded to Hugging Face."""
import argparse
import json
from pathlib import Path


def download(root, model):
    from huggingface_hub import snapshot_download

    root = Path(root)
    catalog = json.loads((root / "model_catalog.json").read_text(encoding="utf-8"))
    for key in (model, "aligner"):
        entry = catalog[key]
        print(f"下载 {entry['repo']} ...", flush=True)
        snapshot_download(
            entry["repo"], revision=entry["revision"],
            local_dir=root / "models" / entry["directory"],
            allow_patterns=["*.json", "*.safetensors", "*.txt"], max_workers=4,
        )
    print("模型已下载完成。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--model", choices=["qwen3-1.7b", "qwen3-0.6b"], default="qwen3-1.7b")
    args = parser.parse_args()
    download(args.root, args.model)
