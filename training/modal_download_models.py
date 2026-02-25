"""
Download trained models from Modal Volume to local project.

Usage:
    modal run training/modal_download_models.py
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

app = modal.App("jurisai-download-models")
volume = modal.Volume.from_name("jurisai-trained-models")


SKIP_PATTERNS = [
    "embeddings/model/model.safetensors",
    "doc_classifier_bert/model/model.safetensors",
    "gaia_sft_model/adapter_model.safetensors",
    "reranker/model/model.safetensors",
    "ner_legal/model/model.safetensors",
    "gaia_summarizer/adapter_model.safetensors",
]


@app.function(volumes={"/models": volume})
def list_and_package() -> dict[str, list[str]]:
    """List all files in the volume and return their contents (skip large model binaries)."""
    result = {}
    for root, dirs, files in os.walk("/models"):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, "/models")
            if any(rel.endswith(p) or rel == p for p in SKIP_PATTERNS):
                size = os.path.getsize(full)
                print(f"  SKIP {rel} ({size} bytes) — stays on Modal Volume")
                continue
            with open(full, "rb") as fh:
                result[rel] = fh.read()
            print(f"  {rel} ({len(result[rel])} bytes)")
    return result


@app.local_entrypoint()
def main():
    print("Downloading trained models from Modal Volume...")

    output_dir = Path("training/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_and_package.remote()

    for rel_path, content in files.items():
        local_path = output_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(content)
        print(f"  Saved: {local_path} ({len(content)} bytes)")

    print(f"\nAll models downloaded to {output_dir}/")
    print("Files:")
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            print(f"  {p.relative_to(output_dir)} ({size:,} bytes)")
