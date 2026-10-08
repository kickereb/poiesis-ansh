#!/usr/bin/env python3
"""Download the pinned public Stable Diffusion snapshot."""

from __future__ import annotations

from huggingface_hub import snapshot_download

from model_utils import CACHE, load_config


def main() -> None:
    """Download only the components used by this experiment."""
    config = load_config()
    path = snapshot_download(
        repo_id=config["model_id"],
        revision=config["model_revision"],
        cache_dir=CACHE,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "text_encoder/model.fp16.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ],
        max_workers=4,
    )
    print(path)


if __name__ == "__main__":
    main()
