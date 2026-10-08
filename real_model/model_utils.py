"""Shared Stable Diffusion utilities for the real-model experiment."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CACHE = ROOT / "cache"
ARTIFACTS = ROOT / "artifacts"
MANIFESTS = ROOT / "manifests"
CONFIG_PATH = ROOT / "configs" / "pilot.json"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the declared pilot configuration."""
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot_path(config: dict[str, Any]) -> Path:
    """Return the pinned local Hugging Face snapshot path."""
    model_directory = "models--" + config["model_id"].replace("/", "--")
    path = CACHE / model_directory / "snapshots" / config["model_revision"]
    if not path.exists():
        raise FileNotFoundError(f"model snapshot is absent: {path}")
    return path


def declared_dtype(config: dict[str, Any]) -> torch.dtype:
    """Resolve the declared PyTorch data type."""
    choices = {"float32": torch.float32, "float16": torch.float16}
    try:
        return choices[config["dtype"]]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {config['dtype']}") from exc


def load_pipeline(config: dict[str, Any]) -> StableDiffusionPipeline:
    """Load the pinned local model and deterministic DDIM scheduler."""
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    path = snapshot_path(config)
    pipeline = StableDiffusionPipeline.from_pretrained(
        path,
        torch_dtype=declared_dtype(config),
        use_safetensors=True,
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline.scheduler = DDIMScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing="leading",
        clip_sample=False,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to("mps")
    return pipeline


def initial_latents(
    pipeline: StableDiffusionPipeline,
    config: dict[str, Any],
    seed: int | None = None,
) -> torch.Tensor:
    """Create the shared initial latent on CPU, then move it to MPS."""
    seed = config["seed"] if seed is None else seed
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (
        1,
        pipeline.unet.config.in_channels,
        config["height"] // pipeline.vae_scale_factor,
        config["width"] // pipeline.vae_scale_factor,
    )
    latents = torch.randn(shape, generator=generator, dtype=torch.float32)
    return latents.to(device="mps", dtype=declared_dtype(config))


def generate(
    pipeline: StableDiffusionPipeline,
    config: dict[str, Any],
    prompt: str,
    latents: torch.Tensor,
) -> Image.Image:
    """Generate one matched image from an explicit latent."""
    result = pipeline(
        prompt=prompt,
        negative_prompt="",
        height=config["height"],
        width=config["width"],
        num_inference_steps=config["steps"],
        guidance_scale=config["guidance_scale"],
        eta=config["eta"],
        latents=latents.clone(),
        output_type="pil",
    )
    return result.images[0]


def image_array(image: Image.Image) -> np.ndarray:
    """Convert an RGB image to a float array in [0, 1]."""
    return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def image_sha256(image: Image.Image) -> str:
    """Hash canonical RGB pixel bytes."""
    return hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it completely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def regional_metrics(reference: Image.Image, changed: Image.Image) -> list[dict[str, Any]]:
    """Measure paired pixel change in fixed horizontal regions."""
    first = image_array(reference)
    second = image_array(changed)
    if first.shape != second.shape:
        raise ValueError("images must have equal shapes")
    width = first.shape[1]
    regions = {
        "full": slice(0, width),
        "left": slice(0, width // 3),
        "centre": slice(width // 3, 2 * width // 3),
        "right": slice(2 * width // 3, width),
    }
    rows = []
    for name, region in regions.items():
        difference = second[:, region] - first[:, region]
        rows.append(
            {
                "region": name,
                "rmse": float(np.sqrt(np.mean(difference**2))),
                "mae": float(np.mean(np.abs(difference))),
                "mean_signed_change": float(np.mean(difference)),
                "changed_pixel_share_1pct": float(
                    np.mean(np.max(np.abs(difference), axis=2) >= 0.01)
                ),
            }
        )
    return rows


def difference_heatmap(reference: Image.Image, changed: Image.Image) -> Image.Image:
    """Create a visible absolute-difference heatmap."""
    difference = np.mean(
        np.abs(image_array(changed) - image_array(reference)), axis=2
    )
    scale = float(np.quantile(difference, 0.995))
    normalised = np.clip(difference / max(scale, 1e-12), 0.0, 1.0)
    red = np.uint8(np.round(255.0 * normalised))
    green = np.uint8(np.round(100.0 * np.sqrt(normalised)))
    blue = np.uint8(np.round(30.0 * (1.0 - normalised)))
    return Image.fromarray(np.stack([red, green, blue], axis=2), mode="RGB")


def environment_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Return the software and device manifest."""
    import accelerate
    import diffusers
    import huggingface_hub
    import transformers

    return {
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "model_license": "CreativeML OpenRAIL-M",
        "model_card": "https://huggingface.co/CompVis/stable-diffusion-v1-4",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "dtype": config["dtype"],
        "scheduler": config["scheduler"],
    }


def write_model_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Hash the exact model files used by this experiment."""
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(config)
    candidates = sorted(
        file
        for file in path.rglob("*")
        if file.is_file()
        and (
            file.suffix == ".safetensors"
            or file.name.endswith("config.json")
            or file.name in {"merges.txt", "vocab.json", "model_index.json"}
        )
    )
    manifest = environment_manifest(config)
    manifest["files"] = [
        {
            "path": str(file.relative_to(path)),
            "bytes": file.stat().st_size,
            "sha256": file_sha256(file),
        }
        for file in candidates
    ]
    output = MANIFESTS / "model_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
