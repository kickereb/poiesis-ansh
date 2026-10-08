#!/usr/bin/env python3
"""Generate the matched source and target baseline images."""

from __future__ import annotations

import json

import torch

from model_utils import (
    ARTIFACTS,
    generate,
    image_sha256,
    initial_latents,
    load_config,
    load_pipeline,
    regional_metrics,
    write_model_manifest,
)


def main() -> None:
    config = load_config()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    write_model_manifest(config)
    pipeline = load_pipeline(config)
    latents = initial_latents(pipeline, config)
    with torch.inference_mode():
        source = generate(pipeline, config, config["source_prompt"], latents)
        target = generate(pipeline, config, config["target_prompt"], latents)
    source_path = ARTIFACTS / "baseline_source.png"
    target_path = ARTIFACTS / "baseline_target.png"
    source.save(source_path)
    target.save(target_path)
    result = {
        "source_prompt": config["source_prompt"],
        "target_prompt": config["target_prompt"],
        "seed": config["seed"],
        "source_sha256": image_sha256(source),
        "target_sha256": image_sha256(target),
        "source_target_metrics": regional_metrics(source, target),
    }
    (ARTIFACTS / "baseline_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
