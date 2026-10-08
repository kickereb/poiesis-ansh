#!/usr/bin/env python3
"""Measure low-rank weight effects and finite-difference output responses."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from model_utils import (
    ARTIFACTS,
    difference_heatmap,
    generate,
    image_sha256,
    initial_latents,
    load_config,
    load_pipeline,
    regional_metrics,
)
from weight_tools import (
    LowRankDirection,
    directional_output_gradient,
    gradient_heatmap,
    selected_value_modules,
)


def safe_name(module_name: str) -> str:
    return module_name.replace(".", "_")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = load_config()
    pipeline = load_pipeline(config)
    latents = initial_latents(pipeline, config)
    output_directory = ARTIFACTS / "weights"
    output_directory.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        baseline = generate(
            pipeline, config, config["target_prompt"], latents
        )
    baseline.save(output_directory / "baseline.png")

    rows: list[dict] = []
    gradient_summary = []
    directions: dict[str, LowRankDirection] = {}
    gradients: dict[str, np.ndarray] = {}
    selections = selected_value_modules(pipeline.unet)
    epsilon = float(config["gradient_epsilon"])

    for index, (module_name, module) in enumerate(selections):
        direction = LowRankDirection.random(
            module,
            rank=int(config["lora_rank"]),
            relative_norm=float(config["lora_relative_norm"]),
            seed=int(config["seed"]) + index,
        )
        directions[module_name] = direction
        with torch.inference_mode():
            with direction.applied(-epsilon):
                negative = generate(
                    pipeline, config, config["target_prompt"], latents
                )
            with direction.applied(epsilon):
                positive = generate(
                    pipeline, config, config["target_prompt"], latents
                )
        stem = safe_name(module_name)
        negative.save(output_directory / f"{stem}_negative.png")
        positive.save(output_directory / f"{stem}_positive.png")
        gradient = directional_output_gradient(negative, positive, epsilon)
        gradients[module_name] = gradient
        np.save(output_directory / f"{stem}_directional_gradient.npy", gradient.astype(np.float32))
        gradient_heatmap(gradient).save(
            output_directory / f"{stem}_directional_gradient.png"
        )
        gradient_summary.append(
            {
                "module": module_name,
                "rank": direction.rank,
                "relative_norm": direction.relative_norm,
                "epsilon": epsilon,
                "gradient_rms": float(np.sqrt(np.mean(gradient**2))),
                "gradient_max": float(np.max(np.abs(gradient))),
                "negative_sha256": image_sha256(negative),
                "positive_sha256": image_sha256(positive),
            }
        )
        for sign, image in ((-epsilon, negative), (epsilon, positive)):
            for metric in regional_metrics(baseline, image):
                rows.append(
                    {
                        "module": module_name,
                        "intervention": "central_difference",
                        "strength": sign,
                        "rank": direction.rank,
                        "relative_norm": direction.relative_norm,
                        **metric,
                    }
                )

    primary_name, _primary_module = selections[len(selections) // 2]
    primary = directions[primary_name]
    half_epsilon = epsilon / 2.0
    with torch.inference_mode():
        with primary.applied(-half_epsilon):
            half_negative = generate(
                pipeline, config, config["target_prompt"], latents
            )
        with primary.applied(half_epsilon):
            half_positive = generate(
                pipeline, config, config["target_prompt"], latents
            )
    half_gradient = directional_output_gradient(
        half_negative, half_positive, half_epsilon
    )
    primary_gradient = gradients[primary_name]
    flattened = primary_gradient.reshape(-1)
    half_flattened = half_gradient.reshape(-1)
    cosine = float(
        np.dot(flattened, half_flattened)
        / max(np.linalg.norm(flattened) * np.linalg.norm(half_flattened), 1e-30)
    )
    half_negative.save(output_directory / "primary_half_epsilon_negative.png")
    half_positive.save(output_directory / "primary_half_epsilon_positive.png")
    np.save(
        output_directory / "primary_half_epsilon_directional_gradient.npy",
        half_gradient.astype(np.float32),
    )
    gradient_heatmap(half_gradient).save(
        output_directory / "primary_half_epsilon_directional_gradient.png"
    )

    zero_hash = None
    for strength in config["lora_doses"]:
        with torch.inference_mode():
            with primary.applied(float(strength)):
                image = generate(
                    pipeline, config, config["target_prompt"], latents
                )
        stem = f"dose_{float(strength):+.2f}".replace("+", "p").replace("-", "m")
        image.save(output_directory / f"primary_{stem}.png")
        difference_heatmap(baseline, image).save(
            output_directory / f"primary_{stem}_difference.png"
        )
        for metric in regional_metrics(baseline, image):
            rows.append(
                {
                    "module": primary_name,
                    "intervention": "lora_dose",
                    "strength": float(strength),
                    "rank": primary.rank,
                    "relative_norm": primary.relative_norm,
                    **metric,
                }
            )
        if float(strength) == 0.0:
            zero_hash = image_sha256(image)

    result = {
        "baseline_sha256": image_sha256(baseline),
        "selected_modules": [name for name, _module in selections],
        "primary_lora_module": primary_name,
        "primary_direction_seed": primary.seed,
        "zero_dose_sha256": zero_hash,
        "zero_dose_equal": zero_hash == image_sha256(baseline),
        "finite_difference_check": {
            "epsilon": epsilon,
            "half_epsilon": half_epsilon,
            "gradient_cosine": cosine,
            "rms_ratio_half_to_full": float(
                np.sqrt(np.mean(half_gradient**2))
                / max(np.sqrt(np.mean(primary_gradient**2)), 1e-30)
            ),
        },
        "gradient_summary": gradient_summary,
        "interpretation": {
            "central_difference": "Central finite-difference estimate along one declared rank-4 direction, using 8-bit final images.",
            "lora_dose": "Causal output change from a controlled low-rank weight intervention.",
        },
    }
    write_csv(output_directory / "weight_intervention_metrics.csv", rows)
    (output_directory / "weight_intervention_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
