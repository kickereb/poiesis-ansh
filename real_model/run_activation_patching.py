#!/usr/bin/env python3
"""Run matched activation-patching interventions on Stable Diffusion."""

from __future__ import annotations

import csv
import json

import torch

from activation_tools import ActivationPatcher, ActivationRecorder
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


def write_csv(path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def intervention_rows(
    condition: str,
    patched_calls: int,
    source,
    target,
    patched,
) -> list[dict]:
    """Measure change from target and restoration toward the source."""
    baseline = {
        row["region"]: row for row in regional_metrics(source, target)
    }
    target_change = {
        row["region"]: row for row in regional_metrics(target, patched)
    }
    source_distance = {
        row["region"]: row for row in regional_metrics(source, patched)
    }
    rows = []
    for region in baseline:
        denominator = baseline[region]["rmse"]
        restoration = 1.0 - source_distance[region]["rmse"] / max(
            denominator, 1e-12
        )
        rows.append(
            {
                "condition": condition,
                "patched_calls": patched_calls,
                "region": region,
                "baseline_source_target_rmse": denominator,
                "target_change_rmse": target_change[region]["rmse"],
                "target_change_mae": target_change[region]["mae"],
                "source_distance_rmse": source_distance[region]["rmse"],
                "source_distance_mae": source_distance[region]["mae"],
                "restoration": restoration,
                "changed_pixel_share_1pct": target_change[region][
                    "changed_pixel_share_1pct"
                ],
            }
        )
    return rows


def main() -> None:
    config = load_config()
    pipeline = load_pipeline(config)
    module = pipeline.unet.get_submodule(config["patch_module"])
    latents = initial_latents(pipeline, config)
    output_directory = ARTIFACTS / "patching"
    output_directory.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        with ActivationRecorder(module) as source_recording:
            source = generate(
                pipeline, config, config["source_prompt"], latents
            )
        with ActivationRecorder(module) as target_recording:
            target = generate(
                pipeline, config, config["target_prompt"], latents
            )

    if source_recording.call_index != config["steps"]:
        raise AssertionError("source activation count does not match denoising steps")
    if target_recording.call_index != config["steps"]:
        raise AssertionError("target activation count does not match denoising steps")

    source.save(output_directory / "source.png")
    target.save(output_directory / "target.png")
    rows: list[dict] = []
    outputs = {
        "source_sha256": image_sha256(source),
        "target_sha256": image_sha256(target),
        "module": config["patch_module"],
        "patch_scope": "Complete BasicTransformerBlock output; conditional CFG branch only.",
        "equal_window_lengths": len(
            {len(indices) for indices in config["patch_windows"].values()}
        )
        == 1,
        "windows": {},
    }

    all_steps = set(range(config["steps"]))
    with torch.inference_mode():
        with ActivationPatcher(
            module,
            target_recording.activations,
            all_steps,
        ) as identity_patcher:
            identity = generate(
                pipeline, config, config["target_prompt"], latents
            )
    identity.save(output_directory / "identity_control.png")
    rows.extend(
        intervention_rows(
            "identity_control",
            identity_patcher.patched_calls,
            source,
            target,
            identity,
        )
    )
    outputs["identity_control_sha256"] = image_sha256(identity)

    for name, indices in config["patch_windows"].items():
        with torch.inference_mode():
            with ActivationPatcher(
                module,
                source_recording.activations,
                set(indices),
            ) as patcher:
                patched = generate(
                    pipeline, config, config["target_prompt"], latents
                )
        patched.save(output_directory / f"source_into_target_{name}.png")
        difference_heatmap(target, patched).save(
            output_directory / f"source_into_target_{name}_difference.png"
        )
        rows.extend(
            intervention_rows(
                f"source_into_target_{name}",
                patcher.patched_calls,
                source,
                target,
                patched,
            )
        )
        outputs["windows"][name] = {
            "steps": indices,
            "patched_calls": patcher.patched_calls,
            "image_sha256": image_sha256(patched),
        }

    write_csv(output_directory / "activation_patching_metrics.csv", rows)
    (output_directory / "activation_patching_results.json").write_text(
        json.dumps(outputs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
