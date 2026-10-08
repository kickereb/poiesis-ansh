#!/usr/bin/env python3
"""Test fixed interventions across several initial-noise seeds."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from activation_tools import ActivationPatcher, ActivationRecorder
from model_utils import (
    ARTIFACTS,
    file_sha256,
    generate,
    image_sha256,
    initial_latents,
    load_config,
    load_pipeline,
    regional_metrics,
)
from run_gradient_svd_lora import (
    cfg_embeddings,
    cfg_noise_prediction,
    direction_declaration,
    target_trajectory_state,
)
from weight_tools import LowRankDirection, selected_value_modules


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rectangular result rows."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def intervention_rows(
    seed: int,
    intervention: str,
    source,
    target,
    changed,
) -> list[dict]:
    """Measure an intervention against target and source images."""
    baseline = {
        row["region"]: row for row in regional_metrics(source, target)
    }
    target_change = {
        row["region"]: row for row in regional_metrics(target, changed)
    }
    source_distance = {
        row["region"]: row for row in regional_metrics(source, changed)
    }
    rows = []
    for region in baseline:
        denominator = baseline[region]["rmse"]
        rows.append(
            {
                "seed": seed,
                "intervention": intervention,
                "region": region,
                "baseline_source_target_rmse": denominator,
                "target_change_rmse": target_change[region]["rmse"],
                "source_distance_rmse": source_distance[region]["rmse"],
                "restoration": 1.0
                - source_distance[region]["rmse"] / max(denominator, 1e-12),
                "changed_pixel_share_1pct": target_change[region][
                    "changed_pixel_share_1pct"
                ],
            }
        )
    return rows


def aggregate(rows: list[dict]) -> dict:
    """Summarize full-image results without inferential claims."""
    summary = {}
    names = sorted({row["intervention"] for row in rows})
    for name in names:
        selected = [
            row
            for row in rows
            if row["intervention"] == name and row["region"] == "full"
        ]
        restoration = [row["restoration"] for row in selected]
        target_change = [row["target_change_rmse"] for row in selected]
        summary[name] = {
            "seeds": len(selected),
            "restoration_mean": statistics.fmean(restoration),
            "restoration_median": statistics.median(restoration),
            "restoration_min": min(restoration),
            "restoration_max": max(restoration),
            "positive_restoration_share": sum(value > 0 for value in restoration)
            / len(restoration),
            "target_change_rmse_mean": statistics.fmean(target_change),
            "target_change_rmse_median": statistics.median(target_change),
        }
    return summary


def aggregate_objectives(rows: list[dict]) -> dict:
    """Summarize the matched one-step objective across seeds."""
    summary = {}
    strengths = sorted({row["strength"] for row in rows})
    for strength in strengths:
        selected = [row for row in rows if row["strength"] == strength]
        reductions = [row["objective_reduction"] for row in selected]
        summary[f"{strength:+.2f}"] = {
            "seeds": len(selected),
            "objective_reduction_mean": statistics.fmean(reductions),
            "objective_reduction_median": statistics.median(reductions),
            "objective_reduction_min": min(reductions),
            "objective_reduction_max": max(reductions),
            "positive_reduction_share": sum(value > 0 for value in reductions)
            / len(reductions),
        }
    return summary


def main() -> None:
    """Run fixed activation and weight interventions across seeds."""
    config = load_config()
    pipeline = load_pipeline(config)
    patch_module = pipeline.unet.get_submodule(config["patch_module"])
    selections = selected_value_modules(pipeline.unet)
    primary_name, primary_module = selections[len(selections) // 2]

    random_direction = LowRankDirection.random(
        primary_module,
        rank=int(config["lora_rank"]),
        relative_norm=float(config["lora_relative_norm"]),
        seed=int(config["seed"]) + len(selections) // 2,
    )
    fitted_directory = ARTIFACTS / "gradient_svd_lora"
    fitted_result = json.loads(
        (fitted_directory / "gradient_aligned_lora_results.json").read_text(
            encoding="utf-8"
        )
    )
    provenance = fitted_result["direction_provenance"]
    expected_declaration = direction_declaration(
        config, pipeline, primary_name
    )
    if provenance["declaration"] != expected_declaration:
        raise ValueError("the stored direction declaration does not match this run")
    delta_path = fitted_directory / "aligned_rank4_delta.npy"
    factor_path = fitted_directory / "aligned_rank4_factors.npz"
    if file_sha256(delta_path) != provenance["delta_sha256"]:
        raise ValueError("the stored direction checksum does not match")
    if file_sha256(factor_path) != provenance["factor_sha256"]:
        raise ValueError("the stored factor checksum does not match")
    delta = torch.from_numpy(np.load(delta_path))
    factors = np.load(factor_path)
    reconstructed = torch.from_numpy(factors["B"]) @ torch.from_numpy(
        factors["A"]
    )
    torch.testing.assert_close(reconstructed, delta, rtol=1e-10, atol=1e-12)
    weight_norm = float(primary_module.weight.detach().float().norm().cpu())
    relative_norm = float(delta.norm()) / weight_norm
    if not np.isclose(
        relative_norm,
        float(config["lora_relative_norm"]),
        rtol=1e-6,
        atol=1e-9,
    ):
        raise ValueError("the stored direction norm does not match")
    aligned_direction = LowRankDirection(
        module=primary_module,
        delta=delta,
        rank=int(config["lora_rank"]),
        relative_norm=relative_norm,
        seed=int(config["seed"]) + 200 + len(selections) // 2,
    )
    source_embeddings = cfg_embeddings(pipeline, config["source_prompt"])
    target_embeddings = cfg_embeddings(pipeline, config["target_prompt"])

    output_directory = ARTIFACTS / "seed_robustness"
    output_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    objective_rows: list[dict] = []
    image_hashes = {}
    early_steps = set(config["patch_windows"]["early"])

    for seed_value in config["robustness_seeds"]:
        seed = int(seed_value)
        latents = initial_latents(pipeline, config, seed=seed)
        seed_directory = output_directory / str(seed)
        seed_directory.mkdir(parents=True, exist_ok=True)
        state, timestep = target_trajectory_state(
            pipeline,
            config,
            latents,
            target_embeddings,
            int(config["gradient_step_index"]),
        )
        if int(timestep) != int(provenance["scheduler_timestep"]):
            raise ValueError("the scheduler timestep does not match the fitted direction")
        with torch.inference_mode():
            source_reference = cfg_noise_prediction(
                pipeline,
                state,
                timestep,
                source_embeddings,
                float(config["guidance_scale"]),
            )
            base_prediction = cfg_noise_prediction(
                pipeline,
                state,
                timestep,
                target_embeddings,
                float(config["guidance_scale"]),
            )
            base_objective = 0.5 * torch.mean(
                (base_prediction - source_reference) ** 2
            )
            for strength_value in config["objective_robustness_doses"]:
                strength = float(strength_value)
                with aligned_direction.applied(strength):
                    changed_prediction = cfg_noise_prediction(
                        pipeline,
                        state,
                        timestep,
                        target_embeddings,
                        float(config["guidance_scale"]),
                    )
                changed_objective = 0.5 * torch.mean(
                    (changed_prediction - source_reference) ** 2
                )
                objective_rows.append(
                    {
                        "seed": seed,
                        "strength": strength,
                        "baseline_objective": float(base_objective),
                        "changed_objective": float(changed_objective),
                        "objective_reduction": 1.0
                        - float(changed_objective) / float(base_objective),
                    }
                )
        with torch.inference_mode():
            with ActivationRecorder(patch_module) as source_recording:
                source = generate(
                    pipeline, config, config["source_prompt"], latents
                )
            target = generate(
                pipeline, config, config["target_prompt"], latents
            )
            with ActivationPatcher(
                patch_module,
                source_recording.activations,
                early_steps,
            ):
                early_patch = generate(
                    pipeline, config, config["target_prompt"], latents
                )
            with random_direction.applied(1.0):
                random_lora = generate(
                    pipeline, config, config["target_prompt"], latents
                )
            with aligned_direction.applied(0.5):
                aligned_lora = generate(
                    pipeline, config, config["target_prompt"], latents
                )

        images = {
            "source": source,
            "target": target,
            "early_patch": early_patch,
            "random_lora": random_lora,
            "aligned_lora": aligned_lora,
        }
        image_hashes[str(seed)] = {
            name: image_sha256(image) for name, image in images.items()
        }
        for name, image in images.items():
            image.save(seed_directory / f"{name}.png")
        rows.extend(
            intervention_rows(
                seed, "early_activation_patch", source, target, early_patch
            )
        )
        rows.extend(
            intervention_rows(
                seed, "random_rank4_dose_1", source, target, random_lora
            )
        )
        rows.extend(
            intervention_rows(
                seed,
                "gradient_aligned_rank4_dose_0.5",
                source,
                target,
                aligned_lora,
            )
        )

    fitting_seed = int(config["seed"])
    held_out_rows = [row for row in rows if row["seed"] != fitting_seed]
    held_out_objective_rows = [
        row for row in objective_rows if row["seed"] != fitting_seed
    ]
    result = {
        "seeds": [int(seed) for seed in config["robustness_seeds"]],
        "fitting_seed": fitting_seed,
        "primary_module": primary_name,
        "random_direction_relative_norm": random_direction.relative_norm,
        "aligned_direction_relative_norm": aligned_direction.relative_norm,
        "image_hashes": image_hashes,
        "summary": aggregate(rows),
        "held_out_summary": aggregate(held_out_rows),
        "objective_summary": aggregate_objectives(objective_rows),
        "held_out_objective_summary": aggregate_objectives(
            held_out_objective_rows
        ),
        "direction_provenance_validated": True,
        "direction_delta_sha256": provenance["delta_sha256"],
        "direction_factor_sha256": provenance["factor_sha256"],
        "scope": "Descriptive seed robustness for one prompt pair and fixed directions.",
    }
    write_csv(output_directory / "seed_robustness_metrics.csv", rows)
    write_csv(output_directory / "seed_objective_metrics.csv", objective_rows)
    (output_directory / "seed_robustness_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "image_summary": result["summary"],
                "held_out_image_summary": result["held_out_summary"],
                "objective_summary": result["objective_summary"],
                "held_out_objective_summary": result[
                    "held_out_objective_summary"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
