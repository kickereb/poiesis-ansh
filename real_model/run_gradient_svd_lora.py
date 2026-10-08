#!/usr/bin/env python3
"""Measure selected weight gradients and test a gradient-aligned rank-4 delta."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model_utils import (
    ARTIFACTS,
    MANIFESTS,
    difference_heatmap,
    environment_manifest,
    file_sha256,
    generate,
    image_sha256,
    initial_latents,
    load_config,
    load_pipeline,
    regional_metrics,
)
from weight_tools import LowRankDirection, selected_value_modules


def safe_name(module_name: str) -> str:
    """Convert a module path to a file-safe name."""
    return module_name.replace(".", "_")


def direction_declaration(config: dict, pipeline, module_name: str) -> dict:
    """Return the fields that define the fitted weight direction."""
    keys = [
        "model_id",
        "model_revision",
        "source_prompt",
        "target_prompt",
        "seed",
        "width",
        "height",
        "steps",
        "guidance_scale",
        "scheduler",
        "eta",
        "dtype",
        "gradient_step_index",
        "lora_rank",
        "lora_relative_norm",
    ]
    declaration = {key: config[key] for key in keys}
    declaration["module"] = module_name
    declaration["model_manifest_sha256"] = file_sha256(
        MANIFESTS / "model_manifest.json"
    )
    scheduler_config = json.loads(
        json.dumps(dict(pipeline.scheduler.config), sort_keys=True)
    )
    default_values = scheduler_config.get("_use_default_values")
    if isinstance(default_values, list):
        scheduler_config["_use_default_values"] = sorted(default_values)
    declaration["scheduler_config"] = scheduler_config
    declaration["runtime"] = environment_manifest(config)
    return declaration


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rectangular result rows."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def cfg_embeddings(pipeline, prompt: str) -> torch.Tensor:
    """Encode one prompt with its empty unconditional prompt."""
    with torch.no_grad():
        positive, negative = pipeline.encode_prompt(
            prompt=prompt,
            device=torch.device("mps"),
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
        )
    return torch.cat([negative, positive]).detach()


def cfg_noise_prediction(
    pipeline,
    latent: torch.Tensor,
    timestep: torch.Tensor,
    embeddings: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Return the classifier-free-guided U-Net prediction."""
    latent_input = torch.cat([latent, latent])
    latent_input = pipeline.scheduler.scale_model_input(latent_input, timestep)
    prediction = pipeline.unet(
        latent_input,
        timestep,
        encoder_hidden_states=embeddings,
        return_dict=False,
    )[0]
    unconditional, conditional = prediction.chunk(2)
    return unconditional + guidance_scale * (conditional - unconditional)


def target_trajectory_state(
    pipeline,
    config: dict,
    latents: torch.Tensor,
    embeddings: torch.Tensor,
    step_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the target trajectory to one declared denoising step."""
    pipeline.scheduler.set_timesteps(config["steps"], device="mps")
    timesteps = pipeline.scheduler.timesteps
    latent = latents.clone() * pipeline.scheduler.init_noise_sigma
    with torch.no_grad():
        for timestep in timesteps[:step_index]:
            prediction = cfg_noise_prediction(
                pipeline,
                latent,
                timestep,
                embeddings,
                float(config["guidance_scale"]),
            )
            latent = pipeline.scheduler.step(
                prediction,
                timestep,
                latent,
                eta=float(config["eta"]),
                return_dict=False,
            )[0]
    return latent.detach(), timesteps[step_index]


def rank_approximation(
    gradient: torch.Tensor,
    rank: int,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray, float, torch.Tensor, torch.Tensor]:
    """Return a deterministic randomized rank approximation."""
    cpu_gradient = gradient.detach().to(device="cpu", dtype=torch.float32)
    torch.manual_seed(seed)
    left, singular, right = torch.svd_lowrank(
        cpu_gradient,
        q=rank,
        niter=8,
    )
    approximation = (left * singular.unsqueeze(0)) @ right.T
    explained_energy = float(
        singular.square().sum() / cpu_gradient.square().sum().clamp_min(1e-30)
    )
    return approximation, singular.numpy(), explained_energy, left, right


def intervention_rows(
    intervention: str,
    strength: float,
    source: Image.Image,
    target: Image.Image,
    changed: Image.Image,
) -> list[dict]:
    """Measure target change and restoration toward the source."""
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
        source_target_rmse = baseline[region]["rmse"]
        rows.append(
            {
                "intervention": intervention,
                "strength": strength,
                "region": region,
                "baseline_source_target_rmse": source_target_rmse,
                "target_change_rmse": target_change[region]["rmse"],
                "target_change_mae": target_change[region]["mae"],
                "source_distance_rmse": source_distance[region]["rmse"],
                "restoration": 1.0
                - source_distance[region]["rmse"] / max(source_target_rmse, 1e-12),
                "changed_pixel_share_1pct": target_change[region][
                    "changed_pixel_share_1pct"
                ],
            }
        )
    return rows


def main() -> None:
    """Run the exact local-gradient and aligned low-rank experiment."""
    config = load_config()
    pipeline = load_pipeline(config)
    latents = initial_latents(pipeline, config)
    output_directory = ARTIFACTS / "gradient_svd_lora"
    output_directory.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        source_image = generate(
            pipeline, config, config["source_prompt"], latents
        )
        target_image = generate(
            pipeline, config, config["target_prompt"], latents
        )
    source_image.save(output_directory / "source.png")
    target_image.save(output_directory / "target.png")

    source_embeddings = cfg_embeddings(pipeline, config["source_prompt"])
    target_embeddings = cfg_embeddings(pipeline, config["target_prompt"])
    step_index = int(config.get("gradient_step_index", config["steps"] // 2))
    state, timestep = target_trajectory_state(
        pipeline,
        config,
        latents,
        target_embeddings,
        step_index,
    )

    selections = selected_value_modules(pipeline.unet)
    for parameter in pipeline.unet.parameters():
        parameter.requires_grad_(False)
    for _name, module in selections:
        module.weight.requires_grad_(True)

    with torch.no_grad():
        source_reference = cfg_noise_prediction(
            pipeline,
            state,
            timestep,
            source_embeddings,
            float(config["guidance_scale"]),
        ).detach()
    target_prediction = cfg_noise_prediction(
        pipeline,
        state,
        timestep,
        target_embeddings,
        float(config["guidance_scale"]),
    )
    objective = 0.5 * torch.mean((target_prediction - source_reference) ** 2)
    gradients = torch.autograd.grad(
        objective,
        [module.weight for _name, module in selections],
    )

    gradient_arrays: dict[str, np.ndarray] = {}
    gradient_summary = []
    approximations: dict[str, torch.Tensor] = {}
    factorisations: dict[str, tuple[np.ndarray, torch.Tensor, torch.Tensor]] = {}
    rank = int(config["lora_rank"])
    svd_seed = int(config["seed"]) + 200
    for index, ((module_name, module), gradient) in enumerate(
        zip(selections, gradients, strict=True)
    ):
        approximation, singular, explained_energy, left, right = rank_approximation(
            gradient,
            rank,
            svd_seed + index,
        )
        approximations[module_name] = approximation
        factorisations[module_name] = (singular, left, right)
        cpu_gradient = gradient.detach().float().cpu()
        gradient_arrays[safe_name(module_name)] = cpu_gradient.numpy()
        gradient_summary.append(
            {
                "module": module_name,
                "parameter_count": module.weight.numel(),
                "gradient_frobenius": float(cpu_gradient.norm()),
                "gradient_rms": float(cpu_gradient.square().mean().sqrt()),
                "gradient_max_abs": float(cpu_gradient.abs().max()),
                "rank": rank,
                "rank_singular_values": singular.tolist(),
                "rank_explained_energy": explained_energy,
            }
        )

    gradient_archive_path = output_directory / "selected_weight_gradients.npz"
    np.savez_compressed(
        gradient_archive_path,
        **gradient_arrays,
    )

    primary_name, primary_module = selections[len(selections) // 2]
    primary_approximation = approximations[primary_name]
    aligned = LowRankDirection.from_delta(
        primary_module,
        -primary_approximation,
        rank=rank,
        relative_norm=float(config["lora_relative_norm"]),
        seed=svd_seed + len(selections) // 2,
    )
    singular, left, right = factorisations[primary_name]
    target_delta_norm = aligned.delta.norm()
    factor_scale = float(
        target_delta_norm
        / primary_approximation.to(dtype=torch.float64).norm()
    )
    left_factor = (
        -factor_scale
        * left.to(dtype=torch.float64)
        * torch.from_numpy(singular).to(dtype=torch.float64).unsqueeze(0)
    )
    right_factor = right.to(dtype=torch.float64).T
    factor_delta = left_factor @ right_factor
    left_factor = left_factor * float(target_delta_norm / factor_delta.norm())
    aligned.delta = left_factor @ right_factor
    delta_path = output_directory / "aligned_rank4_delta.npy"
    np.save(delta_path, aligned.delta.numpy())
    factor_path = output_directory / "aligned_rank4_factors.npz"
    np.savez_compressed(
        factor_path,
        B=left_factor.numpy(),
        A=right_factor.numpy(),
    )
    primary_gradient = gradients[len(selections) // 2].detach().float().cpu()
    local_directional_derivative = float(
        torch.sum(primary_gradient * aligned.delta.float())
    )

    for _name, module in selections:
        module.weight.requires_grad_(False)
    del target_prediction, gradients
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()

    objective_by_strength = {}
    rows: list[dict] = []
    zero_hash = None
    for strength_value in config["gradient_aligned_doses"]:
        strength = float(strength_value)
        with aligned.applied(strength):
            with torch.no_grad():
                changed_prediction = cfg_noise_prediction(
                    pipeline,
                    state,
                    timestep,
                    target_embeddings,
                    float(config["guidance_scale"]),
                )
                local_objective = 0.5 * torch.mean(
                    (changed_prediction - source_reference) ** 2
                )
                image = generate(
                    pipeline, config, config["target_prompt"], latents
                )
        objective_by_strength[f"{strength:+.2f}"] = float(local_objective)
        stem = f"dose_{strength:+.2f}".replace("+", "p").replace("-", "m")
        image.save(output_directory / f"aligned_{stem}.png")
        difference_heatmap(target_image, image).save(
            output_directory / f"aligned_{stem}_difference.png"
        )
        rows.extend(
            intervention_rows(
                "gradient_aligned_rank4",
                strength,
                source_image,
                target_image,
                image,
            )
        )
        if strength == 0.0:
            zero_hash = image_sha256(image)

    target_hash = image_sha256(target_image)
    result = {
        "source_sha256": image_sha256(source_image),
        "target_sha256": target_hash,
        "zero_dose_sha256": zero_hash,
        "zero_dose_equal": zero_hash == target_hash,
        "gradient_objective": "Half mean squared guided-noise difference between target and source prompts.",
        "gradient_step_index": step_index,
        "scheduler_timestep": int(timestep),
        "baseline_objective": float(objective.detach()),
        "selected_weight_gradients": gradient_summary,
        "primary_lora_module": primary_name,
        "rank": rank,
        "relative_norm": aligned.relative_norm,
        "local_directional_derivative": local_directional_derivative,
        "objective_by_strength": objective_by_strength,
        "direction_provenance": {
            "declaration": direction_declaration(config, pipeline, primary_name),
            "scheduler_timestep": int(timestep),
            "gradient_archive_sha256": file_sha256(gradient_archive_path),
            "delta_sha256": file_sha256(delta_path),
            "factor_sha256": file_sha256(factor_path),
        },
        "interpretation": {
            "gradient": "Exact local sensitivity of the declared one-step denoiser objective.",
            "rank4": "A rank-4 gradient descent direction, rescaled to one percent of the base weight norm.",
            "image": "A full-generation causal test of that local direction under fixed inputs.",
        },
    }
    write_csv(output_directory / "gradient_aligned_lora_metrics.csv", rows)
    (output_directory / "gradient_aligned_lora_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
