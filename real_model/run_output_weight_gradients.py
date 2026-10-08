#!/usr/bin/env python3
"""Differentiate a final float-image objective with respect to selected weights."""

from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

from model_utils import (
    ARTIFACTS,
    MANIFESTS,
    file_sha256,
    image_sha256,
    initial_latents,
    load_config,
    load_pipeline,
)
from run_gradient_svd_lora import (
    cfg_embeddings,
    cfg_noise_prediction,
    direction_declaration,
    rank_approximation,
    safe_name,
)
from weight_tools import LowRankDirection, selected_value_modules


def final_image_tensor(
    pipeline,
    config: dict,
    embeddings: torch.Tensor,
    latents: torch.Tensor,
) -> torch.Tensor:
    """Return the final decoded RGB tensor without disabling gradients."""
    pipeline.scheduler.set_timesteps(config["steps"], device="mps")
    latent = latents.clone() * pipeline.scheduler.init_noise_sigma
    for timestep in pipeline.scheduler.timesteps:
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
    decoded = pipeline.vae.decode(
        latent / pipeline.vae.config.scaling_factor,
        return_dict=False,
    )[0]
    return (decoded / 2.0 + 0.5).clamp(0.0, 1.0)


def tensor_image(value: torch.Tensor) -> Image.Image:
    """Convert one decoded image tensor to an RGB image."""
    array = value[0].detach().float().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.uint8(np.round(255.0 * array)), mode="RGB")


def objective_value(
    pipeline,
    config: dict,
    embeddings: torch.Tensor,
    latents: torch.Tensor,
    source_reference: torch.Tensor,
) -> float:
    """Evaluate the declared final-image objective without a graph."""
    with torch.no_grad():
        image = final_image_tensor(pipeline, config, embeddings, latents)
        objective = 0.5 * torch.mean((image - source_reference) ** 2)
    return float(objective.cpu())


def main() -> None:
    """Run one exact scalar-output vector-Jacobian product."""
    base_config = load_config()
    config = {
        **base_config,
        "width": int(base_config["output_gradient_width"]),
        "height": int(base_config["output_gradient_height"]),
        "steps": int(base_config["output_gradient_steps"]),
    }
    pipeline = load_pipeline(config)
    pipeline.unet.eval()
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    for component in (
        pipeline.unet,
        pipeline.vae,
        pipeline.text_encoder,
    ):
        component.requires_grad_(False)

    latents = initial_latents(pipeline, config)
    source_embeddings = cfg_embeddings(pipeline, config["source_prompt"])
    target_embeddings = cfg_embeddings(pipeline, config["target_prompt"])
    output_directory = ARTIFACTS / "output_weight_gradients"
    output_directory.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        source_reference = final_image_tensor(
            pipeline, config, source_embeddings, latents
        ).detach()
    source_image = tensor_image(source_reference)
    source_image.save(output_directory / "source.png")

    selections = selected_value_modules(pipeline.unet)
    for _name, module in selections:
        module.weight.requires_grad_(True)
    target_image = final_image_tensor(
        pipeline, config, target_embeddings, latents
    )
    objective = 0.5 * torch.mean((target_image - source_reference) ** 2)
    gradients = torch.autograd.grad(
        objective,
        [module.weight for _name, module in selections],
    )
    target_pil = tensor_image(target_image)
    target_pil.save(output_directory / "target.png")

    rank = int(config["lora_rank"])
    summaries = []
    archives = {}
    approximations = {}
    for index, ((module_name, module), gradient) in enumerate(
        zip(selections, gradients, strict=True)
    ):
        cpu_gradient = gradient.detach().float().cpu()
        approximation, singular, energy, _left, _right = rank_approximation(
            cpu_gradient,
            rank,
            int(config["seed"]) + 500 + index,
        )
        archives[safe_name(module_name)] = cpu_gradient.numpy()
        approximations[module_name] = approximation
        summaries.append(
            {
                "module": module_name,
                "parameter_count": module.weight.numel(),
                "gradient_frobenius": float(cpu_gradient.norm()),
                "gradient_rms": float(cpu_gradient.square().mean().sqrt()),
                "gradient_max_abs": float(cpu_gradient.abs().max()),
                "rank": rank,
                "rank_singular_values": singular.tolist(),
                "rank_explained_energy": energy,
            }
        )

    gradient_path = output_directory / "final_image_weight_gradients.npz"
    np.savez_compressed(gradient_path, **archives)
    primary_index = len(selections) // 2
    primary_name, primary_module = selections[primary_index]
    direction = LowRankDirection.from_delta(
        primary_module,
        -approximations[primary_name],
        rank=rank,
        relative_norm=float(config["output_gradient_relative_norm"]),
        seed=int(config["seed"]) + 600,
    )
    primary_gradient = gradients[primary_index].detach().float().cpu()
    predicted_derivative = float(
        torch.sum(primary_gradient * direction.delta.float())
    )
    direction_path = output_directory / "validation_rank4_delta.npy"
    np.save(direction_path, direction.delta.numpy())
    for _name, module in selections:
        module.weight.requires_grad_(False)
    del target_image, gradients
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()

    epsilon = float(config["output_gradient_validation_epsilon"])
    with direction.applied(-epsilon):
        negative_objective = objective_value(
            pipeline,
            config,
            target_embeddings,
            latents,
            source_reference,
        )
    with direction.applied(epsilon):
        positive_objective = objective_value(
            pipeline,
            config,
            target_embeddings,
            latents,
            source_reference,
        )
    measured_derivative = (
        positive_objective - negative_objective
    ) / (2.0 * epsilon)

    result = {
        "objective": (
            "Half mean squared final float-RGB difference between target and "
            "source prompt outputs."
        ),
        "objective_value": float(objective.detach().cpu()),
        "width": config["width"],
        "height": config["height"],
        "steps": config["steps"],
        "seed": int(config["seed"]),
        "source_sha256": image_sha256(source_image),
        "target_sha256": image_sha256(target_pil),
        "selected_weight_gradients": summaries,
        "gradient_archive_sha256": file_sha256(gradient_path),
        "direction_sha256": file_sha256(direction_path),
        "model_manifest_sha256": file_sha256(
            MANIFESTS / "model_manifest.json"
        ),
        "declaration": direction_declaration(
            config, pipeline, primary_name
        ),
        "validation": {
            "module": primary_name,
            "rank": rank,
            "relative_norm": direction.relative_norm,
            "epsilon": epsilon,
            "negative_objective": negative_objective,
            "positive_objective": positive_objective,
            "autograd_directional_derivative": predicted_derivative,
            "central_difference_derivative": measured_derivative,
            "relative_error": abs(
                measured_derivative - predicted_derivative
            )
            / max(abs(predicted_derivative), 1e-30),
        },
        "interpretation": (
            "Exact weight gradient of one final float-image scalar objective. "
            "It is not the full image Jacobian."
        ),
    }
    (output_directory / "output_weight_gradient_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
