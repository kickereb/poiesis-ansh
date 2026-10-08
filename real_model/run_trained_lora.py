#!/usr/bin/env python3
"""Train and evaluate one controlled rank-4 LoRA adapter."""

from __future__ import annotations

import csv
import json

import numpy as np
import torch

from model_utils import (
    ARTIFACTS,
    file_sha256,
    generate,
    image_sha256,
    initial_latents,
    load_config,
    load_pipeline,
)
from run_gradient_svd_lora import (
    cfg_embeddings,
    cfg_noise_prediction,
    direction_declaration,
    target_trajectory_state,
)
from run_seed_robustness import aggregate, intervention_rows
from weight_tools import (
    LowRankDirection,
    TrainableLoRAAdapter,
    selected_value_modules,
)


def write_csv(path, rows: list[dict]) -> None:
    """Write one rectangular table."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Fit a LoRA adapter and test it against a random control."""
    config = load_config()
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

    selections = selected_value_modules(pipeline.unet)
    primary_name, primary_module = selections[len(selections) // 2]
    source_embeddings = cfg_embeddings(pipeline, config["source_prompt"])
    target_embeddings = cfg_embeddings(pipeline, config["target_prompt"])
    fitting_latents = initial_latents(pipeline, config)
    training_examples = []
    for step_index in config["trained_lora_step_indices"]:
        state, timestep = target_trajectory_state(
            pipeline,
            config,
            fitting_latents,
            target_embeddings,
            int(step_index),
        )
        with torch.no_grad():
            source_reference = cfg_noise_prediction(
                pipeline,
                state,
                timestep,
                source_embeddings,
                float(config["guidance_scale"]),
            ).detach()
            base_prediction = cfg_noise_prediction(
                pipeline,
                state,
                timestep,
                target_embeddings,
                float(config["guidance_scale"]),
            )
            base_loss = 0.5 * torch.mean(
                (base_prediction - source_reference) ** 2
            )
        training_examples.append(
            {
                "step_index": int(step_index),
                "timestep": int(timestep),
                "state": state,
                "source_reference": source_reference,
                "base_loss": float(base_loss),
            }
        )

    adapter = TrainableLoRAAdapter(
        primary_module,
        rank=int(config["lora_rank"]),
        seed=int(config["seed"]) + 700,
    )
    optimizer = torch.optim.Adam(
        adapter.parameters(),
        lr=float(config["trained_lora_learning_rate"]),
    )
    training_rows = []
    with adapter:
        for iteration in range(int(config["trained_lora_training_steps"])):
            example = training_examples[iteration % len(training_examples)]
            optimizer.zero_grad(set_to_none=True)
            prediction = cfg_noise_prediction(
                pipeline,
                example["state"],
                torch.tensor(example["timestep"], device="mps"),
                target_embeddings,
                float(config["guidance_scale"]),
            )
            loss = 0.5 * torch.mean(
                (prediction - example["source_reference"]) ** 2
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer.step()
            training_rows.append(
                {
                    "iteration": iteration,
                    "step_index": example["step_index"],
                    "scheduler_timestep": example["timestep"],
                    "loss": float(loss.detach().cpu()),
                }
            )

    raw_delta = adapter.delta()
    base_weight_norm = float(primary_module.weight.detach().float().norm().cpu())
    raw_relative_norm = float(raw_delta.norm()) / base_weight_norm
    target_relative_norm = float(config["trained_lora_relative_norm"])
    factor_scale = target_relative_norm / raw_relative_norm
    trained_B = adapter.B.detach().to(device="cpu").to(torch.float64)
    trained_A = adapter.A.detach().to(device="cpu").to(torch.float64)
    trained_B = trained_B * factor_scale
    trained_delta = trained_B @ trained_A
    trained_direction = LowRankDirection(
        module=primary_module,
        delta=trained_delta,
        rank=int(config["lora_rank"]),
        relative_norm=float(trained_delta.norm()) / base_weight_norm,
        seed=int(config["seed"]) + 700,
    )
    random_direction = LowRankDirection.random(
        primary_module,
        rank=int(config["lora_rank"]),
        relative_norm=trained_direction.relative_norm,
        seed=int(config["seed"]) + 701,
    )

    output_directory = ARTIFACTS / "trained_lora"
    output_directory.mkdir(parents=True, exist_ok=True)
    factor_path = output_directory / "trained_rank4_factors.npz"
    np.savez_compressed(
        factor_path,
        B=trained_B.numpy(),
        A=trained_A.numpy(),
    )
    delta_path = output_directory / "trained_rank4_delta.npy"
    np.save(delta_path, trained_delta.numpy())

    training_objectives = []
    for example in training_examples:
        row = {
            "step_index": example["step_index"],
            "scheduler_timestep": example["timestep"],
            "base_loss": example["base_loss"],
        }
        for dose in (0.25, 0.5, 1.0):
            with torch.no_grad(), trained_direction.applied(dose):
                prediction = cfg_noise_prediction(
                    pipeline,
                    example["state"],
                    torch.tensor(example["timestep"], device="mps"),
                    target_embeddings,
                    float(config["guidance_scale"]),
                )
                loss = 0.5 * torch.mean(
                    (prediction - example["source_reference"]) ** 2
                )
            row[f"loss_dose_{dose:.2f}"] = float(loss.cpu())
            row[f"reduction_dose_{dose:.2f}"] = (
                1.0 - float(loss.cpu()) / example["base_loss"]
            )
        training_objectives.append(row)

    rows = []
    dose_rows = []
    image_hashes = {}
    zero_dose_hash = None
    fitting_seed = int(config["seed"])
    for seed_value in config["robustness_seeds"]:
        seed = int(seed_value)
        seed_directory = output_directory / str(seed)
        seed_directory.mkdir(parents=True, exist_ok=True)
        latents = initial_latents(pipeline, config, seed=seed)
        with torch.inference_mode():
            source = generate(
                pipeline, config, config["source_prompt"], latents
            )
            target = generate(
                pipeline, config, config["target_prompt"], latents
            )
            with trained_direction.applied(0.5):
                trained = generate(
                    pipeline, config, config["target_prompt"], latents
                )
            with random_direction.applied(0.5):
                random = generate(
                    pipeline, config, config["target_prompt"], latents
                )
        images = {
            "source": source,
            "target": target,
            "trained_lora_dose_0.5": trained,
            "random_lora_dose_0.5": random,
        }
        image_hashes[str(seed)] = {
            name: image_sha256(image) for name, image in images.items()
        }
        for name, image in images.items():
            image.save(seed_directory / f"{name}.png")
        rows.extend(
            intervention_rows(
                seed,
                "trained_rank4_dose_0.5",
                source,
                target,
                trained,
            )
        )
        rows.extend(
            intervention_rows(
                seed,
                "random_rank4_dose_0.5",
                source,
                target,
                random,
            )
        )

        if seed == fitting_seed:
            for dose_value in config["trained_lora_image_doses"]:
                dose = float(dose_value)
                if dose == 0.5:
                    image = trained
                else:
                    with torch.inference_mode(), trained_direction.applied(dose):
                        image = generate(
                            pipeline,
                            config,
                            config["target_prompt"],
                            latents,
                        )
                stem = f"{dose:+.2f}".replace("+", "p").replace("-", "m")
                image.save(output_directory / f"trained_dose_{stem}.png")
                if dose == 0.0:
                    zero_dose_hash = image_sha256(image)
                dose_rows.extend(
                    intervention_rows(
                        seed,
                        "trained_rank4_dose_sweep",
                        source,
                        target,
                        image,
                    )
                )
                for row in dose_rows[-4:]:
                    row["dose"] = dose

    held_out_rows = [row for row in rows if row["seed"] != fitting_seed]
    result = {
        "module": primary_name,
        "rank": int(config["lora_rank"]),
        "training_seed": fitting_seed,
        "training_step_indices": config["trained_lora_step_indices"],
        "training_iterations": int(config["trained_lora_training_steps"]),
        "learning_rate": float(config["trained_lora_learning_rate"]),
        "training_objective": (
            "Half mean squared guided-noise difference between target and "
            "source prompts."
        ),
        "raw_relative_norm": raw_relative_norm,
        "controlled_relative_norm": trained_direction.relative_norm,
        "direction_delta_sha256": file_sha256(delta_path),
        "direction_factor_sha256": file_sha256(factor_path),
        "direction_declaration": direction_declaration(
            config, pipeline, primary_name
        ),
        "training_objectives": training_objectives,
        "image_hashes": image_hashes,
        "all_seed_summary": aggregate(rows),
        "held_out_summary": aggregate(held_out_rows),
        "zero_dose_equal": zero_dose_hash
        == image_hashes[str(fitting_seed)]["target"],
        "scope": "One prompt pair, one fitting seed, and three held-out seeds.",
    }
    write_csv(output_directory / "training_curve.csv", training_rows)
    write_csv(output_directory / "training_objectives.csv", training_objectives)
    write_csv(output_directory / "trained_lora_metrics.csv", rows)
    write_csv(output_directory / "trained_lora_dose_metrics.csv", dose_rows)
    (output_directory / "trained_lora_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
