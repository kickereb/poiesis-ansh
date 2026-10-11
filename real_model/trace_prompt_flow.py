#!/usr/bin/env python3
"""Capture a compact prompt-memory and latent-flow trace for the visualiser.

This is deliberately a summary trace, not a full activation dump. It records
the 77 CLIP token slots, token hidden-state norms, the source/target embedding
delta, and one matched latent trajectory for each prompt. The raw tensors are
not written, which keeps the artifact small and avoids implying that one
scalar norm is a complete explanation of an image.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model_utils import ARTIFACTS, initial_latents, load_config, load_pipeline


def _prompt_memory_trace(pipeline: Any, prompt: str) -> dict[str, Any]:
    """Return token IDs, decoded tokens, and per-token hidden-state norms."""
    tokenizer = pipeline.tokenizer
    encoded = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids.to(device="mps")
    with torch.inference_mode():
        hidden = pipeline.text_encoder(input_ids)[0][0].float().cpu().numpy()
    ids = encoded.input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    norms = np.linalg.norm(hidden, axis=1)
    return {
        "prompt": prompt,
        "token_ids": ids,
        "tokens": tokens,
        "hidden_width": int(hidden.shape[1]),
        "hidden_state_norms": [float(x) for x in norms],
    }


def _run_trace(
    pipeline: Any,
    config: dict[str, Any],
    prompt: str,
    latents: torch.Tensor,
) -> dict[str, Any]:
    """Run one prompt and record the matched latent and mid-block flow."""
    trace: list[dict[str, Any]] = []
    activation_calls: list[dict[str, float]] = []

    def activation_hook(_module: Any, _inputs: Any, output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor):
            return
        value = tensor.detach().float()
        # CFG normally gives a two-item batch (unconditional, conditional).
        cond_gap = 0.0
        if value.ndim >= 1 and value.shape[0] >= 2 and value.shape[0] % 2 == 0:
            half = value.shape[0] // 2
            cond_gap = float((value[half:] - value[:half]).pow(2).mean().sqrt().cpu())
        activation_calls.append(
            {
                "rms": float(value.pow(2).mean().sqrt().cpu()),
                "mean": float(value.mean().cpu()),
                "conditional_unconditional_gap": cond_gap,
            }
        )

    module = pipeline.unet.mid_block.attentions[0].transformer_blocks[0]
    handle = module.register_forward_hook(activation_hook)

    def callback(_pipe: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        current = callback_kwargs["latents"].detach().float()
        flat = current.reshape(-1)
        midpoint = max(1, flat.numel() // 2)
        # Two fixed projections make a stable 2-D path for the visualizer.
        x = float(flat[:midpoint].mean().cpu())
        y = float(flat[midpoint:].mean().cpu())
        trace.append(
            {
                "step_index": int(step_index),
                "scheduler_timestep": int(timestep),
                "latent_rms": float(current.pow(2).mean().sqrt().cpu()),
                "latent_mean": float(current.mean().cpu()),
                "latent_projection_x": x,
                "latent_projection_y": y,
            }
        )
        return callback_kwargs

    try:
        with torch.inference_mode():
            pipeline(
                prompt=prompt,
                negative_prompt="",
                height=config["height"],
                width=config["width"],
                num_inference_steps=config["steps"],
                guidance_scale=config["guidance_scale"],
                eta=config["eta"],
                latents=latents.clone(),
                output_type="latent",
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )
    finally:
        handle.remove()

    # A normal pipeline call gives one activation record per denoising step.
    # Keep the alignment explicit if a future diffusers version calls a hook
    # more than once per step.
    for index, row in enumerate(trace):
        if index < len(activation_calls):
            row["mid_block_activation_rms"] = activation_calls[index]["rms"]
            row["mid_block_activation_mean"] = activation_calls[index]["mean"]
            row["mid_block_conditional_unconditional_gap"] = activation_calls[index][
                "conditional_unconditional_gap"
            ]
        else:
            row["mid_block_activation_rms"] = None
            row["mid_block_activation_mean"] = None
            row["mid_block_conditional_unconditional_gap"] = None
    return {"prompt": prompt, "steps": trace, "activation_call_count": len(activation_calls)}


def main() -> None:
    config = load_config()
    pipeline = load_pipeline(config)
    latents = initial_latents(pipeline, config)
    source_prompt = config["source_prompt"]
    target_prompt = config["target_prompt"]

    source_memory = _prompt_memory_trace(pipeline, source_prompt)
    target_memory = _prompt_memory_trace(pipeline, target_prompt)
    source_flow = _run_trace(pipeline, config, source_prompt, latents)
    target_flow = _run_trace(pipeline, config, target_prompt, latents)

    source_norms = np.asarray(source_memory["hidden_state_norms"], dtype=float)
    target_norms = np.asarray(target_memory["hidden_state_norms"], dtype=float)
    delta_norms = np.abs(target_norms - source_norms)
    source_steps = source_flow["steps"]
    target_steps = target_flow["steps"]
    for source, target in zip(source_steps, target_steps):
        source["target_latent_rms_delta"] = float(
            target["latent_rms"] - source["latent_rms"]
        )
        source["target_latent_projection_distance"] = float(
            (
                (target["latent_projection_x"] - source["latent_projection_x"]) ** 2
                + (target["latent_projection_y"] - source["latent_projection_y"]) ** 2
            )
            ** 0.5
        )

    payload = {
        "trace_version": 1,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "scheduler": config["scheduler"],
        "seed": config["seed"],
        "steps": config["steps"],
        "height": config["height"],
        "width": config["width"],
        "guidance_scale": config["guidance_scale"],
        "prompt_memory": {
            "source": source_memory,
            "target": target_memory,
            "hidden_state_norm_delta": [float(x) for x in delta_norms],
            "changed_token_indices": [
                int(index)
                for index, value in enumerate(delta_norms)
                if value > 1e-6
            ],
        },
        "latent_flow": {"source": source_flow, "target": target_flow},
        "interpretation": {
            "prompt_location": "The prompt is represented across 77 CLIP token hidden-state slots; it is not stored in one address.",
            "latent_projection": "Two fixed half-vector means are a visualization projection, not a semantic coordinate system.",
            "activation": "The mid-block hook measures one internal vector field summary per denoising step; it is not a full activation tensor.",
        },
    }
    output = ARTIFACTS / "prompt_flow_trace.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "tokens": len(source_memory["tokens"]), "steps": len(source_flow["steps"])}, indent=2))


if __name__ == "__main__":
    main()
