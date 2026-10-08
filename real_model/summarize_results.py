#!/usr/bin/env python3
"""Build one machine-readable summary from completed pilot artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from model_utils import ARTIFACTS


def read_json(path: Path) -> dict:
    """Read one JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows."""
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_name(module_name: str) -> str:
    """Convert a module path to its artifact stem."""
    return module_name.replace(".", "_")


def regional_gradient_rms(gradient: np.ndarray) -> dict[str, float]:
    """Measure image-gradient RMS in fixed horizontal regions."""
    width = gradient.shape[1]
    regions = {
        "full": slice(0, width),
        "left": slice(0, width // 3),
        "centre": slice(width // 3, 2 * width // 3),
        "right": slice(2 * width // 3, width),
    }
    return {
        name: float(np.sqrt(np.mean(gradient[:, region] ** 2)))
        for name, region in regions.items()
    }


def numeric_rows(rows: list[dict[str, str]]) -> list[dict]:
    """Convert numeric CSV values while retaining labels."""
    converted = []
    for row in rows:
        output = {}
        for key, value in row.items():
            try:
                output[key] = float(value)
            except ValueError:
                output[key] = value
        converted.append(output)
    return converted


def main() -> None:
    """Aggregate baseline, gradient, LoRA, and patching outputs."""
    baseline = read_json(ARTIFACTS / "baseline_results.json")
    patch_rows = numeric_rows(
        read_csv(ARTIFACTS / "patching" / "activation_patching_metrics.csv")
    )
    patch_result = read_json(
        ARTIFACTS / "patching" / "activation_patching_results.json"
    )
    weight_result = read_json(
        ARTIFACTS / "weights" / "weight_intervention_results.json"
    )
    weight_rows = numeric_rows(
        read_csv(ARTIFACTS / "weights" / "weight_intervention_metrics.csv")
    )
    aligned_result = read_json(
        ARTIFACTS
        / "gradient_svd_lora"
        / "gradient_aligned_lora_results.json"
    )
    aligned_rows = numeric_rows(
        read_csv(
            ARTIFACTS
            / "gradient_svd_lora"
            / "gradient_aligned_lora_metrics.csv"
        )
    )
    output_gradient_result = read_json(
        ARTIFACTS
        / "output_weight_gradients"
        / "output_weight_gradient_results.json"
    )
    trained_lora_result = read_json(
        ARTIFACTS / "trained_lora" / "trained_lora_results.json"
    )
    trained_lora_rows = numeric_rows(
        read_csv(ARTIFACTS / "trained_lora" / "trained_lora_metrics.csv")
    )
    trained_lora_dose_rows = numeric_rows(
        read_csv(
            ARTIFACTS / "trained_lora" / "trained_lora_dose_metrics.csv"
        )
    )
    seed_result = read_json(
        ARTIFACTS / "seed_robustness" / "seed_robustness_results.json"
    )
    seed_rows = numeric_rows(
        read_csv(
            ARTIFACTS / "seed_robustness" / "seed_robustness_metrics.csv"
        )
    )
    seed_objective_rows = numeric_rows(
        read_csv(
            ARTIFACTS / "seed_robustness" / "seed_objective_metrics.csv"
        )
    )

    gradient_regions = []
    for module_name in weight_result["selected_modules"]:
        path = (
            ARTIFACTS
            / "weights"
            / f"{safe_name(module_name)}_directional_gradient.npy"
        )
        gradient_regions.append(
            {
                "module": module_name,
                "regional_rms": regional_gradient_rms(np.load(path)),
            }
        )

    summary = {
        "scope": "One-model, one-prompt-pair, one-seed causal pilot.",
        "baseline": baseline,
        "activation_patching_full_region": [
            row for row in patch_rows if row["region"] == "full"
        ],
        "activation_patching_scope": {
            "module": patch_result["module"],
            "patch_scope": patch_result["patch_scope"],
            "equal_window_lengths": patch_result["equal_window_lengths"],
        },
        "random_low_rank_full_region": [
            row
            for row in weight_rows
            if row["intervention"] == "lora_dose" and row["region"] == "full"
        ],
        "directional_output_gradient_regions": gradient_regions,
        "finite_difference_check": weight_result["finite_difference_check"],
        "selected_weight_gradients": aligned_result[
            "selected_weight_gradients"
        ],
        "gradient_aligned_full_region": [
            row for row in aligned_rows if row["region"] == "full"
        ],
        "gradient_aligned_objective": aligned_result[
            "objective_by_strength"
        ],
        "final_image_weight_gradient": output_gradient_result,
        "trained_lora": trained_lora_result,
        "trained_lora_full_region": [
            row for row in trained_lora_rows if row["region"] == "full"
        ],
        "trained_lora_dose_full_region": [
            row
            for row in trained_lora_dose_rows
            if row["region"] == "full"
        ],
        "seed_robustness_summary": seed_result["summary"],
        "held_out_seed_robustness_summary": seed_result[
            "held_out_summary"
        ],
        "seed_robustness_full_region": [
            row for row in seed_rows if row["region"] == "full"
        ],
        "seed_objective_summary": seed_result["objective_summary"],
        "held_out_seed_objective_summary": seed_result[
            "held_out_objective_summary"
        ],
        "seed_objective_rows": seed_objective_rows,
        "controls": {
            "activation_identity_equal": (
                patch_result["identity_control_sha256"]
                == baseline["target_sha256"]
            ),
            "random_lora_zero_equal": weight_result["zero_dose_equal"],
            "gradient_lora_zero_equal": aligned_result["zero_dose_equal"],
            "direction_provenance_validated": seed_result[
                "direction_provenance_validated"
            ],
        },
    }
    output = ARTIFACTS / "influence_summary.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
