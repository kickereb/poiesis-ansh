#!/usr/bin/env python3
"""Run robustness checks for the image attribution experiment."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

import experiment as core


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ROBUSTNESS_SEED = 20261005


def write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summary(
    source_codes: Sequence[np.ndarray], x_code: np.ndarray, weights: np.ndarray
) -> Dict[str, float]:
    game = core.information_game(source_codes, x_code, weights)
    values = core.shapley_values(game)
    entropy = core.weighted_entropy(x_code, weights)
    return {
        "output_entropy_bits": entropy,
        "prompt_share": float(values[0] / entropy),
        "model_share": float(values[1] / entropy),
        "seed_share": float(values[2] / entropy),
        "efficiency_error_bits": abs(float(values.sum()) - entropy),
    }


def mutual_information(
    source_code: np.ndarray, x_code: np.ndarray, weights: np.ndarray
) -> float:
    source_entropy = core.weighted_entropy(source_code, weights)
    output_entropy = core.weighted_entropy(x_code, weights)
    output_card = int(x_code.max()) + 1
    joint_code = source_code * output_card + x_code
    joint_entropy = core.weighted_entropy(joint_code, weights)
    return source_entropy + output_entropy - joint_entropy


def representation_checks(
    sources: Sequence[np.ndarray], x_code: np.ndarray, weights: np.ndarray
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    representations = {
        "full_semantics_and_background": x_code,
        "semantic_attributes_only": x_code // 4,
        "background_only": x_code % 4,
    }
    for name, target in representations.items():
        row: Dict[str, object] = {"representation": name}
        row.update(summary(sources, target, weights))
        rows.append(row)
    return rows


def denominator_checks(
    sources: Sequence[np.ndarray], x_code: np.ndarray, weights: np.ndarray
) -> List[Dict[str, object]]:
    base = summary(sources, x_code, weights)
    system_code = core.coalition_code((1, 2), sources)
    entropy = core.weighted_entropy(x_code, weights)
    prompt_mi = mutual_information(sources[0], x_code, weights)
    system_mi = mutual_information(system_code, x_code, weights)
    grouped_prompt = 0.5 * (prompt_mi + entropy - system_mi) / entropy
    prompt_model_total = base["prompt_share"] + base["model_share"]
    return [
        {
            "claim": "three_player_total",
            "prompt_share": base["prompt_share"],
            "other_share": 1.0 - base["prompt_share"],
            "other_definition": "model plus seed, separate players",
        },
        {
            "claim": "prompt_among_prompt_and_model",
            "prompt_share": base["prompt_share"] / prompt_model_total,
            "other_share": base["model_share"] / prompt_model_total,
            "other_definition": "model only; seed omitted",
        },
        {
            "claim": "prompt_vs_grouped_system",
            "prompt_share": grouped_prompt,
            "other_share": 1.0 - grouped_prompt,
            "other_definition": "model and seed as one system player",
        },
    ]


def prior_checks(
    learned_probs: np.ndarray, energy: np.ndarray
) -> List[Dict[str, object]]:
    conditions = {
        "concentrated": np.r_[0.9, np.full(80, 0.1 / 80)],
        "learned_correlated": learned_probs,
        "uniform": np.full(81, 1.0 / 81),
    }
    rows: List[Dict[str, object]] = []
    for name, probs in conditions.items():
        result = core.analyse_level(2, probs, core.BASE_GUIDANCE, energy)
        rows.append(
            {
                "prior": name,
                "prior_entropy_bits": float(-np.sum(probs * np.log2(probs))),
                "output_entropy_bits": result["output_entropy_bits"],
                "prompt_share": result["shapley_prompt_share"],
                "model_share": result["shapley_model_share"],
                "seed_share": result["shapley_seed_share"],
            }
        )
    return rows


def aligned_prompt_weights(
    prompts: np.ndarray,
    prompt_codes: np.ndarray,
    base_weights: np.ndarray,
    model_probs: np.ndarray,
) -> np.ndarray:
    prompt_probs = []
    for prompt in prompts:
        mask = prompt >= 0
        match = np.all(core.ALL_TUPLES[:, mask] == prompt[mask], axis=1)
        prompt_probs.append(float(model_probs[match].sum()))
    prompt_probs = np.asarray(prompt_probs, dtype=float)
    prompt_probs /= prompt_probs.sum()
    uniform_probability = 1.0 / len(prompts)
    probability_by_code = {
        int(core.encode_prompt(prompt.reshape(1, -1))[0]): probability
        for prompt, probability in zip(prompts, prompt_probs)
    }
    factors = np.asarray(
        [probability_by_code[int(code)] / uniform_probability for code in prompt_codes],
        dtype=float,
    )
    weights = base_weights * factors
    return weights / weights.sum()


def prompt_population_checks(
    sources: Sequence[np.ndarray],
    x_code: np.ndarray,
    weights: np.ndarray,
    prompts: np.ndarray,
    model_probs: np.ndarray,
) -> List[Dict[str, object]]:
    aligned = aligned_prompt_weights(
        prompts, sources[0], weights, model_probs
    )
    rows: List[Dict[str, object]] = []
    for name, condition_weights in (
        ("uniform_prompt_values", weights),
        ("prior_aligned_prompt_values", aligned),
    ):
        row: Dict[str, object] = {"prompt_population": name}
        row.update(summary(sources, x_code, condition_weights))
        rows.append(row)
    return rows


def finite_sample_checks(
    rng: np.random.Generator,
    sources: Sequence[np.ndarray],
    x_code: np.ndarray,
    weights: np.ndarray,
) -> List[Dict[str, object]]:
    exact = summary(sources, x_code, weights)
    conditions = ((128, 100), (512, 100), (2048, 100), (8192, 60), (32768, 30), (131072, 10))
    rows: List[Dict[str, object]] = []
    for sample_size, repetitions in conditions:
        estimates = []
        for _ in range(repetitions):
            index = rng.choice(len(weights), size=sample_size, p=weights)
            sample_weights = np.full(sample_size, 1.0 / sample_size)
            game = core.information_game(
                [source[index] for source in sources], x_code[index], sample_weights
            )
            values = core.shapley_values(game)
            entropy = core.weighted_entropy(x_code[index], sample_weights)
            estimates.append(values / entropy)
        array = np.asarray(estimates)
        mean = array.mean(axis=0)
        lower = np.quantile(array, 0.025, axis=0)
        upper = np.quantile(array, 0.975, axis=0)
        exact_array = np.asarray(
            [exact["prompt_share"], exact["model_share"], exact["seed_share"]]
        )
        rows.append(
            {
                "sample_size": sample_size,
                "repetitions": repetitions,
                "prompt_mean": float(mean[0]),
                "prompt_p025": float(lower[0]),
                "prompt_p975": float(upper[0]),
                "model_mean": float(mean[1]),
                "seed_mean": float(mean[2]),
                "mean_absolute_error_all_players": float(
                    np.mean(np.abs(array - exact_array))
                ),
            }
        )
    return rows


def attention_reparameterisation_check() -> List[Dict[str, float]]:
    rows = []
    for value_scale in (1.0, 2.0, 4.0):
        attention_weight = 0.8 / value_scale
        rows.append(
            {
                "value_scale": value_scale,
                "attention_weight": attention_weight,
                "attention_output": attention_weight * value_scale,
            }
        )
    return rows


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(ROBUSTNESS_SEED)
    training_rng = np.random.default_rng(core.SEED)
    _, model_probs = core.learn_model_prior(training_rng)
    energy = core.renderer_feature_energy()
    sources, x_code, weights, prompts = core.enumerate_experiment(
        2, model_probs, core.BASE_GUIDANCE
    )

    results = {
        "experiment": {
            "specificity": 2,
            "robustness_seed": ROBUSTNESS_SEED,
            "exact_state_count": int(len(weights)),
        },
        "representation": representation_checks(sources, x_code, weights),
        "denominator": denominator_checks(sources, x_code, weights),
        "model_prior": prior_checks(model_probs, energy),
        "prompt_population": prompt_population_checks(
            sources, x_code, weights, prompts, model_probs
        ),
        "finite_sample": finite_sample_checks(rng, sources, x_code, weights),
        "attention_reparameterisation": attention_reparameterisation_check(),
    }

    json_path = ARTIFACTS / "robustness_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_csv(ARTIFACTS / "representation_sensitivity.csv", results["representation"])
    write_csv(ARTIFACTS / "denominator_sensitivity.csv", results["denominator"])
    write_csv(ARTIFACTS / "prior_sensitivity.csv", results["model_prior"])
    write_csv(ARTIFACTS / "prompt_population_sensitivity.csv", results["prompt_population"])
    write_csv(ARTIFACTS / "finite_sample_bias.csv", results["finite_sample"])
    write_csv(
        ARTIFACTS / "attention_reparameterisation.csv",
        results["attention_reparameterisation"],
    )
    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    (ARTIFACTS / "robustness_results.sha256").write_text(
        f"{digest}  robustness_results.json\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    print(f"\nSHA256 robustness_results.json: {digest}")


if __name__ == "__main__":
    main()
