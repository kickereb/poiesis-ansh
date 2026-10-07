#!/usr/bin/env python3
"""Exact Gaussian spectral Shapley benchmark for Poiesis Ansh.

The benchmark models each frequency band as a jointly Gaussian prompt, model,
and seed source. It computes coalition mutual information and exact Shapley
values. The implementation supports independent and correlated sources.

Dependencies: numpy and Pillow only.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
PLAYERS = ("prompt", "model_prior", "seed")
PLAYER_COLOURS = {
    "prompt": (42, 105, 196),
    "model_prior": (213, 94, 0),
    "seed": (85, 139, 47),
}
N_BANDS = 64
NOISE_TOTAL_VARIANCE = 0.15
TOLERANCE = 1e-10


@dataclass(frozen=True)
class Scenario:
    """One declared Gaussian source population."""

    name: str
    description: str
    covariance: np.ndarray
    noise_variance: np.ndarray


def all_coalitions(n_players: int = len(PLAYERS)) -> Iterable[tuple[int, ...]]:
    """Yield each coalition in stable size order."""
    for size in range(n_players + 1):
        yield from itertools.combinations(range(n_players), size)


def power_spectrum(
    exponent: float, total_variance: float = 1.0, n_bands: int = N_BANDS
) -> np.ndarray:
    """Return a normalised power-law spectrum."""
    frequencies = np.arange(1, n_bands + 1, dtype=float)
    power = frequencies ** (-exponent)
    return total_variance * power / power.sum()


def covariance_from_spectra(
    spectra: Sequence[np.ndarray], prompt_model_correlation: float = 0.0
) -> np.ndarray:
    """Build one three-source covariance matrix for each frequency band."""
    stacked = np.asarray(spectra, dtype=float)
    if stacked.shape[0] != len(PLAYERS):
        raise ValueError("spectra must contain prompt, model, and seed arrays")
    if not -1.0 <= prompt_model_correlation <= 1.0:
        raise ValueError("correlation must be between -1 and 1")

    n_bands = stacked.shape[1]
    covariance = np.zeros((n_bands, len(PLAYERS), len(PLAYERS)), dtype=float)
    for player in range(len(PLAYERS)):
        covariance[:, player, player] = stacked[player]
    cross = prompt_model_correlation * np.sqrt(stacked[0] * stacked[1])
    covariance[:, 0, 1] = cross
    covariance[:, 1, 0] = cross
    return covariance


def validate_inputs(covariance: np.ndarray, noise_variance: np.ndarray) -> None:
    """Reject invalid covariance and noise inputs."""
    if covariance.ndim != 3 or covariance.shape[1:] != (3, 3):
        raise ValueError("covariance must have shape (bands, 3, 3)")
    if noise_variance.shape != (covariance.shape[0],):
        raise ValueError("noise variance must contain one value per band")
    if np.any(noise_variance <= 0):
        raise ValueError("noise variance must be positive")
    if not np.allclose(covariance, covariance.transpose(0, 2, 1), atol=TOLERANCE):
        raise ValueError("covariance matrices must be symmetric")
    if np.min(np.linalg.eigvalsh(covariance)) < -TOLERANCE:
        raise ValueError("covariance matrices must be positive semidefinite")


def coalition_information_by_band(
    covariance: np.ndarray,
    noise_variance: np.ndarray,
    coalition: Sequence[int],
) -> np.ndarray:
    """Compute I(source coalition; output) in bits for each band."""
    validate_inputs(covariance, noise_variance)
    coalition = tuple(sorted(coalition))
    if len(set(coalition)) != len(coalition):
        raise ValueError("coalition players must be unique")
    if any(player < 0 or player >= len(PLAYERS) for player in coalition):
        raise ValueError("coalition contains an unknown player")

    weights = np.ones(len(PLAYERS), dtype=float)
    total_variance = np.einsum("i,bij,j->b", weights, covariance, weights)
    total_variance = total_variance + noise_variance
    if not coalition:
        return np.zeros(covariance.shape[0], dtype=float)

    conditional_variance = np.empty(covariance.shape[0], dtype=float)
    indices = np.asarray(coalition, dtype=int)
    for band, band_covariance in enumerate(covariance):
        observed_covariance = band_covariance[np.ix_(indices, indices)]
        output_observed_covariance = (weights @ band_covariance)[indices]
        explained = output_observed_covariance @ np.linalg.pinv(
            observed_covariance, hermitian=True
        ) @ output_observed_covariance
        conditional_variance[band] = total_variance[band] - explained

    conditional_variance = np.maximum(conditional_variance, noise_variance)
    information = 0.5 * np.log2(total_variance / conditional_variance)
    return np.maximum(information, 0.0)


def information_game_by_band(
    covariance: np.ndarray, noise_variance: np.ndarray
) -> Dict[tuple[int, ...], np.ndarray]:
    """Compute every coalition value for each frequency band."""
    return {
        coalition: coalition_information_by_band(
            covariance, noise_variance, coalition
        )
        for coalition in all_coalitions()
    }


def exact_shapley_by_band(
    game: Mapping[tuple[int, ...], np.ndarray]
) -> np.ndarray:
    """Return exact player-by-band Shapley values."""
    n_players = len(PLAYERS)
    n_bands = len(game[()])
    values = np.zeros((n_players, n_bands), dtype=float)
    all_players = set(range(n_players))
    for player in range(n_players):
        others = sorted(all_players - {player})
        for size in range(len(others) + 1):
            coefficient = (
                math.factorial(size)
                * math.factorial(n_players - size - 1)
                / math.factorial(n_players)
            )
            for subset in itertools.combinations(others, size):
                with_player = tuple(sorted(subset + (player,)))
                values[player] += coefficient * (
                    game[with_player] - game[tuple(subset)]
                )
    return values


def spectral_centroid(values: np.ndarray) -> float:
    """Return the normalised centre frequency of nonnegative attribution."""
    frequencies = np.arange(1, len(values) + 1, dtype=float) / len(values)
    total = float(values.sum())
    if total <= TOLERANCE:
        return 0.0
    return float(np.dot(frequencies, values) / total)


def make_scenarios(n_bands: int = N_BANDS) -> list[Scenario]:
    """Create the first declared spectral benchmark suite."""
    noise = np.full(n_bands, NOISE_TOTAL_VARIANCE / n_bands, dtype=float)

    def scenario(
        name: str,
        description: str,
        exponents: tuple[float, float, float],
        totals: tuple[float, float, float] = (1.0, 1.0, 1.0),
        correlation: float = 0.0,
    ) -> Scenario:
        spectra = [
            power_spectrum(exponent, total, n_bands)
            for exponent, total in zip(exponents, totals)
        ]
        return Scenario(
            name=name,
            description=description,
            covariance=covariance_from_spectra(spectra, correlation),
            noise_variance=noise.copy(),
        )

    return [
        scenario(
            "balanced_white",
            "All sources have equal white spectra.",
            (0.0, 0.0, 0.0),
        ),
        scenario(
            "red_model",
            "The model prior concentrates power at low frequencies.",
            (0.0, 2.0, 0.0),
        ),
        scenario(
            "red_prompt",
            "The prompt concentrates power at low frequencies.",
            (2.0, 0.0, 0.0),
        ),
        scenario(
            "frequency_separated",
            "The model is low-frequency and the prompt is high-frequency.",
            (-2.0, 2.0, 0.0),
        ),
        scenario(
            "seed_dominant",
            "The seed has four times the total source variance.",
            (0.5, 1.5, 0.0),
            (1.0, 1.0, 4.0),
        ),
        scenario(
            "correlated_prompt_model",
            "Prompt and model signals have a 0.65 correlation.",
            (1.0, 1.8, 0.0),
            correlation=0.65,
        ),
    ]


def analyse_scenario(scenario: Scenario) -> tuple[dict, list[dict]]:
    """Compute exact total and bandwise results for one scenario."""
    game = information_game_by_band(
        scenario.covariance, scenario.noise_variance
    )
    shapley = exact_shapley_by_band(game)
    total_shapley = shapley.sum(axis=1)
    total_information = float(game[(0, 1, 2)].sum())
    shares = total_shapley / total_information
    efficiency_error = abs(float(total_shapley.sum()) - total_information)
    band_efficiency = np.abs(shapley.sum(axis=0) - game[(0, 1, 2)])

    summary = {
        "scenario": scenario.name,
        "description": scenario.description,
        "bands": scenario.covariance.shape[0],
        "total_information_bits": total_information,
        "prompt_shapley_bits": float(total_shapley[0]),
        "model_prior_shapley_bits": float(total_shapley[1]),
        "seed_shapley_bits": float(total_shapley[2]),
        "prompt_share": float(shares[0]),
        "model_prior_share": float(shares[1]),
        "seed_share": float(shares[2]),
        "prompt_spectral_centroid": spectral_centroid(shapley[0]),
        "model_prior_spectral_centroid": spectral_centroid(shapley[1]),
        "seed_spectral_centroid": spectral_centroid(shapley[2]),
        "efficiency_error_bits": efficiency_error,
        "max_band_efficiency_error_bits": float(band_efficiency.max()),
    }

    rows: list[dict] = []
    for band in range(scenario.covariance.shape[0]):
        band_total = float(game[(0, 1, 2)][band])
        band_shares = shapley[:, band] / band_total
        prompt_variance = scenario.covariance[band, 0, 0]
        model_variance = scenario.covariance[band, 1, 1]
        correlation = scenario.covariance[band, 0, 1] / math.sqrt(
            prompt_variance * model_variance
        )
        rows.append(
            {
                "scenario": scenario.name,
                "band": band + 1,
                "normalised_frequency": (band + 1) / scenario.covariance.shape[0],
                "prompt_variance": float(prompt_variance),
                "model_prior_variance": float(model_variance),
                "seed_variance": float(scenario.covariance[band, 2, 2]),
                "prompt_model_correlation": float(correlation),
                "band_information_bits": band_total,
                "prompt_shapley_bits": float(shapley[0, band]),
                "model_prior_shapley_bits": float(shapley[1, band]),
                "seed_shapley_bits": float(shapley[2, band]),
                "prompt_share": float(band_shares[0]),
                "model_prior_share": float(band_shares[1]),
                "seed_share": float(band_shares[2]),
                "efficiency_error_bits": float(band_efficiency[band]),
            }
        )
    return summary, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write stable CSV output."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_profiles(path: Path, summaries: list[dict], band_rows: list[dict]) -> None:
    """Draw deterministic bandwise Shapley profiles without Matplotlib."""
    width = 1100
    row_height = 185
    margin_left = 70
    margin_right = 25
    plot_width = width - margin_left - margin_right
    height = 60 + row_height * len(summaries)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((20, 12), "Poiesis Ansh: exact spectral Shapley profiles", fill="black", font=font)
    legend_x = 470
    for player in PLAYERS:
        colour = PLAYER_COLOURS[player]
        draw.line((legend_x, 18, legend_x + 24, 18), fill=colour, width=3)
        draw.text((legend_x + 30, 12), player, fill="black", font=font)
        legend_x += 165

    for row_index, summary in enumerate(summaries):
        top = 52 + row_index * row_height
        bottom = top + 135
        scenario_rows = [
            row for row in band_rows if row["scenario"] == summary["scenario"]
        ]
        maximum = max(
            row[f"{player}_shapley_bits"]
            for row in scenario_rows
            for player in PLAYERS
        )
        maximum = max(maximum, 1e-12)
        draw.text((margin_left, top - 14), summary["scenario"], fill="black", font=font)
        draw.line((margin_left, top, margin_left, bottom), fill=(80, 80, 80), width=1)
        draw.line((margin_left, bottom, width - margin_right, bottom), fill=(80, 80, 80), width=1)

        for player in PLAYERS:
            points = []
            for index, row in enumerate(scenario_rows):
                x = margin_left + index * plot_width / (len(scenario_rows) - 1)
                value = row[f"{player}_shapley_bits"]
                y = bottom - value * (bottom - top) / maximum
                points.append((x, y))
            draw.line(points, fill=PLAYER_COLOURS[player], width=3)

        draw.text((15, top), f"{maximum:.3f}", fill=(70, 70, 70), font=font)
        draw.text((43, bottom - 7), "0", fill=(70, 70, 70), font=font)
        draw.text((margin_left, bottom + 6), "low frequency", fill=(70, 70, 70), font=font)
        draw.text((width - 115, bottom + 6), "high frequency", fill=(70, 70, 70), font=font)

    image.save(path)


def run() -> dict:
    """Run the benchmark and write all declared artifacts."""
    ARTIFACTS.mkdir(exist_ok=True)
    summaries: list[dict] = []
    band_rows: list[dict] = []
    for scenario in make_scenarios():
        summary, rows = analyse_scenario(scenario)
        summaries.append(summary)
        band_rows.extend(rows)

    result = {
        "project": "Poiesis Ansh",
        "benchmark": "exact Gaussian spectral Shapley",
        "players": list(PLAYERS),
        "bands": N_BANDS,
        "noise_total_variance": NOISE_TOTAL_VARIANCE,
        "assumptions": [
            "Frequency bands are independent.",
            "Each band contains jointly Gaussian source values.",
            "The output is the source sum plus independent Gaussian noise.",
            "Architecture is an experimental condition, not a player.",
        ],
        "summaries": summaries,
        "band_results": band_rows,
    }

    write_csv(ARTIFACTS / "spectral_summary.csv", summaries)
    write_csv(ARTIFACTS / "spectral_shapley.csv", band_rows)
    draw_profiles(
        ARTIFACTS / "spectral_shapley_profiles.png", summaries, band_rows
    )
    json_path = ARTIFACTS / "spectral_results.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    (ARTIFACTS / "spectral_results.sha256").write_text(
        f"{digest}  spectral_results.json\n", encoding="utf-8"
    )
    return result


def main() -> None:
    result = run()
    compact = {
        row["scenario"]: {
            "total_information_bits": row["total_information_bits"],
            "prompt_share": row["prompt_share"],
            "model_prior_share": row["model_prior_share"],
            "seed_share": row["seed_share"],
            "efficiency_error_bits": row["efficiency_error_bits"],
        }
        for row in result["summaries"]
    }
    print(json.dumps(compact, indent=2))
    digest = (ARTIFACTS / "spectral_results.sha256").read_text(encoding="utf-8").split()[0]
    print(f"\nSHA256 spectral_results.json: {digest}")


if __name__ == "__main__":
    main()
