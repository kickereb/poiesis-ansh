#!/usr/bin/env python3
"""Random-function architecture priors for Poiesis Ansh.

This pilot samples untrained networks with approximately matched parameter
counts. It measures each induced image spectrum and inserts the mean spectrum
into the exact Gaussian Shapley benchmark.

Dependencies: numpy and Pillow only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import spectral_theory


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
SEED = 20261007
GRID_SIZE = 32
DRAWS = 128
TARGET_PARAMETERS = 4096
HIDDEN_LAYERS = 3
SPECTRAL_BANDS = GRID_SIZE // 2


@dataclass(frozen=True)
class ArchitectureSpec:
    """One random-network design."""

    name: str
    family: str
    activation: str
    width: int
    parameter_count: int
    description: str
    layer_normalisation: bool = False


def dense_parameter_count(input_width: int, hidden_width: int) -> int:
    """Count parameters in a three-hidden-layer scalar MLP."""
    first = input_width * hidden_width + hidden_width
    hidden = (HIDDEN_LAYERS - 1) * (
        hidden_width * hidden_width + hidden_width
    )
    output = hidden_width + 1
    return first + hidden + output


def convolutional_parameter_count(width: int) -> int:
    """Count parameters in a three-hidden-layer 3x3 CNN."""
    first = 3 * 3 * 2 * width + width
    hidden = 2 * (3 * 3 * width * width + width)
    output = width + 1
    return first + hidden + output


def residual_parameter_count(width: int) -> int:
    """Count parameters in a two-block residual CNN."""
    input_projection = 2 * width + width
    residual_blocks = 4 * (3 * 3 * width * width + width)
    output = width + 1
    return input_projection + residual_blocks + output


def nearest_width(
    counter: Callable[[int], int], target: int = TARGET_PARAMETERS
) -> tuple[int, int]:
    """Find the integer width with the nearest parameter count."""
    candidates = [(width, counter(width)) for width in range(2, 257)]
    return min(candidates, key=lambda item: abs(item[1] - target))


def make_architecture_specs(
    target_parameters: int = TARGET_PARAMETERS,
) -> list[ArchitectureSpec]:
    """Create the matched random-network suite."""
    dense_width, dense_count = nearest_width(
        lambda width: dense_parameter_count(2, width), target_parameters
    )
    fourier_width, fourier_count = nearest_width(
        lambda width: dense_parameter_count(16, width), target_parameters
    )
    conv_width, conv_count = nearest_width(
        convolutional_parameter_count, target_parameters
    )
    residual_width, residual_count = nearest_width(
        residual_parameter_count, target_parameters
    )
    return [
        ArchitectureSpec(
            "relu_mlp",
            "coordinate_mlp",
            "relu",
            dense_width,
            dense_count,
            "Coordinate MLP with ReLU activations.",
        ),
        ArchitectureSpec(
            "tanh_mlp",
            "coordinate_mlp",
            "tanh",
            dense_width,
            dense_count,
            "Coordinate MLP with tanh activations.",
        ),
        ArchitectureSpec(
            "gaussian_mlp",
            "coordinate_mlp",
            "gaussian",
            dense_width,
            dense_count,
            "Coordinate MLP with Gaussian activations.",
        ),
        ArchitectureSpec(
            "relu_layernorm_mlp",
            "coordinate_mlp",
            "relu",
            dense_width,
            dense_count,
            "Coordinate MLP with ReLU and non-affine layer normalisation.",
            layer_normalisation=True,
        ),
        ArchitectureSpec(
            "siren_mlp",
            "siren",
            "sine",
            dense_width,
            dense_count,
            "Coordinate MLP with sinusoidal activations.",
        ),
        ArchitectureSpec(
            "fourier_relu_mlp",
            "fourier_mlp",
            "relu",
            fourier_width,
            fourier_count,
            "ReLU MLP with fixed Fourier input features.",
        ),
        ArchitectureSpec(
            "convolutional_relu",
            "convolutional",
            "relu",
            conv_width,
            conv_count,
            "Reflected-padding 3x3 convolutional network with ReLU activations.",
        ),
        ArchitectureSpec(
            "residual_convolutional_relu",
            "residual_convolutional",
            "relu",
            residual_width,
            residual_count,
            "Reflected-padding residual convolutional network with ReLU activations.",
        ),
    ]


def coordinate_grid(grid_size: int = GRID_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened and image-shaped coordinates in [-1, 1]."""
    axis = np.linspace(-1.0, 1.0, grid_size, dtype=float)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    image = np.stack([x, y], axis=-1)
    return image.reshape(-1, 2), image


def fourier_features(coordinates: np.ndarray) -> np.ndarray:
    """Return 16 fixed axis-aligned Fourier features."""
    features = []
    for frequency in (1.0, 2.0, 4.0, 8.0):
        for axis in range(2):
            angle = math.pi * frequency * coordinates[:, axis]
            features.extend([np.sin(angle), np.cos(angle)])
    return np.stack(features, axis=1)


def apply_activation(values: np.ndarray, activation: str) -> np.ndarray:
    """Apply the declared activation function."""
    if activation == "relu":
        return np.maximum(values, 0.0)
    if activation == "tanh":
        return np.tanh(values)
    if activation == "gaussian":
        return np.exp(-(values**2))
    raise ValueError(f"unknown activation: {activation}")


def layer_normalise(values: np.ndarray) -> np.ndarray:
    """Apply non-affine feature normalisation at each coordinate."""
    mean = values.mean(axis=1, keepdims=True)
    variance = values.var(axis=1, keepdims=True)
    return (values - mean) / np.sqrt(variance + 1e-6)


def sample_dense_function(
    spec: ArchitectureSpec,
    rng: np.random.Generator,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Sample one coordinate or Fourier MLP function."""
    values = (
        fourier_features(coordinates)
        if spec.family == "fourier_mlp"
        else coordinates
    )
    for _ in range(HIDDEN_LAYERS):
        input_width = values.shape[1]
        scale = math.sqrt(2.0 / input_width) if spec.activation == "relu" else 1.0 / math.sqrt(input_width)
        weights = rng.normal(0.0, scale, size=(input_width, spec.width))
        biases = rng.normal(0.0, 0.05, size=spec.width)
        values = values @ weights + biases
        if spec.layer_normalisation:
            values = layer_normalise(values)
        values = apply_activation(values, spec.activation)
    output_weights = rng.normal(0.0, 1.0 / math.sqrt(spec.width), size=spec.width)
    return values @ output_weights + rng.normal(0.0, 0.05)


def sample_siren_function(
    spec: ArchitectureSpec,
    rng: np.random.Generator,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Sample one sinusoidal representation network function."""
    omega = 20.0
    values = coordinates
    for layer in range(HIDDEN_LAYERS):
        input_width = values.shape[1]
        if layer == 0:
            limit = 1.0 / input_width
        else:
            limit = math.sqrt(6.0 / input_width) / omega
        weights = rng.uniform(-limit, limit, size=(input_width, spec.width))
        biases = rng.uniform(-limit, limit, size=spec.width)
        values = np.sin(omega * (values @ weights + biases))
    output_weights = rng.uniform(
        -1.0 / math.sqrt(spec.width),
        1.0 / math.sqrt(spec.width),
        size=spec.width,
    )
    return values @ output_weights


def spatial_convolution(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply one reflected-padding 3x3 convolution without external libraries."""
    output = np.zeros(
        (values.shape[0], values.shape[1], weights.shape[3]), dtype=float
    )
    padded = np.pad(values, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    height, width = values.shape[:2]
    for row in range(3):
        for column in range(3):
            window = padded[row : row + height, column : column + width]
            output += window @ weights[row, column]
    return output


def convolution_weights(
    rng: np.random.Generator, input_width: int, output_width: int
) -> np.ndarray:
    """Sample He-normal 3x3 convolution weights."""
    scale = math.sqrt(2.0 / (9 * input_width))
    return rng.normal(
        0.0, scale, size=(3, 3, input_width, output_width)
    )


def sample_convolutional_function(
    spec: ArchitectureSpec,
    rng: np.random.Generator,
    coordinate_image: np.ndarray,
) -> np.ndarray:
    """Sample one plain circular convolutional function."""
    values = coordinate_image
    for _ in range(HIDDEN_LAYERS):
        weights = convolution_weights(rng, values.shape[2], spec.width)
        biases = rng.normal(0.0, 0.05, size=spec.width)
        values = np.maximum(spatial_convolution(values, weights) + biases, 0.0)
    output_weights = rng.normal(
        0.0, 1.0 / math.sqrt(spec.width), size=spec.width
    )
    return values @ output_weights + rng.normal(0.0, 0.05)


def sample_residual_function(
    spec: ArchitectureSpec,
    rng: np.random.Generator,
    coordinate_image: np.ndarray,
) -> np.ndarray:
    """Sample one two-block circular residual function."""
    input_weights = rng.normal(
        0.0, math.sqrt(2.0 / 2), size=(2, spec.width)
    )
    values = np.maximum(
        coordinate_image @ input_weights
        + rng.normal(0.0, 0.05, size=spec.width),
        0.0,
    )
    for _ in range(2):
        first = convolution_weights(rng, spec.width, spec.width)
        second = convolution_weights(rng, spec.width, spec.width)
        hidden = np.maximum(
            spatial_convolution(values, first)
            + rng.normal(0.0, 0.05, size=spec.width),
            0.0,
        )
        residual = spatial_convolution(hidden, second)
        values = np.maximum((values + 0.5 * residual) / math.sqrt(1.25), 0.0)
    output_weights = rng.normal(
        0.0, 1.0 / math.sqrt(spec.width), size=spec.width
    )
    return values @ output_weights + rng.normal(0.0, 0.05)


def standardise_field(field: np.ndarray) -> np.ndarray:
    """Remove the mean and set unit root-mean-square amplitude."""
    field = np.asarray(field, dtype=float)
    field = field - field.mean()
    scale = math.sqrt(float(np.mean(field**2)))
    if scale < 1e-12:
        raise ValueError("sampled function is constant")
    return field / scale


def sample_architecture(
    spec: ArchitectureSpec,
    rng: np.random.Generator,
    coordinates: np.ndarray,
    coordinate_image: np.ndarray,
) -> np.ndarray:
    """Sample one standardised image function from an architecture."""
    grid_size = coordinate_image.shape[0]
    if spec.family in {"coordinate_mlp", "fourier_mlp"}:
        field = sample_dense_function(spec, rng, coordinates).reshape(
            grid_size, grid_size
        )
    elif spec.family == "siren":
        field = sample_siren_function(spec, rng, coordinates).reshape(
            grid_size, grid_size
        )
    elif spec.family == "convolutional":
        field = sample_convolutional_function(spec, rng, coordinate_image)
    elif spec.family == "residual_convolutional":
        field = sample_residual_function(spec, rng, coordinate_image)
    else:
        raise ValueError(f"unknown architecture family: {spec.family}")
    return standardise_field(field)


def radial_spectrum(
    field: np.ndarray, n_bands: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return radial energy and radial power density without the DC term."""
    grid_size = field.shape[0]
    n_bands = n_bands or grid_size // 2
    transformed = np.fft.fftshift(np.fft.fft2(field))
    power = np.abs(transformed) ** 2
    axis = np.arange(grid_size) - grid_size // 2
    y, x = np.meshgrid(axis, axis, indexing="ij")
    radius = np.sqrt(x**2 + y**2)
    band_index = np.floor(radius).astype(int)
    valid = (band_index >= 1) & (band_index <= n_bands)
    indices = band_index[valid] - 1
    energy = np.bincount(
        indices, weights=power[valid], minlength=n_bands
    ).astype(float)
    counts = np.bincount(indices, minlength=n_bands).astype(float)
    density = energy / np.maximum(counts, 1.0)
    if energy.sum() <= 0:
        raise ValueError("spectrum has no non-DC energy")
    return energy / energy.sum(), density / density.sum()


def field_metrics(field: np.ndarray, spectrum: np.ndarray) -> dict[str, float]:
    """Measure spatial and spectral function complexity."""
    horizontal = np.diff(field, axis=1)
    vertical = np.diff(field, axis=0)
    laplacian = (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    )
    frequencies = np.arange(1, len(spectrum) + 1, dtype=float) / len(spectrum)
    entropy = -float(np.sum(spectrum * np.log(np.maximum(spectrum, 1e-15))))
    return {
        "total_variation": float(np.mean(np.abs(horizontal)) + np.mean(np.abs(vertical))),
        "laplacian_energy": float(np.mean(laplacian**2)),
        "spectral_centroid": float(np.dot(frequencies, spectrum)),
        "low_frequency_share": float(spectrum[: max(1, len(spectrum) // 4)].sum()),
        "effective_frequency_bands": float(math.exp(entropy)),
        "bands_to_90_percent": float(np.searchsorted(np.cumsum(spectrum), 0.9) + 1),
    }


def spectral_slope(density: np.ndarray) -> float:
    """Fit a log-log radial power-density slope."""
    frequencies = np.arange(1, len(density) + 1, dtype=float)
    valid = density > 1e-15
    slope, _ = np.polyfit(
        np.log(frequencies[valid]), np.log(density[valid]), deg=1
    )
    return float(slope)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write stable CSV output."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_sample_montage(path: Path, samples: dict[str, list[np.ndarray]]) -> None:
    """Render six sampled functions from each architecture."""
    scale = 3
    label_width = 185
    cell = GRID_SIZE * scale
    rows = len(samples)
    columns = max(len(fields) for fields in samples.values())
    image = Image.new("RGB", (label_width + columns * cell, rows * cell), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for row, (name, fields) in enumerate(samples.items()):
        draw.text((5, row * cell + 8), name, fill="black", font=font)
        for column, field in enumerate(fields):
            normalised = (field - field.min()) / max(field.max() - field.min(), 1e-12)
            tile = Image.fromarray(np.uint8(np.round(normalised * 255)), mode="L")
            tile = tile.resize((cell, cell), Image.Resampling.NEAREST).convert("RGB")
            image.paste(tile, (label_width + column * cell, row * cell))
    image.save(path)


def make_spectrum_plot(path: Path, spectra_rows: list[dict]) -> None:
    """Draw the mean normalised energy spectrum for each architecture."""
    width, height = 1100, 620
    left, right, top, bottom = 75, 25, 45, 65
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (23, 190, 207),
    ]
    names = list(dict.fromkeys(row["architecture"] for row in spectra_rows))
    maximum = max(row["mean_energy_share"] for row in spectra_rows)
    draw.text((20, 12), "Random-function architecture spectra", fill="black", font=font)
    draw.line((left, top, left, height - bottom), fill=(70, 70, 70), width=1)
    draw.line((left, height - bottom, width - right, height - bottom), fill=(70, 70, 70), width=1)
    for index, name in enumerate(names):
        rows = [row for row in spectra_rows if row["architecture"] == name]
        points = []
        for band, row in enumerate(rows):
            x = left + band * (width - left - right) / (len(rows) - 1)
            y = height - bottom - row["mean_energy_share"] * (height - top - bottom) / maximum
            points.append((x, y))
        colour = palette[index]
        draw.line(points, fill=colour, width=3)
        legend_x = 75 + (index % 4) * 250
        legend_y = height - 48 + (index // 4) * 18
        draw.line((legend_x, legend_y + 5, legend_x + 22, legend_y + 5), fill=colour, width=3)
        draw.text((legend_x + 28, legend_y), name, fill="black", font=font)
    draw.text((left, height - bottom + 8), "low frequency", fill=(70, 70, 70), font=font)
    draw.text((width - 115, height - bottom + 8), "high frequency", fill=(70, 70, 70), font=font)
    image.save(path)


def run(
    draws: int = DRAWS,
    grid_size: int = GRID_SIZE,
    target_parameters: int = TARGET_PARAMETERS,
) -> dict:
    """Run the random-function architecture pilot."""
    if grid_size != GRID_SIZE:
        raise ValueError("artifact rendering currently requires a 32x32 grid")
    ARTIFACTS.mkdir(exist_ok=True)
    coordinates, coordinate_image = coordinate_grid(grid_size)
    specs = make_architecture_specs(target_parameters)
    seed_sequences = np.random.SeedSequence(SEED).spawn(len(specs))

    summaries: list[dict] = []
    spectra_rows: list[dict] = []
    attribution_rows: list[dict] = []
    sample_fields: dict[str, list[np.ndarray]] = {}

    for spec, seed_sequence in zip(specs, seed_sequences):
        rng = np.random.default_rng(seed_sequence)
        spectra = []
        densities = []
        metrics = []
        samples = []
        for draw_index in range(draws):
            field = sample_architecture(
                spec, rng, coordinates, coordinate_image
            )
            spectrum, density = radial_spectrum(field, SPECTRAL_BANDS)
            spectra.append(spectrum)
            densities.append(density)
            metrics.append(field_metrics(field, spectrum))
            if draw_index < 6:
                samples.append(field.copy())
        sample_fields[spec.name] = samples

        spectra_array = np.asarray(spectra)
        density_array = np.asarray(densities)
        mean_spectrum = spectra_array.mean(axis=0)
        mean_spectrum = mean_spectrum / mean_spectrum.sum()
        mean_density = density_array.mean(axis=0)
        metric_names = list(metrics[0])
        summary = {
            "architecture": spec.name,
            "family": spec.family,
            "activation": spec.activation,
            "layer_normalisation": spec.layer_normalisation,
            "width": spec.width,
            "parameter_count": spec.parameter_count,
            "parameter_difference_percent": 100.0
            * (spec.parameter_count - target_parameters)
            / target_parameters,
            "draws": draws,
            "grid_size": grid_size,
            "spectral_slope": spectral_slope(mean_density),
        }
        for metric_name in metric_names:
            values = np.asarray([metric[metric_name] for metric in metrics])
            standard_deviation = float(values.std(ddof=1))
            standard_error = standard_deviation / math.sqrt(draws)
            summary[f"mean_{metric_name}"] = float(values.mean())
            summary[f"sd_{metric_name}"] = standard_deviation
            summary[f"se_{metric_name}"] = standard_error
            summary[f"ci95_low_{metric_name}"] = float(values.mean() - 1.96 * standard_error)
            summary[f"ci95_high_{metric_name}"] = float(values.mean() + 1.96 * standard_error)
        summaries.append(summary)

        for band in range(SPECTRAL_BANDS):
            spectra_rows.append(
                {
                    "architecture": spec.name,
                    "band": band + 1,
                    "normalised_frequency": (band + 1) / SPECTRAL_BANDS,
                    "mean_energy_share": float(mean_spectrum[band]),
                    "sd_energy_share": float(spectra_array[:, band].std(ddof=1)),
                    "mean_power_density_share": float(mean_density[band]),
                }
            )

        uniform = np.full(SPECTRAL_BANDS, 1.0 / SPECTRAL_BANDS)
        covariance = spectral_theory.covariance_from_spectra(
            [uniform, mean_spectrum, uniform]
        )
        noise = np.full(
            SPECTRAL_BANDS,
            spectral_theory.NOISE_TOTAL_VARIANCE / SPECTRAL_BANDS,
        )
        attribution_scenario = spectral_theory.Scenario(
            name=spec.name,
            description=spec.description,
            covariance=covariance,
            noise_variance=noise,
        )
        attribution, _ = spectral_theory.analyse_scenario(attribution_scenario)
        attribution_rows.append(
            {
                "architecture": spec.name,
                "total_information_bits": attribution["total_information_bits"],
                "prompt_share": attribution["prompt_share"],
                "model_prior_share": attribution["model_prior_share"],
                "seed_share": attribution["seed_share"],
                "efficiency_error_bits": attribution["efficiency_error_bits"],
            }
        )

    result = {
        "project": "Poiesis Ansh",
        "experiment": "random-function architecture prior pilot",
        "seed": SEED,
        "draws_per_architecture": draws,
        "grid_size": grid_size,
        "target_parameters": target_parameters,
        "spectral_bands": SPECTRAL_BANDS,
        "scope_note": "This is a spectral extension, not a direct replication of the paper's Boolean-function experiments.",
        "architecture_summaries": summaries,
        "spectra": spectra_rows,
        "spectral_attribution": attribution_rows,
    }

    write_csv(ARTIFACTS / "architecture_prior_summary.csv", summaries)
    write_csv(ARTIFACTS / "architecture_prior_spectra.csv", spectra_rows)
    write_csv(ARTIFACTS / "architecture_attribution.csv", attribution_rows)
    make_sample_montage(
        ARTIFACTS / "architecture_prior_samples.png", sample_fields
    )
    make_spectrum_plot(
        ARTIFACTS / "architecture_prior_spectra.png", spectra_rows
    )
    json_path = ARTIFACTS / "architecture_prior_results.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    (ARTIFACTS / "architecture_prior_results.sha256").write_text(
        f"{digest}  architecture_prior_results.json\n", encoding="utf-8"
    )
    return result


def main() -> None:
    result = run()
    output = []
    attribution_by_name = {
        row["architecture"]: row for row in result["spectral_attribution"]
    }
    for row in result["architecture_summaries"]:
        attribution = attribution_by_name[row["architecture"]]
        output.append(
            {
                "architecture": row["architecture"],
                "parameters": row["parameter_count"],
                "low_frequency_share": row["mean_low_frequency_share"],
                "spectral_centroid": row["mean_spectral_centroid"],
                "model_prior_shapley_share": attribution["model_prior_share"],
            }
        )
    print(json.dumps(output, indent=2))
    digest = (ARTIFACTS / "architecture_prior_results.sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    print(f"\nSHA256 architecture_prior_results.json: {digest}")


if __name__ == "__main__":
    main()
