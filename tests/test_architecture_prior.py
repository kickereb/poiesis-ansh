"""Tests for random-function architecture priors."""

from __future__ import annotations

import unittest

import numpy as np

import architecture_prior
import spectral_theory


class ArchitecturePriorTests(unittest.TestCase):
    def test_parameter_counts_are_approximately_matched(self) -> None:
        specs = architecture_prior.make_architecture_specs()
        for spec in specs:
            relative_error = abs(
                spec.parameter_count - architecture_prior.TARGET_PARAMETERS
            ) / architecture_prior.TARGET_PARAMETERS
            self.assertLess(relative_error, 0.12, spec.name)

    def test_radial_spectrum_is_normalised(self) -> None:
        rng = np.random.default_rng(4)
        field = rng.normal(size=(16, 16))
        spectrum, density = architecture_prior.radial_spectrum(field, 8)
        self.assertAlmostEqual(float(spectrum.sum()), 1.0, places=12)
        self.assertAlmostEqual(float(density.sum()), 1.0, places=12)
        self.assertTrue(np.all(spectrum >= 0))

    def test_radial_spectrum_locates_a_sinusoid(self) -> None:
        axis = np.arange(32)
        field = np.sin(2.0 * np.pi * 3.0 * axis / 32.0)[None, :]
        field = np.repeat(field, 32, axis=0)
        spectrum, _ = architecture_prior.radial_spectrum(field, 16)
        self.assertEqual(int(np.argmax(spectrum)) + 1, 3)

    def test_all_architectures_generate_finite_fields(self) -> None:
        coordinates, coordinate_image = architecture_prior.coordinate_grid(8)
        for index, spec in enumerate(architecture_prior.make_architecture_specs(512)):
            rng = np.random.default_rng(index)
            field = architecture_prior.sample_architecture(
                spec, rng, coordinates, coordinate_image
            )
            self.assertEqual(field.shape, (8, 8))
            self.assertTrue(np.all(np.isfinite(field)), spec.name)
            self.assertAlmostEqual(float(np.mean(field**2)), 1.0, places=10)

    def test_sampling_is_deterministic(self) -> None:
        coordinates, coordinate_image = architecture_prior.coordinate_grid(8)
        spec = architecture_prior.make_architecture_specs(512)[0]
        first = architecture_prior.sample_architecture(
            spec, np.random.default_rng(99), coordinates, coordinate_image
        )
        second = architecture_prior.sample_architecture(
            spec, np.random.default_rng(99), coordinates, coordinate_image
        )
        np.testing.assert_array_equal(first, second)

    def test_measured_spectrum_preserves_shapley_efficiency(self) -> None:
        rng = np.random.default_rng(8)
        fields = rng.normal(size=(4, 16, 16))
        model_spectrum = np.mean(
            [architecture_prior.radial_spectrum(field, 8)[0] for field in fields],
            axis=0,
        )
        uniform = np.full(8, 1.0 / 8.0)
        covariance = spectral_theory.covariance_from_spectra(
            [uniform, model_spectrum, uniform]
        )
        noise = np.full(8, spectral_theory.NOISE_TOTAL_VARIANCE / 8.0)
        scenario = spectral_theory.Scenario(
            "test", "test", covariance, noise
        )
        summary, _ = spectral_theory.analyse_scenario(scenario)
        self.assertLess(summary["efficiency_error_bits"], 1e-12)


if __name__ == "__main__":
    unittest.main()
