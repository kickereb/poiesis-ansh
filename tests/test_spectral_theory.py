"""Tests for the exact Gaussian spectral benchmark."""

from __future__ import annotations

import unittest

import numpy as np

import spectral_theory as spectral


class SpectralTheoryTests(unittest.TestCase):
    def test_balanced_white_is_symmetric(self) -> None:
        scenario = spectral.make_scenarios(n_bands=8)[0]
        summary, _ = spectral.analyse_scenario(scenario)
        self.assertAlmostEqual(summary["prompt_share"], 1.0 / 3.0, places=12)
        self.assertAlmostEqual(summary["model_prior_share"], 1.0 / 3.0, places=12)
        self.assertAlmostEqual(summary["seed_share"], 1.0 / 3.0, places=12)

    def test_shapley_efficiency_and_band_additivity(self) -> None:
        for scenario in spectral.make_scenarios(n_bands=12):
            game = spectral.information_game_by_band(
                scenario.covariance, scenario.noise_variance
            )
            shapley = spectral.exact_shapley_by_band(game)
            np.testing.assert_allclose(
                shapley.sum(axis=0), game[(0, 1, 2)], atol=1e-12
            )
            self.assertAlmostEqual(
                float(shapley.sum()), float(game[(0, 1, 2)].sum()), places=12
            )

    def test_independent_full_coalition_has_closed_form(self) -> None:
        spectra = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
        covariance = spectral.covariance_from_spectra(spectra)
        noise = np.array([0.5])
        actual = spectral.coalition_information_by_band(
            covariance, noise, (0, 1, 2)
        )[0]
        expected = 0.5 * np.log2((1.0 + 2.0 + 3.0 + 0.5) / 0.5)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_empty_coalition_has_zero_information(self) -> None:
        scenario = spectral.make_scenarios(n_bands=5)[1]
        actual = spectral.coalition_information_by_band(
            scenario.covariance, scenario.noise_variance, ()
        )
        np.testing.assert_array_equal(actual, np.zeros(5))

    def test_source_information_increases_with_own_variance(self) -> None:
        noise = np.array([1.0])
        low = spectral.covariance_from_spectra(
            [np.array([0.5]), np.array([1.0]), np.array([1.0])]
        )
        high = spectral.covariance_from_spectra(
            [np.array([2.0]), np.array([1.0]), np.array([1.0])]
        )
        low_value = spectral.coalition_information_by_band(low, noise, (0,))[0]
        high_value = spectral.coalition_information_by_band(high, noise, (0,))[0]
        self.assertGreater(high_value, low_value)

    def test_correlated_scenario_is_finite(self) -> None:
        scenario = spectral.make_scenarios(n_bands=10)[-1]
        summary, rows = spectral.analyse_scenario(scenario)
        self.assertTrue(np.isfinite(summary["total_information_bits"]))
        self.assertLess(summary["efficiency_error_bits"], 1e-12)
        self.assertTrue(all(np.isfinite(row["prompt_share"]) for row in rows))


if __name__ == "__main__":
    unittest.main()
