"""Unit tests that do not require loading model weights."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn

from activation_tools import ActivationPatcher, ActivationRecorder
from model_utils import load_config, regional_metrics
from run_gradient_svd_lora import direction_declaration
from run_seed_robustness import aggregate, aggregate_objectives
from weight_tools import (
    LowRankDirection,
    TrainableLoRAAdapter,
    directional_output_gradient,
)


class RealModelToolTests(unittest.TestCase):
    def test_direction_declaration_binds_run_inputs(self) -> None:
        config = load_config()
        pipeline = SimpleNamespace(
            scheduler=SimpleNamespace(
                config={
                    "_use_default_values": ["zeta", "alpha"],
                    "beta_start": 0.001,
                    "sample_steps": (1, 2),
                }
            )
        )
        declaration = direction_declaration(config, pipeline, "test.module")
        self.assertEqual(declaration["width"], 512)
        self.assertEqual(declaration["height"], 512)
        self.assertEqual(declaration["scheduler_config"]["sample_steps"], [1, 2])
        self.assertEqual(
            declaration["scheduler_config"]["_use_default_values"],
            ["alpha", "zeta"],
        )
        self.assertEqual(len(declaration["model_manifest_sha256"]), 64)
        self.assertIn("torch", declaration["runtime"])

    def test_held_out_aggregates_exclude_fitting_seed(self) -> None:
        rows = [
            {
                "seed": 1,
                "intervention": "test",
                "region": "full",
                "restoration": 0.9,
                "target_change_rmse": 0.2,
            },
            {
                "seed": 2,
                "intervention": "test",
                "region": "full",
                "restoration": -0.1,
                "target_change_rmse": 0.4,
            },
        ]
        summary = aggregate([row for row in rows if row["seed"] != 1])
        self.assertEqual(summary["test"]["seeds"], 1)
        self.assertEqual(summary["test"]["restoration_mean"], -0.1)
        objective = aggregate_objectives(
            [
                {"strength": 0.1, "objective_reduction": 0.2},
                {"strength": 0.1, "objective_reduction": -0.2},
            ]
        )
        self.assertEqual(objective["+0.10"]["positive_reduction_share"], 0.5)

    def test_activation_recorder_captures_calls(self) -> None:
        module = nn.Linear(3, 3, bias=False)
        with ActivationRecorder(module) as recorder:
            module(torch.ones(2, 3))
            module(torch.zeros(2, 3))
        self.assertEqual(set(recorder.activations), {0, 1})

    def test_activation_patcher_replaces_conditional_half(self) -> None:
        module = nn.Identity()
        donor = {0: torch.full((4, 3), 7.0)}
        values = torch.zeros(4, 3)
        with ActivationPatcher(module, donor, {0}) as patcher:
            output = module(values)
        torch.testing.assert_close(output[:2], torch.zeros(2, 3))
        torch.testing.assert_close(output[2:], torch.full((2, 3), 7.0))
        self.assertEqual(patcher.patched_calls, 1)

    def test_low_rank_context_restores_weight(self) -> None:
        module = nn.Linear(5, 4, bias=False)
        original = module.weight.detach().clone()
        direction = LowRankDirection.random(module, rank=2, relative_norm=0.01, seed=4)
        with direction.applied(1.0):
            self.assertFalse(torch.equal(module.weight, original))
        self.assertTrue(torch.equal(module.weight, original))

    def test_trainable_lora_hook_adds_low_rank_output(self) -> None:
        module = nn.Linear(5, 4, bias=False)
        values = torch.ones(2, 5)
        baseline = module(values).detach()
        adapter = TrainableLoRAAdapter(module, rank=2, seed=5)
        with adapter:
            torch.testing.assert_close(module(values), baseline)
            with torch.no_grad():
                adapter.B.fill_(0.1)
            self.assertFalse(torch.equal(module(values), baseline))
        torch.testing.assert_close(module(values), baseline)
        self.assertGreater(float(adapter.delta().norm()), 0.0)
        self.assertLessEqual(torch.linalg.matrix_rank(adapter.delta()), 2)

    def test_directional_output_gradient(self) -> None:
        negative = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
        positive = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
        gradient = directional_output_gradient(negative, positive, 0.5)
        np.testing.assert_allclose(gradient, np.ones((4, 4, 3)))

    def test_regional_metrics_identify_left_change(self) -> None:
        reference = Image.fromarray(np.zeros((6, 6, 3), dtype=np.uint8))
        changed_array = np.zeros((6, 6, 3), dtype=np.uint8)
        changed_array[:, :2] = 255
        changed = Image.fromarray(changed_array)
        metrics = {row["region"]: row for row in regional_metrics(reference, changed)}
        self.assertGreater(metrics["left"]["rmse"], 0.9)
        self.assertEqual(metrics["right"]["rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
