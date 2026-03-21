from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimizer import SpectralControlOptimizer, _power_iteration_sigma_max


class PowerIterationSigmaMaxTests(unittest.TestCase):
    def test_power_iteration_uses_non_fixed_start_vector(self) -> None:
        weight = torch.tensor([[2.6, -0.8], [-0.8, 1.4]])
        start_vector = torch.tensor([1.0, 0.0])

        with mock.patch("torch.randn", return_value=start_vector) as randn_mock:
            sigma = _power_iteration_sigma_max(weight, iters=8)

        randn_mock.assert_called_once()
        self.assertAlmostEqual(sigma.item(), 3.0, places=4)

    def test_spectral_constraint_clips_when_sigma_exceeds_radius(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([[2.6, -0.8], [-0.8, 1.4]]))
        optimizer = SpectralControlOptimizer(
            [parameter],
            spectral_radius=2.5,
            spectral_update_period=1,
            power_iteration_steps=8,
        )

        parameter.grad = torch.zeros_like(parameter)
        with mock.patch("torch.randn", return_value=torch.tensor([1.0, 0.0])):
            optimizer.step()

        sigma = torch.linalg.matrix_norm(parameter.detach(), ord=2)
        self.assertLessEqual(sigma.item(), 2.5 + 1e-5)


if __name__ == "__main__":
    unittest.main()
