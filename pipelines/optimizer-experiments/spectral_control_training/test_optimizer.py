from __future__ import annotations

import math
import unittest
from unittest import mock

import torch

from optimizer import SpectralControlOptimizer, _power_iteration_sigma_max


class PowerIterationSigmaMaxTests(unittest.TestCase):
    def test_power_iteration_avoids_random_initialization(self) -> None:
        weight = torch.tensor([[3.0, 0.0], [0.0, 1.0]])

        with mock.patch("torch.randn", side_effect=AssertionError("random init used")):
            sigma = _power_iteration_sigma_max(weight, iters=8)

        self.assertAlmostEqual(sigma.item(), 3.0, places=4)


class SpectralControlOptimizerTests(unittest.TestCase):
    def test_step_scales_by_natural_energy(self) -> None:
        param = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = SpectralControlOptimizer([param], T0=0.5, beta2=0.0, noise_beta=0.0)
        param.grad = torch.tensor([4.0])

        optimizer.step()

        expected_update = 0.5
        self.assertAlmostEqual(param.item(), 2.0 - expected_update, places=6)
        self.assertAlmostEqual(
            optimizer.state[param]["temperature"].item(),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            optimizer.state[param]["natural_energy"].item(),
            math.sqrt(4.0),
            places=6,
        )

    def test_step_averages_replicated_distributed_statistics(self) -> None:
        param = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = SpectralControlOptimizer([param], T0=0.5, beta2=0.0, noise_beta=0.5)
        param.grad = torch.tensor([4.0])

        world_size = 4

        def fake_all_reduce(tensor: torch.Tensor, op=None) -> None:
            tensor.mul_(world_size)

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=world_size),
            mock.patch("torch.distributed.all_reduce", side_effect=fake_all_reduce),
        ):
            optimizer.step()

        self.assertAlmostEqual(param.item(), 1.5, places=6)
        self.assertAlmostEqual(
            optimizer.state[param]["temperature"].item(),
            0.5 / (1.0 + 6.0),
            places=6,
        )
        self.assertAlmostEqual(
            optimizer.state[param]["natural_energy"].item(),
            math.sqrt(4.0),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
