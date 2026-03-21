from __future__ import annotations

import sys
import unittest
from unittest import mock
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimizer import _power_iteration_sigma_max


class PowerIterationSigmaMaxTests(unittest.TestCase):
    def test_power_iteration_avoids_random_initialization(self) -> None:
        weight = torch.tensor([[3.0, 0.0], [0.0, 1.0]])

        with mock.patch("torch.randn", side_effect=AssertionError("random init used")):
            sigma = _power_iteration_sigma_max(weight, iters=8)

        self.assertAlmostEqual(sigma.item(), 3.0, places=4)


if __name__ == "__main__":
    unittest.main()
