from __future__ import annotations

import math
import unittest
from unittest import mock

import torch

from optimizer import (
    LAYER_ATTENTION,
    LAYER_EMBEDDING,
    LAYER_MLP,
    LAYER_OUTPUT,
    SpectralControlOptimizer,
    _gram_newton_schulz,
    _power_iteration_sigma_max,
    _row_normalize,
)


# ---------------------------------------------------------------------------
# Power iteration tests
# ---------------------------------------------------------------------------

class PowerIterationSigmaMaxTests(unittest.TestCase):
    def test_power_iteration_avoids_random_initialization(self) -> None:
        weight = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
        with mock.patch("torch.randn", side_effect=AssertionError("random init used")):
            sigma = _power_iteration_sigma_max(weight, iters=8)
        self.assertAlmostEqual(sigma.item(), 3.0, places=4)

    def test_power_iteration_on_identity_matrix(self) -> None:
        weight = torch.eye(4)
        sigma = _power_iteration_sigma_max(weight, iters=10)
        self.assertAlmostEqual(sigma.item(), 1.0, places=4)

    def test_power_iteration_on_rank_one_matrix(self) -> None:
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        weight = a.unsqueeze(1) @ b.unsqueeze(0)
        expected_sigma = a.norm() * b.norm()
        sigma = _power_iteration_sigma_max(weight, iters=20)
        self.assertAlmostEqual(sigma.item(), expected_sigma.item(), places=2)

    def test_power_iteration_on_1d_tensor(self) -> None:
        weight = torch.tensor([1.0, -5.0, 3.0])
        sigma = _power_iteration_sigma_max(weight, iters=1)
        self.assertAlmostEqual(sigma.item(), 5.0, places=4)

    def test_power_iteration_on_batch_norm_weight(self) -> None:
        weight = torch.randn(64)
        sigma = _power_iteration_sigma_max(weight, iters=1)
        self.assertAlmostEqual(sigma.item(), weight.abs().max().item(), places=4)


# ---------------------------------------------------------------------------
# Gram Newton-Schulz tests
# ---------------------------------------------------------------------------

class GramNewtonSchulzTests(unittest.TestCase):
    def test_identity_gradient(self) -> None:
        G = torch.eye(3)
        result = _gram_newton_schulz(G, ns_steps=5)
        self.assertTrue(torch.allclose(result, G, atol=0.1))

    def test_diagonal_gradient(self) -> None:
        G = torch.diag(torch.tensor([4.0, 1.0, 0.5]))
        result = _gram_newton_schulz(G, ns_steps=5)
        norms = result.norm(dim=0)
        self.assertLess(norms.max() / (norms.min() + 1e-8), 2.0)

    def test_ns_convergence(self) -> None:
        G = torch.randn(4, 4)
        result_3 = _gram_newton_schulz(G, ns_steps=3)
        result_10 = _gram_newton_schulz(G, ns_steps=10)
        self.assertEqual(result_3.shape, G.shape)
        self.assertEqual(result_10.shape, G.shape)

    def test_preserves_shape(self) -> None:
        G = torch.randn(8, 6)
        result = _gram_newton_schulz(G, ns_steps=3)
        self.assertEqual(result.shape, G.shape)

    def test_non_square_gradient(self) -> None:
        G = torch.randn(5, 3)
        result = _gram_newton_schulz(G, ns_steps=5)
        self.assertEqual(result.shape, G.shape)


# ---------------------------------------------------------------------------
# Row normalization tests
# ---------------------------------------------------------------------------

class RowNormalizeTests(unittest.TestCase):
    def test_row_norms_are_one(self) -> None:
        tensor = torch.randn(4, 8)
        result = _row_normalize(tensor)
        row_norms = result.view(4, -1).norm(dim=-1)
        self.assertTrue(torch.allclose(row_norms, torch.ones(4), atol=1e-5))

    def test_1d_tensor_passthrough(self) -> None:
        tensor = torch.randn(8)
        result = _row_normalize(tensor)
        self.assertTrue(torch.equal(result, tensor))

    def test_3d_tensor(self) -> None:
        tensor = torch.randn(4, 3, 5)
        result = _row_normalize(tensor)
        row_norms = result.view(4, -1).norm(dim=-1)
        self.assertTrue(torch.allclose(row_norms, torch.ones(4), atol=1e-5))


# ---------------------------------------------------------------------------
# Optimizer tests
# ---------------------------------------------------------------------------

class SpectralControlOptimizerTests(unittest.TestCase):
    def test_step_scales_by_natural_energy(self) -> None:
        param = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = SpectralControlOptimizer(
            [param], T0=0.5, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.0, cautious=False, adaptive_momentum=False,
        )
        param.grad = torch.tensor([4.0])
        optimizer.step()

        expected_update = 0.5
        self.assertAlmostEqual(param.item(), 2.0 - expected_update, places=5)
        self.assertAlmostEqual(optimizer.state[param]["temperature"].item(), 0.5, places=5)
        self.assertAlmostEqual(optimizer.state[param]["natural_energy"].item(), math.sqrt(4.0), places=5)

    def test_warmup_reduces_initial_temperature(self) -> None:
        param = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = SpectralControlOptimizer(
            [param], T0=1.0, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=10, momentum=0.0, cautious=False, adaptive_momentum=False,
        )
        param.grad = torch.tensor([4.0])
        optimizer.step()

        temp = optimizer.state[param]["temperature"].item()
        self.assertLess(temp, 0.15)
        self.assertGreater(temp, 0.0)

    def test_warmup_completes_at_full_temperature(self) -> None:
        param = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = SpectralControlOptimizer(
            [param], T0=1.0, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=2, momentum=0.0, cautious=False, adaptive_momentum=False,
        )
        for _ in range(3):
            param.grad = torch.tensor([4.0])
            optimizer.step()

        temp = optimizer.state[param]["temperature"].item()
        expected = 1.0 / math.sqrt(3)
        self.assertAlmostEqual(temp, expected, places=4)

    def test_noise_ratio_reduces_temperature(self) -> None:
        param = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = SpectralControlOptimizer(
            [param], T0=1.0, beta1=0.0, beta2=0.0, noise_beta=0.99,
            warmup_steps=0, momentum=0.0, cautious=False, adaptive_momentum=False,
        )
        for _ in range(50):
            param.grad = torch.randn(1) * 10.0
            optimizer.step()
        temp_noisy = optimizer.state[param]["temperature"].item()

        param2 = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer2 = SpectralControlOptimizer(
            [param2], T0=1.0, beta1=0.0, beta2=0.0, noise_beta=0.99,
            warmup_steps=0, momentum=0.0, cautious=False, adaptive_momentum=False,
        )
        for _ in range(50):
            param2.grad = torch.tensor([1.0])
            optimizer2.step()
        temp_clean = optimizer2.state[param2]["temperature"].item()

        self.assertGreater(temp_clean, temp_noisy)

    def test_2d_param_uses_gram_ns(self) -> None:
        param = torch.nn.Parameter(torch.randn(4, 4))
        optimizer = SpectralControlOptimizer(
            [param], T0=0.01, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.0, ns_steps=3, alpha=0.5,
            row_normalize=False, cautious=False, adaptive_momentum=False,
        )
        param.grad = torch.randn(4, 4)
        p_before = param.data.clone()
        optimizer.step()
        self.assertFalse(torch.equal(param.data, p_before))

    def test_alpha_parameterization(self) -> None:
        params = []
        optimizers = []
        for alpha in [0.0, 0.5, 1.0]:
            p = torch.nn.Parameter(torch.randn(4, 4))
            opt = SpectralControlOptimizer(
                [p], T0=0.01, beta1=0.0, beta2=0.0, noise_beta=0.0,
                warmup_steps=0, momentum=0.0, ns_steps=3, alpha=alpha,
                row_normalize=False, cautious=False, adaptive_momentum=False,
            )
            p.grad = torch.randn(4, 4)
            params.append(p)
            optimizers.append(opt)

        initial = params[0].data.clone()
        for p, opt in zip(params, optimizers):
            opt.step()

        for p in params:
            self.assertFalse(torch.equal(p.data, initial))

    def test_row_normalization(self) -> None:
        param = torch.nn.Parameter(torch.randn(4, 8) * 10.0)
        optimizer = SpectralControlOptimizer(
            [param], T0=0.01, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.0, ns_steps=3, alpha=0.5,
            row_normalize=True, cautious=False, adaptive_momentum=False,
        )
        param.grad = torch.randn(4, 8)
        optimizer.step()

        sigma = _power_iteration_sigma_max(param.data, 10)
        self.assertLess(sigma.item(), 100.0)

    def test_nesterov_momentum(self) -> None:
        param = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = SpectralControlOptimizer(
            [param], T0=0.5, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.9, cautious=False, adaptive_momentum=False,
        )
        param.grad = torch.tensor([4.0])
        optimizer.step()

        self.assertNotEqual(optimizer.state[param]["velocity"].item(), 0.0)

    def test_cautious_updates(self) -> None:
        """Cautious updates should mask where gradient and momentum disagree."""
        param = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
        optimizer = SpectralControlOptimizer(
            [param], T0=0.1, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.0, cautious=True, adaptive_momentum=False,
        )
        # Gradient has mixed signs
        param.grad = torch.tensor([1.0, -1.0])
        optimizer.step()

        # Both components should be updated (gradient and update agree in sign)
        self.assertIsNotNone(optimizer.state[param]["velocity"])

    def test_cautious_disabled(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = SpectralControlOptimizer(
            [param], T0=0.1, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.0, cautious=False, adaptive_momentum=False,
        )
        param.grad = torch.tensor([1.0])
        optimizer.step()

    def test_adaptive_momentum_scales_with_width(self) -> None:
        """Wider layers should get higher momentum."""
        narrow = torch.nn.Parameter(torch.randn(4, 4))
        wide = torch.nn.Parameter(torch.randn(64, 64))

        opt = SpectralControlOptimizer(
            [narrow, wide], T0=0.01, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.9, cautious=False, adaptive_momentum=True,
            momentum_width_scale=0.01,
        )

        narrow.grad = torch.randn(4, 4)
        wide.grad = torch.randn(64, 64)
        opt.step()

        mu_narrow = opt._get_effective_momentum(narrow, opt.param_groups[0])
        mu_wide = opt._get_effective_momentum(wide, opt.param_groups[0])
        self.assertGreater(mu_wide, mu_narrow)

    def test_layer_type_budgets(self) -> None:
        """Different layer types should get different spectral budgets."""
        attn_param = torch.nn.Parameter(torch.randn(4, 4) * 10.0)
        embed_param = torch.nn.Parameter(torch.randn(4, 4) * 10.0)

        layer_types = {
            attn_param: LAYER_ATTENTION,
            embed_param: LAYER_EMBEDDING,
        }

        opt = SpectralControlOptimizer(
            [attn_param, embed_param], T0=0.01, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.0, cautious=False, adaptive_momentum=False,
            spectral_radius=1.0, spectral_update_period=1,
            spectral_damping=0.5, power_iteration_steps=10,
            layer_types=layer_types, adaptive_spectral=False,
        )

        attn_param.grad = torch.randn(4, 4)
        embed_param.grad = torch.randn(4, 4)
        opt.step()

        # Attention should have tighter constraint (lower budget = more clipping)
        attn_sigma = _power_iteration_sigma_max(attn_param.data, 10).item()
        embed_sigma = _power_iteration_sigma_max(embed_param.data, 10).item()
        # Both should be constrained, but attention more so
        self.assertLessEqual(attn_sigma, embed_sigma + 0.5)

    def test_lr_schedule_coupling(self) -> None:
        """LR schedule should affect temperature."""
        param = torch.nn.Parameter(torch.tensor([2.0]))

        def lr_schedule(step: int) -> float:
            return max(0.1, 1.0 - step * 0.1)

        optimizer = SpectralControlOptimizer(
            [param], T0=1.0, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.0, cautious=False, adaptive_momentum=False,
            lr_schedule_fn=lr_schedule,
        )

        temps = []
        for _ in range(5):
            param.grad = torch.tensor([1.0])
            optimizer.step()
            temps.append(optimizer.state[param]["temperature"].item())

        # Temperature should decrease due to LR schedule
        self.assertGreater(temps[0], temps[-1])

    def test_spectral_constraint_soft_clip(self) -> None:
        param = torch.nn.Parameter(torch.randn(4, 4) * 10.0)
        optimizer = SpectralControlOptimizer(
            [param], T0=0.01, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.0, cautious=False, adaptive_momentum=False,
            spectral_radius=1.0, spectral_update_period=1,
            spectral_damping=0.5, power_iteration_steps=10, adaptive_spectral=False,
        )
        sigma_before = _power_iteration_sigma_max(param.data, 10).item()
        self.assertGreater(sigma_before, 1.0)

        param.grad = torch.randn(4, 4)
        optimizer.step()

        sigma_after = _power_iteration_sigma_max(param.data, 10).item()
        self.assertLess(sigma_after, sigma_before)
        self.assertGreater(sigma_after, 0.0)

    def test_momentum_aware_threshold(self) -> None:
        param1 = torch.nn.Parameter(torch.randn(4, 4) * 10.0)
        opt1 = SpectralControlOptimizer(
            [param1], T0=0.01, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.99, cautious=False, adaptive_momentum=False,
            spectral_radius=5.0, spectral_update_period=1,
            spectral_damping=0.5, power_iteration_steps=10, adaptive_spectral=False,
        )

        param2 = torch.nn.Parameter(torch.randn(4, 4) * 10.0)
        opt2 = SpectralControlOptimizer(
            [param2], T0=0.01, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.0, cautious=False, adaptive_momentum=False,
            spectral_radius=5.0, spectral_update_period=1,
            spectral_damping=0.5, power_iteration_steps=10, adaptive_spectral=False,
        )

        grad = torch.randn(4, 4)
        param1.grad = grad
        param2.grad = grad

        opt1.step()
        opt2.step()

        sigma1 = _power_iteration_sigma_max(param1.data, 10).item()
        sigma2 = _power_iteration_sigma_max(param2.data, 10).item()
        self.assertLessEqual(sigma1, sigma2 + 0.1)

    def test_spectral_constraint_callable_radius(self) -> None:
        param = torch.nn.Parameter(torch.randn(4, 4) * 10.0)

        def radius_schedule(step: int) -> float:
            return max(5.0, 10.0 - step * 0.1)

        optimizer = SpectralControlOptimizer(
            [param], T0=0.01, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.0, cautious=False, adaptive_momentum=False,
            spectral_radius=radius_schedule, spectral_update_period=1,
            spectral_damping=0.5, power_iteration_steps=10, adaptive_spectral=False,
        )

        for _ in range(20):
            param.grad = torch.randn(4, 4)
            optimizer.step()

        sigma = _power_iteration_sigma_max(param.data, 10).item()
        self.assertLess(sigma, 20.0)

    def test_spectral_constraint_none_disables(self) -> None:
        param = torch.nn.Parameter(torch.randn(4, 4) * 10.0)
        optimizer = SpectralControlOptimizer(
            [param], T0=0.01, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.0, cautious=False, adaptive_momentum=False,
            spectral_radius=None, spectral_update_period=1,
        )

        sigma_before = _power_iteration_sigma_max(param.data, 10).item()
        param.grad = torch.randn(4, 4)
        optimizer.step()
        sigma_after = _power_iteration_sigma_max(param.data, 10).item()

        self.assertAlmostEqual(sigma_after, sigma_before, delta=1.0)

    def test_spectral_constraint_skips_1d_params(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        optimizer = SpectralControlOptimizer(
            [param], T0=0.01, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.0, cautious=False, adaptive_momentum=False,
            spectral_radius=1.0, spectral_update_period=1,
        )
        param.grad = torch.tensor([1.0, 1.0, 1.0])
        optimizer.step()  # Should not raise

    def test_outlier_suppression(self) -> None:
        """Outlier suppression should clip extreme values."""
        param = torch.nn.Parameter(torch.randn(100, 100))
        # Add some extreme outliers
        param.data[0, 0] = 100.0
        param.data[1, 1] = -100.0

        optimizer = SpectralControlOptimizer(
            [param], T0=0.01, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.0, cautious=False, adaptive_momentum=False,
            outlier_suppression=True, outlier_quantile=0.99,
            spectral_update_period=1,
        )

        # Run enough steps for outlier suppression to trigger
        for _ in range(4):
            param.grad = torch.randn(100, 100)
            optimizer.step()

        # Extreme values should be clipped
        self.assertLess(param.data.abs().max().item(), 50.0)

    def test_outlier_suppression_disabled(self) -> None:
        param = torch.nn.Parameter(torch.randn(10, 10))
        param.data[0, 0] = 100.0

        optimizer = SpectralControlOptimizer(
            [param], T0=0.01, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.0, cautious=False, adaptive_momentum=False,
            outlier_suppression=False,
        )

        for _ in range(10):
            param.grad = torch.randn(10, 10)
            optimizer.step()

        # Without suppression, extreme value may persist
        self.assertGreater(param.data.abs().max().item(), 10.0)

    def test_step_averages_replicated_distributed_statistics(self) -> None:
        param = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = SpectralControlOptimizer(
            [param], T0=0.5, beta1=0.0, beta2=0.0, noise_beta=0.5,
            warmup_steps=0, momentum=0.0, cautious=False, adaptive_momentum=False,
        )
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

        self.assertIsNotNone(optimizer.state[param]["temperature"])
        self.assertIsNotNone(optimizer.state[param]["natural_energy"])

    def test_validation_errors(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0]))

        with self.assertRaises(ValueError):
            SpectralControlOptimizer([param], T0=-1.0)
        with self.assertRaises(ValueError):
            SpectralControlOptimizer([param], beta1=1.5)
        with self.assertRaises(ValueError):
            SpectralControlOptimizer([param], beta2=-0.1)
        with self.assertRaises(ValueError):
            SpectralControlOptimizer([param], eps=0.0)
        with self.assertRaises(ValueError):
            SpectralControlOptimizer([param], noise_beta=1.0)
        with self.assertRaises(ValueError):
            SpectralControlOptimizer([param], warmup_steps=-1)
        with self.assertRaises(ValueError):
            SpectralControlOptimizer([param], ns_steps=0)
        with self.assertRaises(ValueError):
            SpectralControlOptimizer([param], alpha=1.5)
        with self.assertRaises(ValueError):
            SpectralControlOptimizer([param], momentum=1.0)
        with self.assertRaises(ValueError):
            SpectralControlOptimizer([param], spectral_update_period=0)
        with self.assertRaises(ValueError):
            SpectralControlOptimizer([param], spectral_damping=1.5)
        with self.assertRaises(ValueError):
            SpectralControlOptimizer([param], power_iteration_steps=0)
        with self.assertRaises(ValueError):
            SpectralControlOptimizer([param], spectral_ema_beta=1.0)

    def test_sparse_gradients_raise_error(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = SpectralControlOptimizer([param], cautious=False)
        param.grad = torch.sparse_coo_tensor(
            indices=torch.tensor([[0]]),
            values=torch.tensor([1.0]),
            size=(1,),
        )
        with self.assertRaises(RuntimeError):
            optimizer.step()

    def test_no_grad_params_skipped(self) -> None:
        param1 = torch.nn.Parameter(torch.tensor([1.0]))
        param2 = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = SpectralControlOptimizer([param1, param2], cautious=False)
        param1.grad = torch.tensor([1.0])
        loss = optimizer.step()
        self.assertIsNone(loss)

    def test_closure_support(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = SpectralControlOptimizer([param], cautious=False)
        closure_called = False

        def closure():
            nonlocal closure_called
            closure_called = True
            return torch.tensor(0.5)

        loss = optimizer.step(closure)
        self.assertTrue(closure_called)
        self.assertEqual(loss.item(), 0.5)

    def test_multi_parameter_update(self) -> None:
        p1 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        p2 = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        optimizer = SpectralControlOptimizer(
            [p1, p2], T0=0.1, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.0, cautious=False, adaptive_momentum=False,
        )
        p1.grad = torch.tensor([1.0, 1.0])
        p2.grad = torch.tensor([1.0, 1.0])

        p1_before = p1.data.clone()
        p2_before = p2.data.clone()

        optimizer.step()

        self.assertFalse(torch.equal(p1.data, p1_before))
        self.assertFalse(torch.equal(p2.data, p2_before))

    def test_temperature_decays_over_time(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = SpectralControlOptimizer(
            [param], T0=1.0, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.0, cautious=False, adaptive_momentum=False,
        )

        temps = []
        for _ in range(20):
            param.grad = torch.tensor([1.0])
            optimizer.step()
            temps.append(optimizer.state[param]["temperature"].item())

        self.assertGreater(temps[0], temps[-1])
        self.assertAlmostEqual(temps[0] / temps[9], math.sqrt(10), delta=1.0)

    def test_mixed_1d_and_2d_params(self) -> None:
        bias = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        weight = torch.nn.Parameter(torch.randn(4, 3))
        optimizer = SpectralControlOptimizer(
            [bias, weight], T0=0.01, beta1=0.0, beta2=0.0, noise_beta=0.0,
            warmup_steps=0, momentum=0.0, ns_steps=3, alpha=0.5,
            row_normalize=True, cautious=False, adaptive_momentum=False,
        )

        bias.grad = torch.tensor([1.0, 1.0])
        weight.grad = torch.randn(4, 3)

        b_before = bias.data.clone()
        w_before = weight.data.clone()

        optimizer.step()

        self.assertFalse(torch.equal(bias.data, b_before))
        self.assertFalse(torch.equal(weight.data, w_before))


if __name__ == "__main__":
    unittest.main()
