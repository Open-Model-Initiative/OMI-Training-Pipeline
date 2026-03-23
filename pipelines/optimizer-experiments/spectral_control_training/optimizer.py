from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Optional

import torch
import torch.distributed as dist
from torch.optim import Optimizer


def _distributed_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _global_sum(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    total = None
    for tensor in tensors:
        contribution = tensor.float().sum()
        total = contribution if total is None else total + contribution
    if total is None:
        total = torch.tensor(0.0)
    return _distributed_sum(total)


def _power_iteration_sigma_max(weight: torch.Tensor, iters: int) -> torch.Tensor:
    if weight.ndim < 2:
        return weight.float().abs().max()

    matrix = weight.float().reshape(weight.shape[0], -1)
    device = matrix.device
    vector = torch.arange(1, matrix.shape[1] + 1, device=device, dtype=matrix.dtype)
    vector = vector / vector.norm().clamp_min(1e-12)

    for _ in range(max(iters, 1)):
        left = matrix @ vector
        left = left / left.norm().clamp_min(1e-12)
        vector = matrix.transpose(0, 1) @ left
        vector = vector / vector.norm().clamp_min(1e-12)

    return (matrix @ vector).norm()


class SpectralControlOptimizer(Optimizer):
    """PyTorch implementation of Spectral Control Training v2 (SCT v2).

    The optimizer follows the SCT v2 control loop:
    1. precondition gradients,
    2. estimate gradient noise,
    3. measure natural gradient energy sqrt(g^T P^-1 g),
    4. scale updates to a target energy schedule,
    5. optionally enforce a spectral radius constraint.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        T0: float = 1e-2,
        beta2: float = 0.95,
        eps: float = 1e-8,
        noise_beta: float = 0.95,
        spectral_radius: float | Callable[[int], float] | None = None,
        spectral_update_period: int = 16,
        power_iteration_steps: int = 1,
    ) -> None:
        if T0 <= 0.0:
            raise ValueError(f"Invalid T0 value: {T0}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps value: {eps}")
        if not 0.0 <= noise_beta < 1.0:
            raise ValueError(f"Invalid noise_beta value: {noise_beta}")
        if spectral_update_period <= 0:
            raise ValueError(
                f"Invalid spectral_update_period value: {spectral_update_period}"
            )
        if power_iteration_steps <= 0:
            raise ValueError(
                f"Invalid power_iteration_steps value: {power_iteration_steps}"
            )

        defaults = dict(
            T0=T0,
            beta2=beta2,
            eps=eps,
            noise_beta=noise_beta,
            spectral_radius=spectral_radius,
            spectral_update_period=spectral_update_period,
            power_iteration_steps=power_iteration_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        updates: list[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor, dict, dict]] = []

        for group in self.param_groups:
            beta2 = group["beta2"]
            eps = group["eps"]
            noise_beta = group["noise_beta"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "SpectralControlOptimizer does not support sparse gradients"
                    )

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    state["grad_ema"] = torch.zeros_like(param)
                    state["noise_ema"] = torch.zeros((), device=param.device)
                    state["temperature"] = torch.zeros((), device=param.device)
                    state["natural_energy"] = torch.zeros((), device=param.device)

                exp_avg_sq = state["exp_avg_sq"]
                grad_ema = state["grad_ema"]

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                grad_ema.mul_(noise_beta).add_(grad, alpha=1 - noise_beta)

                preconditioned = grad / exp_avg_sq.sqrt().add(eps)
                noise_instant = grad.float().pow(2).sum() - grad_ema.float().pow(2).sum()
                noise_instant = noise_instant.clamp_min(0.0)
                state["noise_ema"].mul_(noise_beta).add_(
                    noise_instant, alpha=1 - noise_beta
                )

                state["step"] += 1
                updates.append((param, grad, preconditioned, state, group))

        if not updates:
            return loss

        natural_energy_sq = _global_sum(
            grad.float().mul(update.float()) for _, grad, update, _, _ in updates
        ).clamp_min_(0.0)
        current_temperature = natural_energy_sq.sqrt().clamp_min(1e-12)
        mean_noise = (
            _global_sum(state["noise_ema"] for _, _, _, state, _ in updates) / len(updates)
        )

        for param, _, update, state, group in updates:
            step = state["step"]
            target_temperature = (group["T0"] / math.sqrt(step)) / (1.0 + mean_noise.item())
            temperature_scale = target_temperature / current_temperature.item()
            param.add_(update, alpha=-temperature_scale)
            state["temperature"] = torch.tensor(target_temperature, device=param.device)
            state["natural_energy"] = torch.tensor(
                current_temperature.item(), device=param.device
            )

        self._apply_spectral_constraint(updates)
        return loss

    @torch.no_grad()
    def _apply_spectral_constraint(self, updates):
        for param, _, _, state, group in updates:
            spectral_radius = group["spectral_radius"]
            if spectral_radius is None:
                continue

            step = state["step"]
            period = group["spectral_update_period"]
            if step % period != 0:
                continue

            radius_target = (
                spectral_radius(step) if callable(spectral_radius) else spectral_radius
            )
            sigma = _power_iteration_sigma_max(param.data, group["power_iteration_steps"])
            if sigma > radius_target:
                param.mul_(radius_target / sigma)


__all__ = ["SpectralControlOptimizer"]
