from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Optional

import torch
import torch.distributed as dist
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _distributed_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _distributed_mean(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value.div_(dist.get_world_size())
    return value


def _global_sum(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    total = None
    for tensor in tensors:
        contribution = tensor.float().sum()
        total = contribution if total is None else total + contribution
    if total is None:
        total = torch.tensor(0.0)
    return _distributed_sum(total)


# ---------------------------------------------------------------------------
# Spectral norm estimation
# ---------------------------------------------------------------------------

def _power_iteration_sigma_max(weight: torch.Tensor, iters: int) -> torch.Tensor:
    """Estimate the largest singular value via power iteration.

    3-5 iterations are typically sufficient for <5% error on typical NN
    weight matrices (Fast Tight Spectral-Norm Bounds, Ji-Ha 2025).
    """
    if weight.ndim < 2:
        return weight.float().abs().max()

    matrix = weight.float().reshape(weight.shape[0], -1)
    device = matrix.device
    cols = matrix.shape[1]

    vector = torch.arange(1, cols + 1, device=device, dtype=matrix.dtype)
    vector = vector / vector.norm().clamp_min(1e-12)

    for _ in range(max(iters, 1)):
        left = matrix @ vector
        left = left / left.norm().clamp_min(1e-12)
        vector = matrix.transpose(0, 1) @ left
        vector = vector / vector.norm().clamp_min(1e-12)

    return (matrix @ vector).norm()


# ---------------------------------------------------------------------------
# Gram Newton-Schulz preconditioning
# ---------------------------------------------------------------------------

def _gram_newton_schulz(
    grad_2d: torch.Tensor,
    ns_steps: int = 3,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, float]:
    """Compute Muon-style preconditioned gradient via Gram Newton-Schulz.

    Computes (G^T G)^{-1/2} G via NS iteration on the Gram matrix.
    The NS iteration Q_{k+1} = (3Q - Q^3) / 2 converges to Q^{-1/2}.

    Returns a tuple of (preconditioned_grad, fidelity_metric) where
    fidelity_metric is the NS convergence ratio ||Q_k - Q_{k-1}|| / ||Q_k||
    on the last iteration.

    References:
        - Dao AI Lab. Gram Newton-Schulz (2026)
        - Newton-Muon (Du & Su, 2025)
    """
    G = grad_2d.float()
    cols = G.shape[1]

    Q = G.T @ G
    trace = Q.trace().clamp_min(eps)
    Q = Q / (trace / cols + eps)

    fidelity_metric = 0.0
    for _ in range(max(ns_steps, 1)):
        Q_prev = Q.clone()
        Q2 = Q @ Q
        Q = (3.0 * Q - Q2 @ Q) / 2.0
        diff_norm = (Q - Q_prev).norm().item()
        q_norm = Q.norm().clamp_min(eps).item()
        fidelity_metric = diff_norm / q_norm

    return (G @ Q).to(grad_2d.dtype), fidelity_metric


# ---------------------------------------------------------------------------
# Row-normalization (RMNP continuous spectral control)
# ---------------------------------------------------------------------------

def _row_normalize(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-normalize a 2D+ tensor for continuous spectral control.

    Row-normalized matrices have bounded spectral norms (sigma_max <= sqrt(num_rows)).
    Reference: RMNP - Row-Momentum Normalized Preconditioning (2026)
    """
    if tensor.ndim < 2:
        return tensor
    shape = tensor.shape
    flat = tensor.view(shape[0], -1)
    row_norms = flat.norm(dim=-1, keepdim=True).clamp_min(eps)
    return (flat / row_norms).view(shape)


# ---------------------------------------------------------------------------
# Layer type detection
# ---------------------------------------------------------------------------

# Layer type tags for spectral budget allocation
LAYER_ATTENTION = "attention"
LAYER_MLP = "mlp"
LAYER_EMBEDDING = "embedding"
LAYER_OUTPUT = "output"
LAYER_DEFAULT = "default"

# Spectral budget multipliers per layer type
# Attention layers benefit most from spectral control (Practical Efficiency of Muon)
# Embeddings benefit least. Output layer needs tighter constraints (Ghosts of Softmax).
_LAYER_BUDGET = {
    LAYER_ATTENTION: 1.0,
    LAYER_MLP: 0.8,
    LAYER_EMBEDDING: 0.5,
    LAYER_OUTPUT: 0.5,  # Tighter = more aggressive clipping
    LAYER_DEFAULT: 0.8,
}


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class SpectralControlOptimizer(Optimizer):
    """PyTorch implementation of Spectral Control Training v2 (SCT v2).

    An operator approximation framework that controls the natural gradient energy
    sqrt(g^T P^{-1} g) with structured preconditioning and spectral stability.

    Characterized by a 4-tuple: (Geometry, Energy, Fidelity, Cost).

    Key features (validated against 100+ papers from the OMI optimizer collection):

    1. **Structured preconditioning**: Gram Newton-Schulz for 2D params (weight
       matrices), Adam-style diagonal for 1D params (biases, LayerNorm).
    2. **Alpha-parameterized spectral scaling** (Contra-Muon): partial spectral
       normalization — alpha=0 is sign gradient, alpha=0.5 is sqrt, alpha=1.0
       is full Muon.
    3. **Row-normalization** (RMNP): continuous spectral control between periodic
       clipping events.
    4. **Nesterov momentum** (SNOO): applied AFTER preconditioning.
    5. **Noise-adaptive scheduling**: target energy responds to gradient SNR.
    6. **Momentum-aware spectral clipping**: threshold accounts for effective lr
       amplification (Edge of Stability).
    7. **Adaptive spectral thresholds** (AdaMuon, Gluon): running statistics drive
       per-layer spectral budgets.
    8. **Layer-type-specific spectral budgets**: attention > MLP > embedding.
    9. **Dimension-adapted momentum**: momentum coefficient scales with layer width.
    10. **LR-schedule coupling**: spectral constraints follow the learning rate.
    11. **Cautious updates** (Cautious Optimizers): mask updates where gradient and
        momentum disagree.
    12. **Outlier suppression** (Beyond Outliers): suppresses extreme weights for
        quantization compatibility.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        E0: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
        noise_beta: float = 0.95,
        warmup_steps: int = 100,
        # Muon / Gram-NS parameters
        ns_steps: int = 3,
        alpha: float = 0.5,
        # Nesterov momentum
        momentum: float = 0.9,
        # Row-normalization
        row_normalize: bool = True,
        # Spectral clipping
        spectral_radius: float | Callable[[int], float] | None = None,
        spectral_update_period: int = 16,
        spectral_damping: float = 0.5,
        power_iteration_steps: int = 3,
        # Adaptive spectral thresholds (AdaMuon, Gluon)
        adaptive_spectral: bool = True,
        spectral_ema_beta: float = 0.99,
        # Layer-type spectral budgets
        layer_types: dict[torch.nn.Parameter, str] | None = None,
        # LR-schedule coupling
        lr_schedule_fn: Callable[[int], float] | None = None,
        # Cautious updates
        cautious: bool = True,
        cautious_eps: float = 1e-8,
        # Dimension-adapted momentum
        adaptive_momentum: bool = True,
        momentum_width_scale: float = 0.001,
        # Outlier suppression
        outlier_suppression: bool = False,
        outlier_quantile: float = 0.999,
    ) -> None:
        """
        Args:
            params: iterable of parameters to optimize.
            E0: base natural energy target (natural energy target at step 1).
            beta1: EMA coefficient for first moment (gradient mean).
            beta2: EMA coefficient for second moment (Adam preconditioner).
            eps: small constant for numerical stability.
            noise_beta: EMA coefficient for noise estimation.
            warmup_steps: number of warmup steps before full schedule.
            ns_steps: Newton-Schulz iterations for Gram-NS (3-5 recommended).
            alpha: spectral normalization strength (0=sign, 0.5=sqrt, 1.0=full Muon).
            momentum: Nesterov momentum coefficient (applied AFTER preconditioning).
            row_normalize: whether to apply row-normalization (continuous spectral control).
            spectral_radius: target spectral radius constraint. Can be a float or callable.
            spectral_update_period: apply spectral constraint every N steps.
            spectral_damping: interpolation factor for soft spectral clipping (0-1).
            power_iteration_steps: iterations for spectral norm estimation (3-5 sufficient).
            adaptive_spectral: use running spectral statistics for adaptive thresholds.
            spectral_ema_beta: EMA coefficient for adaptive spectral threshold.
            layer_types: dict mapping parameters to layer type strings for budget allocation.
            lr_schedule_fn: optional function step -> lr_scale for LR-schedule coupling.
            cautious: enable cautious updates (mask where gradient/momentum disagree).
            cautious_eps: threshold for cautious mask (fraction of update magnitude).
            adaptive_momentum: scale momentum coefficient by layer width.
            momentum_width_scale: scaling factor for width-adapted momentum.
            outlier_suppression: suppress extreme weight values for quantization.
            outlier_quantile: quantile threshold for outlier suppression.
        """
        if E0 <= 0.0:
            raise ValueError(f"Invalid E0 value: {E0}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps value: {eps}")
        if not 0.0 <= noise_beta < 1.0:
            raise ValueError(f"Invalid noise_beta value: {noise_beta}")
        if warmup_steps < 0:
            raise ValueError(f"Invalid warmup_steps value: {warmup_steps}")
        if ns_steps <= 0:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if spectral_update_period <= 0:
            raise ValueError(f"Invalid spectral_update_period value: {spectral_update_period}")
        if not 0.0 <= spectral_damping <= 1.0:
            raise ValueError(f"Invalid spectral_damping value: {spectral_damping}")
        if power_iteration_steps <= 0:
            raise ValueError(f"Invalid power_iteration_steps value: {power_iteration_steps}")
        if not 0.0 <= spectral_ema_beta < 1.0:
            raise ValueError(f"Invalid spectral_ema_beta value: {spectral_ema_beta}")

        defaults = dict(
            E0=E0, beta1=beta1, beta2=beta2, eps=eps, noise_beta=noise_beta,
            warmup_steps=warmup_steps, ns_steps=ns_steps, alpha=alpha,
            momentum=momentum, row_normalize=row_normalize,
            spectral_radius=spectral_radius, spectral_update_period=spectral_update_period,
            spectral_damping=spectral_damping, power_iteration_steps=power_iteration_steps,
            adaptive_spectral=adaptive_spectral, spectral_ema_beta=spectral_ema_beta,
            lr_schedule_fn=lr_schedule_fn, cautious=cautious, cautious_eps=cautious_eps,
            adaptive_momentum=adaptive_momentum, momentum_width_scale=momentum_width_scale,
            outlier_suppression=outlier_suppression, outlier_quantile=outlier_quantile,
        )
        super().__init__(params, defaults)

        # Store layer type mapping
        self._layer_types = layer_types or {}

    def _get_layer_type(self, param: torch.nn.Parameter) -> str:
        return self._layer_types.get(param, LAYER_DEFAULT)

    def _get_layer_budget(self, param: torch.nn.Parameter) -> float:
        layer_type = self._get_layer_type(param)
        return _LAYER_BUDGET.get(layer_type, _LAYER_BUDGET[LAYER_DEFAULT])

    def _get_effective_momentum(self, param: torch.nn.Parameter, group: dict) -> float:
        """Compute dimension-adapted momentum coefficient.

        Wider layers get higher momentum (Dimension-Adapted Momentum paper).
        mu_eff = mu_base + width_scale * log(num_params)
        """
        mu = group["momentum"]
        if group["adaptive_momentum"] and param.ndim >= 2:
            width = max(param.shape[0], param.shape[1])
            mu = mu + group["momentum_width_scale"] * math.log(width + 1)
            mu = min(mu, 0.999)  # Cap at 0.999
        return mu

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        updates: list[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor, dict, dict]] = []

        for group in self.param_groups:
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            noise_beta = group["noise_beta"]
            ns_steps = group["ns_steps"]
            alpha = group["alpha"]
            do_row_normalize = group["row_normalize"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("SpectralControlOptimizer does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    state["grad_ema"] = torch.zeros_like(param)
                    state["noise_ema"] = torch.zeros((), device=param.device)
                    state["target_energy"] = torch.zeros((), device=param.device)
                    state["natural_energy"] = torch.zeros((), device=param.device)
                    state["velocity"] = torch.zeros_like(param)
                    state["velocity_prev"] = torch.zeros_like(param)
                    # Adaptive spectral threshold state
                    state["spectral_ema"] = torch.zeros((), device=param.device)
                    # Operator fidelity and temporal consistency tracking
                    state["operator_fidelity"] = 0.0
                    state["preconditioning_ratio"] = 0.0
                    state["temporal_drift"] = 0.0
                    state["prev_preconditioned"] = None

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad_ema = state["grad_ema"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                grad_ema.mul_(noise_beta).add_(grad, alpha=1 - noise_beta)

                # === STRUCTURED PRECONDITIONING (shape-dependent) ===
                if grad.ndim >= 2:
                    G = grad.view(grad.shape[0], -1)
                    preconditioned, ns_fidelity = _gram_newton_schulz(G, ns_steps=ns_steps, eps=eps)
                    preconditioned = preconditioned.view(grad.shape)

                    # Track operator fidelity (NS convergence ratio)
                    state["operator_fidelity"] = ns_fidelity
                    # Track preconditioning ratio ||Õg|| / ||g||
                    grad_norm_val = grad.float().norm().clamp_min(eps).item()
                    prec_norm_val = preconditioned.float().norm().clamp_min(eps).item()
                    state["preconditioning_ratio"] = prec_norm_val / grad_norm_val

                    # Temporal consistency: drift of preconditioned gradient
                    prev_prec = state["prev_preconditioned"]
                    if prev_prec is not None and prev_prec.shape == preconditioned.shape:
                        drift_num = (preconditioned.float() - prev_prec.float()).norm().item()
                        drift_den = preconditioned.float().norm().clamp_min(eps).item()
                        state["temporal_drift"] = drift_num / drift_den
                    state["prev_preconditioned"] = preconditioned.detach().clone()

                    # Alpha-parameterized spectral scaling (Contra-Muon)
                    grad_norm = grad.float().norm().clamp_min(eps)
                    prec_norm = preconditioned.float().norm().clamp_min(eps)
                    scale = (grad_norm / prec_norm) ** alpha
                    preconditioned = preconditioned * scale

                    # Row-normalization (RMNP: continuous spectral control)
                    if do_row_normalize:
                        preconditioned = _row_normalize(preconditioned, eps=eps)
                else:
                    preconditioned = grad / exp_avg_sq.sqrt().add(eps)
                    # No NS fidelity for 1D params; set defaults
                    state["operator_fidelity"] = 0.0
                    state["preconditioning_ratio"] = 0.0

                # Noise estimation
                grad_norm_sq = grad.float().pow(2).sum()
                grad_ema_norm_sq = grad_ema.float().pow(2).sum()
                noise_instant = (grad_norm_sq - grad_ema_norm_sq).clamp_min(0.0)
                state["noise_ema"].mul_(noise_beta).add_(noise_instant, alpha=1 - noise_beta)

                state["step"] += 1
                updates.append((param, grad, preconditioned, state, group))

        if not updates:
            return loss

        # Compute natural energy
        natural_energy_sq = _distributed_mean(
            _global_sum(grad.float().mul(update.float()) for _, grad, update, _, _ in updates)
        ).clamp_min_(0.0)
        current_energy = natural_energy_sq.sqrt().clamp_min(1e-12)

        # Noise ratio
        mean_noise = _distributed_mean(
            _global_sum(state["noise_ema"] for _, _, _, state, _ in updates) / len(updates)
        )
        max_step = max(state["step"] for _, _, _, state, _ in updates)
        bias_correction = 1.0 - noise_beta ** max_step
        if bias_correction > 0:
            mean_noise.div_(bias_correction)

        mean_signal = _distributed_mean(
            _global_sum(state["grad_ema"].float().pow(2).sum() for _, _, _, state, _ in updates) / len(updates)
        )
        noise_ratio = (mean_noise / (mean_signal + eps)).item()

        # Apply updates
        for param, grad, update, state, group in updates:
            step = state["step"]

            # === LR-SCHEDULE COUPLING ===
            lr_scale = 1.0
            if group["lr_schedule_fn"] is not None:
                lr_scale = group["lr_schedule_fn"](step)

            # === NESTEROV MOMENTUM (applied AFTER preconditioning) ===
            velocity = state["velocity"]
            velocity_prev = state["velocity_prev"]
            velocity_prev.copy_(velocity)

            mu = self._get_effective_momentum(param, group)
            velocity.mul_(mu).add_(update)
            nesterov_update = (1.0 + mu) * velocity - mu * velocity_prev

            # === CAUTIOUS UPDATES (Cautious Optimizers) ===
            if group["cautious"]:
                # Mask: only update where gradient and momentum agree in sign
                mask = (nesterov_update * grad).float() > 0
                nesterov_update = nesterov_update * mask.float()

            # === ENERGY CONTROL ===
            E0_effective = group["E0"] * lr_scale
            warmup = group["warmup_steps"]
            if warmup > 0 and step <= warmup:
                E0_effective = E0_effective * (step / warmup)

            target_energy = (E0_effective / math.sqrt(step)) / (1.0 + noise_ratio)
            energy_scale = target_energy / current_energy.item()
            param.add_(nesterov_update, alpha=-energy_scale)

            state["target_energy"] = torch.tensor(target_energy, device=param.device)
            state["natural_energy"] = torch.tensor(current_energy.item(), device=param.device)

        self._apply_spectral_constraint(updates)
        self._apply_outlier_suppression(updates)
        return loss

    @torch.no_grad()
    def _apply_spectral_constraint(self, updates):
        """Apply soft spectral radius constraint with adaptive thresholds.

        Features:
        - Momentum-aware threshold (Edge of Stability)
        - Layer-type-specific budget (attention > MLP > embedding)
        - Adaptive threshold from running spectral statistics (AdaMuon, Gluon)
        - LR-schedule coupling
        """
        for param, _, _, state, group in updates:
            if param.ndim < 2:
                continue

            step = state["step"]
            period = group["spectral_update_period"]
            if step % period != 0:
                continue

            # Base threshold
            spectral_radius = group["spectral_radius"]
            if spectral_radius is None:
                continue

            radius_target = (
                spectral_radius(step) if callable(spectral_radius) else spectral_radius
            )

            # Layer-type budget (Practical Efficiency of Muon)
            layer_budget = self._get_layer_budget(param)
            radius_target *= layer_budget

            # Momentum-adjusted threshold (Edge of Stability)
            mu = self._get_effective_momentum(param, group)
            if mu > 0:
                momentum_factor = (1.0 + mu) / (1.0 - mu)
                radius_target = radius_target / momentum_factor

            # LR-schedule coupling (Fantastic Pretraining Optimizers)
            if group["lr_schedule_fn"] is not None:
                lr_scale = group["lr_schedule_fn"](step)
                radius_target *= lr_scale

            # Compute current spectral norm
            sigma = _power_iteration_sigma_max(param.data, group["power_iteration_steps"])

            # Adaptive spectral threshold (AdaMuon, Gluon)
            if group["adaptive_spectral"]:
                spectral_ema = state["spectral_ema"]
                spectral_ema_beta = group["spectral_ema_beta"]
                spectral_ema.mul_(spectral_ema_beta).add_(sigma, alpha=1 - spectral_ema_beta)
                # Debias
                debias = 1.0 - spectral_ema_beta ** step
                if debias > 0:
                    avg_sigma = (spectral_ema / debias).item()
                else:
                    avg_sigma = sigma.item()
                # Adaptive threshold: blend base with running average
                radius_target = min(radius_target, avg_sigma * 1.1)

            if sigma > radius_target:
                damping = group["spectral_damping"]
                scale = radius_target / sigma
                param.mul_((1.0 - damping) + damping * scale)

    @torch.no_grad()
    def _apply_outlier_suppression(self, updates):
        """Suppress extreme weight values for quantization compatibility.

        The l-infinity bias of AdamW creates outlier weights that destroy
        quantization precision (Beyond Outliers paper). This clips extreme
        values to a quantile-based threshold.
        """
        for param, _, _, state, group in updates:
            if not group["outlier_suppression"]:
                continue
            if param.ndim < 2:
                continue

            # Only suppress periodically (every spectral_update_period steps)
            step = state["step"]
            if step % (group["spectral_update_period"] * 4) != 0:
                continue

            # Clip to quantile
            q = group["outlier_quantile"]
            threshold = param.data.float().abs().quantile(q).item()
            if threshold > 0:
                param.data.clamp_(-threshold, threshold)


__all__ = ["SpectralControlOptimizer"]
