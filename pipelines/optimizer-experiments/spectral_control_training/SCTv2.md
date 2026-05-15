# Spectral Control Training v2: An Operator Approximation Framework for LLM Optimization

Large-scale model training is often framed as designing better optimizers.
However, under stochastic gradients, preconditioning, and low precision, this view becomes insufficient.

A more precise formulation is:

> **LLM training is an operator approximation problem in a preconditioned geometry.**

Each optimizer step applies an approximate operator $\tilde{O}$ to the parameters. The quality of training depends on how faithfully this operator approximates the ideal spectral operator $O^*$, and on four measurable properties of the operator: its geometry, its energy, its fidelity, and its cost.

In this post, we present **Spectral Control Training v2 (SCT v2)**—a framework that unifies:

- structured preconditioning (Muon-style for matrices, Adam-style for vectors)
- noise-adaptive scheduling
- spectral stability (row-normalization + periodic spectral clipping)

into an **operator approximation framework** built around a 4-tuple: **(Geometry, Energy, Fidelity, Cost)**.

---

# 1. The Four Pillars: Geometry, Energy, Fidelity, Cost

Every approximate spectral optimizer can be characterized by four quantities. These form the backbone of our analysis.

## 1.1 Geometry: The Preconditioner $P$

Consider a second-order approximation:

```math
L(\theta + \Delta) \approx L(\theta) + g^T \Delta + \frac{1}{2}\Delta^T H \Delta
```

The ideal update is:

```math
\Delta^* = -H^{-1} g
```

In practice, we instead use:

```math
\Delta = -P^{-1} g
```

where $P$ approximates curvature. Optimization does not happen in Euclidean space, but in the geometry induced by $P$.

Define the **natural metric** (Fisher-Rao metric when $P = F$, the Fisher Information Matrix):

```math
\lVert g \rVert_{P^{-1}}^2 = g^T P^{-1} g
```

For matrix-shaped parameters, the optimal preconditioner is $(G^T G)^{-1/2}$ (Newton-Muon, Du & Su 2025),
not a diagonal approximation. This equalizes the effective step size across all singular directions.

The natural gradient direction $P^{-1}g$ is the steepest descent direction in the Riemannian manifold induced by $P$.
The quantity $g^T P^{-1} g$ measures the **information-theoretic distance** moved per step
(Amari, 1998; Martens, 2014).

**Wasserstein Flow Interpretation**: Recent work (Peyré, 2025) shows that Muon-style optimization can be interpreted as
a Wasserstein flow on the space of spectral measures of weight matrices. The optimizer
transports the singular value distribution toward a spectral equilibrium. SCT v2 makes this implicit spectral
transport an explicit, controllable mechanism.

## 1.2 Energy: The Natural Update Energy

The **core controlled quantity** in SCT v2 is the natural update energy:

```math
E_t = g_t^T P_t^{-1} g_t
```

This single scalar captures the information-theoretic impact of the update at step $t$. It is THE thing being controlled.

$E_t$ replaces:
- learning rate as the primary step-size control
- $\lVert\Delta W\rVert$ heuristics (which measure displacement, not information)
- Hyperball scaling (which constrains parameter norms, not update energy)

**Theoretical grounding**: In natural gradient descent, $g^T F^{-1} g$ (where $F$ is the
Fisher information matrix) measures the KL divergence between the current model and the
model after one update step. By controlling $E_t$, we directly control
the information-theoretic impact of each update.

**Connection to implicit regularization**: Smith & Dherin (2021) showed that
the discretization of gradient flow introduces regularization proportional to
the update energy:

```math
L_{\text{discrete}} = L + \frac{\eta}{2} g^T P^{-1} g + \text{higher-order terms}
```

By controlling $E_t = g^T P^{-1} g$, we are controlling
**the strength of this implicit regularization** at each step.

## 1.3 Fidelity: How Well Does $\tilde{O}$ Approximate $O^*$?

Every practical preconditioner is an approximation. Gram-NS approximates $(G^T G)^{-1/2}$ via a finite number of Newton-Schulz iterations. Truncated NS, QR-based methods, and diagonal approximations are all imperfect. What matters is the **operator fidelity**:

```math
\varepsilon_{\text{fidelity}} = \lVert \tilde{O} - O^* \rVert
```

where $\tilde{O}$ is the actual approximate operator applied (e.g., Gram-NS with 3 iterations) and $O^*$ is the ideal spectral operator (exact Newton-Muon).

This error has two sources:

1. **Approximation error**: Finite NS iterations, truncation, quantization of the preconditioner itself.
2. **Stochastic error**: Mini-batch gradients introduce noise into $P$ — the preconditioner is estimated from noisy data.

**Tracking fidelity**: The Gram matrix eigenvalues (available at zero extra cost from Gram-NS) provide a direct window into operator fidelity. After $k$ NS iterations, the approximation error on the inverse square root is $O(c^{2^k})$ where $c$ is related to the condition number. For 3 iterations with condition number < 2, this error is negligible. For ill-conditioned problems, more iterations or regularization may be needed.

**Practical monitoring**: Track $\lVert \tilde{O} g \rVert / \lVert g \rVert$ over training. A drifting ratio signals that the operator approximation is degrading — possibly due to changing spectral structure of the gradients.

## 1.4 Cost: Computational and Memory Overhead

| Component | Compute Cost | Memory Cost |
| --- | --- | --- |
| Gram-NS (3 iterations) | ~1 extra matmul per matrix param/step | Gram matrix $d_{out} \times d_{out}$ |
| Adam diagonal (vectors) | Negligible | Second moment buffer |
| Row-normalization | $O(n)$ per layer | In-place |
| Spectral clipping (periodic) | Power iteration every $k$ steps | Negligible |
| Energy estimation | 1 all-reduce per step | Negligible |

Total overhead: ~5–10% of a forward+backward pass. This is the cost budget we are operating within.

---

# 2. Theoretical Foundation

## Core Insight

> **The correct notion of "step size" is not $\lVert\Delta\rVert$, but $E_t = g^T P^{-1} g$.**

This is well-grounded in information geometry. The update

```math
\Delta = -\alpha_t \cdot P^{-1} g
```

where $\alpha_t$ is chosen to control the natural update energy $E_t$ and $P$ is a
structured preconditioner that differs by parameter shape.

## Edge of Stability Perspective

Gradient descent on smooth nonconvex losses converges to a "sharpness-aware" regime
where the step size is approximately $2/\lambda_{\max}(H)$ (Cohen et al.). In the
stochastic case, gradient noise pushes this threshold lower. SCT v2's momentum-aware
spectral constraint accounts for this by using the effective learning rate
$\eta \cdot (1+\beta)/(1-\beta)$ in the threshold computation.

---

# 3. SCT v2 Algorithm

## Algorithm Pseudocode

```python
# SCT v2 — Operator Approximation Framework

for step t:
    for param in parameters:
        g = grad(param)

        # === PRECONDITIONING (shape-dependent geometry) ===
        if g.ndim >= 2:
            # Matrix params: Muon-style Gram Newton-Schulz
            G = g.view(g.shape[0], -1)
            Q = G.T @ G                              # Gram matrix
            for _ in range(ns_steps):                 # typically 3-5 steps
                Q = (3*Q - Q @ Q @ Q) / 2            # NS iteration
            u = (G @ Q).view(g.shape)

            # Alpha-parameterized spectral scaling (Contra-Muon)
            # alpha=0: sign gradient, alpha=0.5: sqrt, alpha=1: full Muon
            u = u * (g.norm() / (u.norm() + eps)) ** alpha

            # Row-normalization (RMNP: continuous spectral control)
            row_norms = u.norm(dim=-1, keepdim=True).clamp_min(eps)
            u = u / row_norms
        else:
            # Vector params: Adam-style diagonal preconditioner
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * g**2
            u = g / (exp_avg_sq.sqrt() + eps)

        # === NESTEROV MOMENTUM (applied AFTER preconditioning) ===
        v = momentum * v + u
        update = (1 + momentum) * v - momentum * v_prev
        v_prev = v

        # === ENERGY CONTROL ===
        E_current = g^T u                            # natural update energy

        noise_ratio = grad_variance / (grad_signal^2 + eps)
        E_target = E0 / (sqrt(t + warmup) * (1 + noise_ratio))

        update *= E_target / (E_current + eps)

        # === APPLY UPDATE ===
        param -= update

    # === SPECTRAL STABILITY (periodic, momentum-aware) ===
    if step % k == 0:
        for param in matrix_parameters:
            sigma = estimate_sigma_max(param)
            # Momentum-adjusted threshold
            effective_lr = lr * (1 + momentum) / (1 - momentum)
            R_t = base_radius / effective_lr
            if sigma > R_t:
                # Soft clipping (damped projection)
                param *= (1 - damping) + damping * R_t / sigma
```

## Design Rationale

The key novelty is **controlling the energy injected into the system per step**,
rather than controlling raw parameter displacement $\lVert\Delta W\rVert$.

The pseudocode above is an implementation. The underlying theory is:

1. Choose a preconditioner $P$ (the geometry)
2. Compute $E_t = g^T P^{-1} g$ (the energy)
3. Ensure $\tilde{O}$ approximates $O^*$ well (the fidelity)
4. Do all of this within a bounded compute budget (the cost)

---

# 4. Core Components

## 4.1 Structured Preconditioner (Shape-Dependent)

| Parameter Shape | Preconditioner $P$                | Rationale                                    |
| --------------- | --------------------------------- | -------------------------------------------- |
| 1D (bias, LN)   | $\text{diag}(\mathbb{E}[g^2])$   | Adam-style diagonal (sufficient for vectors) |
| 2D (weight)     | $(G^T G)^{-1/2}$ via Gram-NS     | Optimal for matrix structure (Newton-Muon)   |
| Embedding       | $\text{diag}(\mathbb{E}[g^2])$   | Adam-style (rows are independent)            |

**Why Gram Newton-Schulz?** The Gram matrix $G^T G$ is symmetric PSD and smaller than
$G$ itself (d_out × d_out vs d_out × d_in). NS iteration on the Gram matrix computes
$(G^T G)^{-1/2}$ efficiently, and the spectral information (singular values) is directly
available as eigenvalues of $G^T G$ — enabling spectral monitoring at zero extra cost
(Dao AI Lab, 2026).

**Alpha-parameterized control** (Contra-Muon): Full Muon normalization ($\sigma \to 1$)
is too aggressive. Partial normalization $\sigma^\alpha$ with $\alpha \approx 0.5$ preserves
spectral diversity while still conditioning the update. This is cheaper (fewer NS iterations
needed) and more robust.

## 4.2 Natural Update Energy (Key Variable)

```math
E_t = g^T P^{-1} g
```

This is the single scalar that SCT v2 controls. Everything else — the schedule, the noise adaptation, the spectral constraints — serves to keep $E_t$ on its target trajectory.

**Theoretical grounding**: In natural gradient descent, $g^T F^{-1} g$ (where $F$ is the
Fisher information matrix) measures the KL divergence between the current model and the
model after one update step. By controlling $E_t$, we directly control
the information-theoretic impact of each update.

## 4.3 Operator Fidelity Tracking

Every approximate preconditioner introduces error between the applied operator $\tilde{O}$ and the ideal operator $O^*$. We track this in three ways:

**Approximation error from finite iterations**: For Gram-NS with $k$ iterations, the error in approximating $(G^T G)^{-1/2}$ is $O(c^{2^k})$ where $c$ depends on the condition number of $G^T G$. With 3 iterations and condition number < 2, this is < 1%. Monitoring the convergence of the NS iteration (change in $Q$ between iterations) gives a per-step fidelity estimate.

**Operator drift from stochastic estimation**: The Gram matrix $G^T G$ is estimated from mini-batch gradients. Different batches yield different $P$, so the applied operator $\tilde{O}$ drifts across steps even at the same point in parameter space.

**Quantization-induced fidelity loss**: When the preconditioner or gradients are in low precision, quantization noise degrades operator fidelity. This is additive to the approximation error.

**Practical recommendation**: Log $\lVert Q_{k} - Q_{k-1} \rVert / \lVert Q_k \rVert$ (NS convergence ratio) and $\lVert \tilde{O} g \rVert / \lVert g \rVert$ (effective preconditioning ratio) during training. Stable values indicate good operator fidelity.

## 4.4 Temporal Consistency

Warm-started and iterative preconditioners can drift over time. Define the **temporal operator drift**:

```math
\varepsilon_{\text{temporal}} = \lVert \tilde{O}_t - \tilde{O}_{t-1} \rVert
```

This measures how much the applied operator changes between consecutive steps. Sources of temporal drift:

1. **Changing gradient distribution**: As parameters evolve, the Gram matrix $G^T G$ changes, so the preconditioner changes even with the same algorithm.
2. **EMA state drift**: Adam-style second moment estimates track a running average that may lag behind the current gradient distribution.
3. **Warm-start bias**: When reusing $Q$ from a previous step as an NS initialization, convergence is faster but the starting point may be biased toward the previous operator.

**Why it matters**: Large temporal drift ($\varepsilon_{\text{temporal}} \gg \varepsilon_{\text{fidelity}}$) means the optimizer is effectively applying a different operator each step. This can cause oscillation or instability even if each individual operator is a good approximation.

**Practical mitigation**: SCT v2's row-normalization acts as a temporal smoother — it constrains the spectral structure of the update, reducing the impact of operator drift. The periodic spectral clipping also provides a "reset" that prevents drift from accumulating.

**Monitoring**: Track $\varepsilon_{\text{temporal}}$ by periodically computing $\lVert \tilde{O}_t g - \tilde{O}_{t-1} g \rVert / \lVert g \rVert$ on a fixed reference batch. Rising temporal drift signals that the preconditioner is unstable.

## 4.5 Noise Estimator

```python
# Per-parameter gradient variance estimate
# Var(g) = E[||g||^2] - ||E[g]||^2
noise_est = EMA(||g||^2) - ||EMA(g)||^2

# Scale-invariant noise ratio
noise_ratio = noise_est / (||EMA(g)||^2 + eps)
```

**Why scale-invariant?** Raw noise variance grows with gradient magnitude.
The ratio `noise_est / signal_est` captures the **signal-to-noise ratio**,
which is the quantity that actually affects convergence (McCandlish et al., 2018).

**Bias correction**: EMA estimates are biased toward zero at initialization.
We apply a debiasing correction: `noise_est / (1 - beta^t)`.

**Noise-coupled constraints**: When gradient noise is high, spectral constraints
should be tighter (Generalized Gradient Norm Clipping framework). This prevents
noise-amplifying updates from destabilizing training.

## 4.6 Energy Schedule

```math
E_{\text{target}}(t) = \frac{E_0}{\sqrt{t + t_{\text{warmup}}}} \cdot \frac{1}{1 + \text{noise\_ratio}}
```

**Warmup rationale**: At the start of training, gradients are large and noisy.
A warmup phase prevents the optimizer from taking overly aggressive steps
before the preconditioner statistics have stabilized.

**Noise-adaptive rationale**: When gradient noise is high (low SNR), the optimizer
reduces target energy to avoid amplifying noise. When gradients are clean (high SNR),
it can afford larger energy budgets. This is analogous to the "noise-aware" scheduling
in Smith et al. (2017).

## 4.7 Row-Normalization (Continuous Spectral Control)

Between periodic spectral clipping events, row-normalization provides **continuous**
spectral control at negligible cost (RMNP, 2026):

```python
row_norms = W.norm(dim=-1, keepdim=True)
W = W / row_norms.clamp_min(eps)
```

**Why row-normalization?** Row-normalized weight matrices have bounded spectral norms
($\sigma_{\max} \le \sqrt{\text{num\_rows}}$). This prevents spectral blowup between
clipping events without the cost of power iteration.

## 4.8 Spectral Stability (Momentum-Aware)

The spectral constraint must account for **momentum amplification** (Edge of Stability,
Momentum paper):

```python
# The effective step size with momentum is much larger than the nominal lr
effective_lr = lr * (1 + momentum) / (1 - momentum)

# Stability threshold: sigma_max < 2 / effective_lr
R_t = 2.0 / (effective_lr * sharpness_estimate)
```

**Soft clipping** (damped projection) avoids gradient discontinuities:

```python
W_new = (1 - damping) * W + damping * W * (R_t / sigma)
```

**Output layer special handling** (Ghosts of Softmax): The cross-entropy loss has
hidden singularities that create instability barriers near softmax saturation.
The output layer needs 2× more aggressive spectral clipping than hidden layers.

---

# 5. Systems Design

## 5.1 Distributed Consistency

We need global reductions for consistent energy measurement:

```python
# Energy must be globally consistent across all ranks
E_current = all_reduce(sum(g * u)) / world_size

# Noise ratio is per-rank (gradient noise is local)
noise_ratio = local_noise_estimate
```

**Key insight**: The natural energy $g^T P^{-1} g$ must be computed globally because
it determines the scaling factor applied to all parameters. If each rank computed
a different scale, the parameters would diverge across ranks.

The noise ratio is local because gradient noise is an inherent property of each
rank's data shard — averaging it across ranks would mask rank-specific noise patterns.

## 5.2 Efficient Implementation

Reuse optimizer state to avoid extra memory:

```python
# For matrix params: Gram-NS preconditioning
Q = G.T @ G          # Gram matrix (recomputed each step)
for _ in range(3):   # 3 NS iterations sufficient (Fast Spectral-Norm Bounds)
    Q = (3*Q - Q @ Q @ Q) / 2
u = G @ Q            # Preconditioned gradient

# For vector params: Adam-style
u = m / sqrt(v)      # Standard Adam preconditioned gradient
```

**Compute cost**: Gram-NS adds ~1 extra matmul per matrix parameter per step.
Row-normalization is O(n) per layer. Total overhead: ~5-10% of a forward+backward pass.

## 5.3 No Extra Instability

Unlike Hyperball (which normalizes $u$ directly):

- SCT v2 only **scales** $u$ by a scalar factor
- The direction of the update is preserved
- Spectral constraints are applied as **soft clipping** (damped projection),
  not hard clipping, to avoid gradient discontinuities
- Row-normalization provides continuous control between clipping events

---

# 6. Quantization-Aware Training

Low precision introduces noise:

```math
g \rightarrow g + \epsilon_{\text{quant}}
```

## SCT v2 Advantage

SCT v2 controls:

```math
E_t = g^T P^{-1} g
```

This ensures:

> **Update energy remains stable under quantization noise.**

When quantization noise $\epsilon_{\text{quant}}$ increases the apparent gradient variance,
the noise estimator detects this and reduces the target energy accordingly.

Quantization also affects **operator fidelity** — the preconditioner itself may be computed in low precision, introducing additional $\varepsilon_{\text{fidelity}}$. This is a reason to prefer Gram-NS over more expensive preconditioners: Gram-NS is numerically robust because the NS iteration is self-correcting (it converges to the fixed point regardless of small perturbations in the input).

## Adaptive Schedule

```python
# Quantization noise inflates the noise ratio
noise_ratio = (grad_variance + quant_variance) / (signal^2 + eps)
E_target = base_E / (1 + noise_ratio)
```

---

# 7. Experimental Plan

## Baselines

- AdamW (standard baseline)
- Muon (structured preconditioning)
- SOAP (KL-based preconditioning)
- Hyperball (weight-space normalization)
- SCT v2 (our method)

## Metrics

| Metric | What It Measures | Pillar |
| --- | --- | --- |
| Loss vs tokens | Sample efficiency | Overall |
| Gradient noise scale (McCandlish et al.) | Stochastic regime | Energy |
| Natural update energy $g^T P^{-1} g$ | Our controlled variable | Energy |
| Spectral norm $\lambda_{\max}(W)$ over training | Spectral stability | Fidelity |
| Singular value distribution | Spectral diversity | Fidelity |
| NS convergence ratio | Operator approximation quality | Fidelity |
| Operator drift $\varepsilon_{\text{temporal}}$ | Temporal consistency | Fidelity |
| Implicit regularization strength (Smith & Dherin metric) | Regularization | Energy |
| Downstream task generalization | Not just pretraining loss | Overall |
| Wall-clock time per step | Compute overhead | Cost |

## Expected Results

| Method    | Behavior                                    |
| --------- | ------------------------------------------- |
| AdamW     | stable but suboptimal convergence           |
| Muon      | fast convergence, implicit spectral control |
| SOAP      | good conditioning, high memory cost         |
| Hyperball | stable but rigid (over-constraining)        |
| SCT v2    | fast + stable + spectrally-aware            |

## Ablation Studies

1. **Alpha parameterization**: alpha = 0 (sign) vs 0.5 (sqrt) vs 1.0 (full Muon)
2. **With/without row-normalization**: isolate continuous spectral control
3. **With/without noise adaptation**: isolate noise-aware scheduling
4. **With/without momentum-aware threshold**: isolate momentum correction
5. **NS iterations**: 2 vs 3 vs 5 iterations (convergence vs compute tradeoff — fidelity vs cost)
6. **Output layer clipping aggressiveness**: 1x vs 2x vs 4x
7. **Spectral constraint damping**: hard clip (1.0) vs soft clip (0.5) vs gentle (0.2)

---

# 8. Relation to Prior Concepts

## Information Geometry Perspective

SCT v2 improves:

> signal quality per computation step

The structured preconditioner (Gram-NS) approximates the Fisher information matrix
for matrix-shaped parameters. This is natural gradient descent on the Stiefel manifold
(Newton-Muon, Du & Su 2025), with the added benefit of explicit spectral control.

**Mousse correction**: The Gram-NS preconditioner assumes isotropic activations.
When activations are non-isotropic, curvature-aware corrections are needed (Mousse,
Zhang et al. 2025). SCT v2's row-normalization partially addresses this by
normalizing the spectral structure.

## Gradient Dynamics Perspective

- Updates are operators on parameter space
- Stability depends on the operator's spectral properties
- The fidelity of the operator approximation determines convergence quality

SCT v2 ensures:
- controlled update energy (bounded operator norm in preconditioned metric)
- stable operator dynamics (spectral constraints prevent explosion)
- spectral diversity preservation (alpha-parameterization prevents over-normalization)
- tracked operator fidelity (NS convergence monitoring)
- bounded temporal drift (row-normalization as temporal smoother)

**Connection to implicit regularization**: Smith & Dherin (2021) showed that
the discretization of gradient flow introduces regularization proportional to
the update energy. In preconditioned geometry, this becomes:

```math
\text{implicit regularizer} \propto \frac{\eta}{2} g^T P^{-1} g
```

By controlling $E_t = g^T P^{-1} g$, SCT v2 directly controls the implicit regularization
strength — a theoretically grounded connection.

## Edge of Stability Perspective

Gradient descent on smooth nonconvex losses converges to a "sharpness-aware" regime
where the step size is approximately $2/\lambda_{\max}(H)$ (Cohen et al.). In the
stochastic case, gradient noise pushes this threshold lower. SCT v2's momentum-aware
spectral constraint accounts for this by using the effective learning rate
$\eta \cdot (1+\beta)/(1-\beta)$ in the threshold computation.

---

# 9. Unified View

We now unify training as operator approximation:

```math
\Delta = -\alpha_t \cdot \tilde{O}(g)
\quad \text{s.t.} \quad
\begin{cases}
E_t = g^T P^{-1} g \approx E_{\text{target}}(t) & \text{(energy)} \\
\lVert \tilde{O} - O^* \rVert \le \varepsilon_{\text{fidelity}} & \text{(fidelity)} \\
\lambda_{\max}(W) \le R(t, \beta, \text{noise}) & \text{(stability)} \\
\text{compute} \le \text{budget} & \text{(cost)}
\end{cases}
```

## Components Mapped to the 4-Tuple

| Element | Pillar | Role |
| --- | --- | --- |
| $P^{-1}$ (Gram-NS) | Geometry | Structured preconditioner for matrices |
| $P^{-1}$ (Adam diag) | Geometry | Diagonal preconditioner for vectors |
| $E_t = g^T P^{-1} g$ | Energy | Information-theoretic step size |
| $E_{\text{target}}(t)$ | Energy | Noise-adaptive schedule |
| $\alpha$ | Geometry | Spectral normalization strength |
| NS convergence ratio | Fidelity | Operator approximation quality |
| $\varepsilon_{\text{temporal}}$ | Fidelity | Temporal operator drift |
| Row-normalization | Fidelity | Continuous spectral control (temporal smoother) |
| $R(t, \beta, \text{noise})$ | Energy/Stability | Momentum-aware spectral bound |
| Gram-NS compute cost | Cost | ~1 extra matmul per matrix param/step |
| Memory for $Q$ | Cost | $d_{out} \times d_{out}$ per matrix |

## Final Insight

> **Optimization is not about moving parameters.**
>
> **It is about approximating an ideal spectral operator, subject to constraints on energy, fidelity, and cost.**

The key contributions are:
1. **Structured preconditioning**: Gram-NS for matrices, Adam for vectors (geometry)
2. **Alpha-parameterized spectral control**: partial normalization preserves spectral diversity (geometry)
3. **Natural update energy as control variable**: replacing learning rate with $E_t = g^T P^{-1} g$ (energy)
4. **Operator fidelity tracking**: monitoring $\lVert \tilde{O} - O^* \rVert$ via NS convergence (fidelity)
5. **Temporal consistency**: tracking operator drift $\varepsilon_{\text{temporal}}$ (fidelity)
6. **Noise-adaptive scheduling**: step size responds to gradient SNR (energy)
7. **Dual spectral control**: row-normalization (continuous) + spectral clipping (periodic) (fidelity)
8. **Momentum-aware thresholds**: accounts for effective lr amplification (stability)
9. **Preconditioner-agnostic framework**: works with any $P$ (Adam, Muon, SOAP, etc.) (geometry)

SCT v2 provides a unified framework for analyzing and designing approximate spectral optimizers, decomposing the problem into four measurable pillars: Geometry, Energy, Fidelity, and Cost.

---

# 10. References

### Optimization and Preconditioning

- Amari. *Natural Gradient Works Efficiently in Learning* (1998) — foundational natural gradient theory
- Kingma & Ba. *Adam: A Method for Stochastic Optimization* (2014)
- Loshchilov & Hutter. *Decoupled Weight Decay Regularization* (2017)
- Martens. *Hessian-Free Optimization* (2010)
- Grosse & Martens. *K-FAC* (2016)
- Chou. *Correction of Decoupled Weight Decay* (2025) — weight decay ordering matters

---

### Muon Family

- Peyré. *Muon Dynamics as a Spectral Wasserstein Flow* (2025) — spectral transport theory
- Zhang et al. *Mousse: Rectifying the Geometry of Muon with Curvature-Aware Preconditioning* (2025)
- Du & Su. *The Newton-Muon Optimizer* (2025) — Muon = Newton on Stiefel manifold
- Dao AI Lab. *Gram Newton-Schulz: A Fast, Hardware-Aware NS Algorithm* (2026)
- nilin. *Contra-Muon* (2026) — alpha-parameterized spectral control
- Kallusky et al. *SNOO: Step-K Nesterov Outer Optimizer* (2025) — momentum after preconditioning
- *Transport Muon: Beating Muon in 1 Newton Step* (2025)

---

### Spectral Methods and Edge of Stability

- Miyato et al. *Spectral Normalization for GANs* (2018) — spectral constraint for stability
- Cohen et al. *SGD at the Edge of Stability* — sharpness threshold theory
- *The Stochastic Sharpness Gap* — noise pushes sharpness threshold lower
- *Momentum Further Constrains Sharpness at Edge of Stochastic Stability* — momentum amplification
- *Spectra: Rethinking Optimizers for LLMs Under Spectral Anisotropy* (2025)
- *Ghosts of Softmax: Complex Singularities That Limit Safe Step Sizes* — output layer instability
- *Fast Tight Spectral-Norm Bounds* — 3-5 power iterations sufficient
- *ARO: A New Lens On Matrix Optimization For Large Models*
- *RMNP: Row-Momentum Normalized Preconditioning* — cheap continuous spectral control
- *Nora: Normalized Orthogonal Row Alignment* — spectral flattening
- *Sharp Capacity Scaling of Spectral Optimizers* — spectral clipping improves capacity

---

### Noise and Scaling

- McCandlish et al. *An Empirical Model of Large-Batch Training* (2018) — gradient noise scale
- Smith et al. *Don't Decay the Learning Rate, Increase the Batch Size* (2017)
- Smith & Dherin. *Implicit Gradient Regularization* (ICLR 2021) — discretization as implicit regularization
- Khadir & Dherin. *What is the relationship between the learning rate and generalization error?* (2023)
- Kaplan et al. *Scaling Laws for Neural Language Models* (2020)
- Hoffmann et al. *Chinchilla* (2022)
- *Deriving Hyperparameter Scaling Laws via Modern Optimization Theory*
- *Generalized Gradient Norm Clipping & Non-Euclidean (L0,L1)-Smoothness*

---

### Manifold and Geometric Methods

- *Rethinking Language Model Scaling under Transferable Hypersphere Optimization* (2025)
- *Demystifying Manifold Constraints in LLM Pre-training* (2025)
- *Transformers as Constrained Optimization* (Ji-Ha)
- *Understanding and Improving Shampoo and SOAP via KL Minimization*

---

### Information Theory

- Tishby, Pereira, Bialek. *The Information Bottleneck Method* (2000)
- Tishby & Zaslavsky. *Deep Learning and the Information Bottleneck Principle* (2015)
- Shwartz-Ziv & Tishby. *Opening the Black Box of Deep Neural Networks via Information* (2017)
- Saxe et al. *On the Information Bottleneck Theory of Deep Learning* (NeurIPS 2018) — critique of IB theory

---

### Numerical Methods

- Trefethen & Bau. *Numerical Linear Algebra* — power iteration, spectral norm
- Golub & Van Loan. *Matrix Computations*
- Saad. *Iterative Methods for Sparse Linear Systems*

---

### Empirical Comparisons

- *Fantastic Pretraining Optimizers and Where to Find Them II* — Muon/SOAP competitive at scale
- *A Simpler Parametrization for Modern Optimizers* — minimal hyperparameter set
- *Optimizers and ODEs* (Ji-Ha) — ODE stability framework

---

# Closing

> SCT v2 is not a tweak to Adam.
>
> It is a shift from **parameter-space heuristics → operator approximation in preconditioned geometry**.

The key innovations over prior work:
1. **Structured preconditioning** (Gram-NS for matrices) instead of diagonal-only (Adam)
2. **Alpha-parameterized spectral control** instead of full normalization (Muon) or none (Adam)
3. **Dual spectral control**: continuous (row-normalization) + periodic (spectral clipping)
4. **Operator fidelity tracking**: monitoring how well the applied operator approximates the ideal
5. **Temporal consistency monitoring**: detecting operator drift over training
6. **Momentum-aware thresholds** that account for effective lr amplification
7. **Noise-coupled constraints** that tighten under high gradient noise
8. **4-tuple decomposition** (Geometry, Energy, Fidelity, Cost) as a unified analysis framework

---
