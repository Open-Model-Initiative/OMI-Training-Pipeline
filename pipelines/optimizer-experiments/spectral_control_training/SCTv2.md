# Spectral Control Training v2: Geometry-Consistent Optimization for LLMs

Large-scale model training is often framed as designing better optimizers.
However, under stochastic gradients, preconditioning, and low precision, this view becomes insufficient.

A more precise formulation is:

> **LLM training is a control problem in a preconditioned geometry.**

In this post, we present **Spectral Control Training v2 (SCT v2)**—a framework that unifies:

- structured preconditioning (Muon-style for matrices, Adam-style for vectors)
- noise-adaptive scheduling
- spectral stability (row-normalization + periodic spectral clipping)

into a **geometry-consistent control system**.

---

# 1. Theoretical Foundation: Optimization in Preconditioned Geometry

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

where $P$ approximates curvature.

## Key Observation

Optimization does not happen in Euclidean space, but in the geometry induced by $P$.

Define the **natural metric** (Fisher-Rao metric when $P = F$, the Fisher Information Matrix):

```math
\lVert g \rVert_{P^{-1}}^2 = g^T P^{-1} g
```

For matrix-shaped parameters, the optimal preconditioner is $(G^T G)^{-1/2}$ (Newton-Muon, Du & Su 2025),
not a diagonal approximation. This equalizes the effective step size across all singular directions.

## Core Insight

> **The correct notion of "step size" is not $\lVert\Delta\rVert$, but $g^T P^{-1} g$.**

This is well-grounded in information geometry: the natural gradient direction $P^{-1}g$
is the steepest descent direction in the Riemannian manifold induced by $P$.
The quantity $g^T P^{-1} g$ measures the **information-theoretic distance** moved per step
(Amari, 1998; Martens, 2014).

### Wasserstein Flow Interpretation

Recent work (Peyré, 2025) shows that Muon-style optimization can be interpreted as
a **Wasserstein flow on the space of spectral measures** of weight matrices. The optimizer
transports the singular value distribution toward a spectral equilibrium. This validates
SCT v2's approach of explicitly controlling spectral properties—the implicit spectral
regularization of Muon becomes an explicit control mechanism.

---

# 2. SCT v2 Algorithm

We define the update:

```math
\Delta = -\alpha_t \cdot P^{-1} g
```

where $\alpha_t$ is chosen to control the **natural gradient energy** and $P$ is a
**structured preconditioner** that differs by parameter shape.

## Algorithm

```python
# SCT v2

for step t:
    for param in parameters:
        g = grad(param)

        # === PRECONDITIONING (shape-dependent) ===
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
        T_current = sqrt(g^T u)                      # natural energy

        noise_ratio = grad_variance / (grad_signal^2 + eps)
        T_target = T0 / (sqrt(t + warmup) * (1 + noise_ratio))

        update *= T_target / (T_current + eps)

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

This has a principled interpretation via Smith & Dherin (2021): the discrete SGD update
introduces an implicit regularizer proportional to the update energy:

```math
L_{\text{discrete}} = L + \frac{\eta}{2} \|g\|_P^2 + \text{higher-order terms}
```

By controlling $\|g\|_{P^{-1}}^2 = g^T P^{-1} g$, we are controlling
**the strength of this implicit regularization** at each step.

---

# 3. Core Components

## 3.1 Structured Preconditioner (Shape-Dependent)

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

## 3.2 Natural Energy (Key Variable)

```math
T = \sqrt{g^T P^{-1} g}
```

This replaces:
- learning rate as the primary step-size control
- $\lVert\Delta W\rVert$ heuristics (which measure displacement, not information)
- Hyperball scaling (which constrains parameter norms, not update energy)

**Theoretical grounding**: In natural gradient descent, $g^T F^{-1} g$ (where $F$ is the
Fisher information matrix) measures the KL divergence between the current model and the
model after one update step. By controlling this quantity, we directly control
the information-theoretic impact of each update.

## 3.3 Noise Estimator

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

## 3.4 Temperature Schedule

```math
T(t) = \frac{T_0}{\sqrt{t + t_{\text{warmup}}}} \cdot \frac{1}{1 + \text{noise\_ratio}}
```

**Warmup rationale**: At the start of training, gradients are large and noisy.
A warmup phase prevents the optimizer from taking overly aggressive steps
before the preconditioner statistics have stabilized.

**Noise-adaptive rationale**: When gradient noise is high (low SNR), the optimizer
reduces step size to avoid amplifying noise. When gradients are clean (high SNR),
it can afford larger steps. This is analogous to the "noise-aware" scheduling
in Smith et al. (2017).

## 3.5 Row-Normalization (Continuous Spectral Control)

Between periodic spectral clipping events, row-normalization provides **continuous**
spectral control at negligible cost (RMNP, 2026):

```python
row_norms = W.norm(dim=-1, keepdim=True)
W = W / row_norms.clamp_min(eps)
```

**Why row-normalization?** Row-normalized weight matrices have bounded spectral norms
($\sigma_{\max} \le \sqrt{\text{num\_rows}}$). This prevents spectral blowup between
clipping events without the cost of power iteration.

## 3.6 Spectral Stability (Momentum-Aware)

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

# 4. Systems Design

## 4.1 Distributed Consistency

We need global reductions for consistent energy measurement:

```python
# Energy must be globally consistent across all ranks
T_current = all_reduce(sum(g * u)) / world_size

# Noise ratio is per-rank (gradient noise is local)
noise_ratio = local_noise_estimate
```

**Key insight**: The natural energy $g^T P^{-1} g$ must be computed globally because
it determines the scaling factor applied to all parameters. If each rank computed
a different scale, the parameters would diverge across ranks.

The noise ratio is local because gradient noise is an inherent property of each
rank's data shard — averaging it across ranks would mask rank-specific noise patterns.

## 4.2 Efficient Implementation

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

## 4.3 No Extra Instability

Unlike Hyperball (which normalizes $u$ directly):

- SCT v2 only **scales** $u$ by a scalar factor
- The direction of the update is preserved
- Spectral constraints are applied as **soft clipping** (damped projection),
  not hard clipping, to avoid gradient discontinuities
- Row-normalization provides continuous control between clipping events

---

# 5. Quantization-Aware Training

Low precision introduces noise:

```math
g \rightarrow g + \epsilon_{\text{quant}}
```

## SCT v2 Advantage

SCT v2 controls:

```math
g^T P^{-1} g
```

This ensures:

> **update energy remains stable under quantization noise**

When quantization noise $\epsilon_{\text{quant}}$ increases the apparent gradient variance,
the noise estimator detects this and reduces the target temperature accordingly.

## Adaptive Schedule

```python
# Quantization noise inflates the noise ratio
noise_ratio = (grad_variance + quant_variance) / (signal^2 + eps)
T_target = base_T / (1 + noise_ratio)
```

---

# 6. Experimental Plan

## Baselines

- AdamW (standard baseline)
- Muon (structured preconditioning)
- SOAP (KL-based preconditioning)
- Hyperball (weight-space normalization)
- SCT v2 (our method)

## Metrics

- loss vs tokens (sample efficiency)
- gradient noise scale (McCandlish et al.)
- natural step size $g^T P^{-1} g$ (our controlled variable)
- spectral norm $\lambda_{\max}(W)$ over training
- singular value distribution (spectral diversity)
- implicit regularization strength (Smith & Dherin metric)
- downstream task generalization (not just pretraining loss)

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
5. **NS iterations**: 2 vs 3 vs 5 iterations (convergence vs compute tradeoff)
6. **Output layer clipping aggressiveness**: 1x vs 2x vs 4x
7. **Spectral constraint damping**: hard clip (1.0) vs soft clip (0.5) vs gentle (0.2)

---

# 7. Relation to Prior Concepts

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

- updates = operators on parameter space
- stability depends on the operator's spectral properties

SCT v2 ensures:
- controlled update energy (bounded operator norm in preconditioned metric)
- stable operator dynamics (spectral constraints prevent explosion)
- spectral diversity preservation (alpha-parameterization prevents over-normalization)

**Connection to implicit regularization**: Smith & Dherin (2021) showed that
the discretization of gradient flow introduces regularization proportional to
the update energy. In preconditioned geometry, this becomes:

```math
\text{implicit regularizer} \propto \frac{\eta}{2} g^T P^{-1} g
```

By controlling $g^T P^{-1} g$, SCT v2 directly controls the implicit regularization
strength — a theoretically grounded connection.

## Edge of Stability Perspective

Gradient descent on smooth nonconvex losses converges to a "sharpness-aware" regime
where the step size is approximately $2/\lambda_{\max}(H)$ (Cohen et al.). In the
stochastic case, gradient noise pushes this threshold lower. SCT v2's momentum-aware
spectral constraint accounts for this by using the effective learning rate
$\eta \cdot (1+\beta)/(1-\beta)$ in the threshold computation.

---

# 8. Final Unified View

We now unify training as:

```math
\Delta = -\alpha_t \cdot P^{-1} g
\quad \text{s.t.} \quad \lambda_{\max}(W) \le R(t, \beta, \text{noise})
```

## Components

| Element                | Role                                        |
| ---------------------- | ------------------------------------------- |
| $P^{-1}$ (Gram-NS)     | geometry (structured preconditioner)        |
| $P^{-1}$ (Adam diag)   | geometry (diagonal for vectors)             |
| $g^T P^{-1} g$         | energy (information-theoretic step size)    |
| $T(t)$                 | control (noise-adaptive schedule)           |
| $R(t, \beta, \text{noise})$ | stability (momentum-aware spectral bound) |
| $\alpha$               | spectral normalization strength             |
| row-normalization      | continuous spectral control                 |

## Final Insight

> **Optimization is not about moving parameters.**
>
> **It is about controlling energy in a curved space.**

The key contributions are:
1. **Structured preconditioning**: Gram-NS for matrices, Adam for vectors
2. **Alpha-parameterized spectral control**: partial normalization preserves spectral diversity
3. **Natural energy as control variable**: replacing learning rate with $g^T P^{-1} g$
4. **Noise-adaptive scheduling**: step size responds to gradient SNR
5. **Dual spectral control**: row-normalization (continuous) + spectral clipping (periodic)
6. **Momentum-aware thresholds**: accounts for effective lr amplification
7. **Preconditioner-agnostic framework**: works with any $P$ (Adam, Muon, SOAP, etc.)

---

# 9. References

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
> It is a shift from **parameter-space heuristics → geometry-consistent control**.

The key innovations over prior work:
1. **Structured preconditioning** (Gram-NS for matrices) instead of diagonal-only (Adam)
2. **Alpha-parameterized spectral control** instead of full normalization (Muon) or none (Adam)
3. **Dual spectral control**: continuous (row-normalization) + periodic (spectral clipping)
4. **Momentum-aware thresholds** that account for effective lr amplification
5. **Noise-coupled constraints** that tighten under high gradient noise

---
