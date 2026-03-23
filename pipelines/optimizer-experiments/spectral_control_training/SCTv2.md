# Spectral Control Training v2: Geometry-Consistent Optimization for LLMs

Large-scale model training is often framed as designing better optimizers.
However, under stochastic gradients, preconditioning, and low precision, this view becomes insufficient.

A more precise formulation is:

> **LLM training is a control problem in a preconditioned geometry.**

In this post, we present **Spectral Control Training v2 (SCT v2)**—a framework that unifies:

* preconditioning (Adam/Muon)
* noise-aware scheduling
* spectral stability

into a **geometry-consistent control system**.

---

# 1. Theoretical Foundation: Optimization in Preconditioned Geometry

Consider second-order approximation:

[
L(\theta + \Delta) \approx L(\theta) + g^T \Delta + \frac{1}{2}\Delta^T H \Delta
]

Ideal update:

[
\Delta^* = -H^{-1} g
]

In practice:

[
\Delta = -P^{-1} g
]

where (P) approximates curvature.

---

## Key Observation

Optimization does not happen in Euclidean space, but in the geometry induced by (P).

Define the **natural metric**:

[
||g||_{P^{-1}}^2 = g^T P^{-1} g
]

---

## 🔥 Core Insight

> **The correct notion of “step size” is not (||\Delta||), but (g^T P^{-1} g).**

---

# 2. SCT v2 Algorithm

We define the update:

[
\Delta = -\alpha_t \cdot P^{-1} g
]

where (\alpha_t) is chosen to control the **natural gradient energy**.

---

## Algorithm

```python
# SCT v2

for step t:

    g = grad(W)

    # Preconditioner (Adam / Muon / etc.)
    u = P^{-1} g

    # Natural step size
    T_current = sqrt(sum(g * u))

    # Target schedule (noise-aware)
    T_target = schedule(t, noise_est)

    # Scale update
    u *= (T_target / (T_current + eps))

    # Apply update
    W -= u

    # Optional spectral constraint
    if step % k == 0:
        sigma = estimate_sigma_max(W)
        if sigma > R(t):
            W *= (R(t) / sigma)
```

---

# 3. Core Components

## 3.1 Spectral Filter (Preconditioner)

| Method      | Geometry          |
| ----------- | ----------------- |
| SGD         | Euclidean         |
| AdamW       | Diagonal metric   |
| Muon / SOAP | Structured metric |

---

## 3.2 Natural Energy (Key Variable)

[
T = \sqrt{g^T P^{-1} g}
]

This replaces:

* learning rate
* ||ΔW|| heuristics
* Hyperball scaling

---

## 3.3 Noise Estimator

```python
noise_est = EMA(||g||^2) - ||EMA(g)||^2
```

---

## 3.4 Temperature Schedule

[
T(t) = \frac{T_0}{\sqrt{t}} \cdot \frac{1}{1 + \text{noise}}
]

---

## 🔥 Interpretation

> We are controlling **energy injected into the system**, not raw parameter movement.

---

# 4. Systems Design

## 4.1 Distributed Consistency

We need global reductions:

```python
T_current = all_reduce(sum(g * u))
```

---

## 4.2 Efficient Implementation

Reuse optimizer state:

* (u = m / \sqrt{v})
* compute (g \cdot u) cheaply

---

## 4.3 No Extra Instability

Unlike Hyperball:

* no explicit normalization of (u)
* only scaling

---

# 5. Quantization-Aware Training

Low precision introduces noise:

[
g \rightarrow g + \epsilon
]

---

## SCT v2 Advantage

Controls:

[
g^T P^{-1} g
]

This ensures:

> **update energy remains stable under quantization noise**

---

## Adaptive Schedule

```python
T_target = base_T / (1 + quant_noise)
```

---

# 6. Experimental Plan

## Baselines

* AdamW
* Muon
* Hyperball
* SSO
* SCT v2

---

## Metrics

* loss vs tokens
* gradient noise scale
* natural step size (g^T P^{-1} g)
* spectral norm

---

## Expected Results

| Method    | Behavior              |
| --------- | --------------------- |
| AdamW     | stable but suboptimal |
| Muon      | fast but unstable     |
| Hyperball | stable but rigid      |
| SCT v2    | fast + stable         |

---

# 7. Relation to Prior Concepts (IC / GDN)

## IC Perspective

SCT v2 improves:

> signal quality per computation step

Equivalent to:

* reducing effective noise dimension
* improving information flow

---

## GDN Perspective

* updates = operators
* stability depends on spectrum

SCT v2 ensures:

* controlled update energy
* stable operator dynamics

---

# 8. Final Unified View

We now unify training as:

[
\Delta = -\alpha_t \cdot P^{-1} g
\quad \text{s.t.} \quad \lambda_{\max}(W) \le R(t)
]

---

## Components

| Element        | Role      |
| -------------- | --------- |
| (P^{-1})       | geometry  |
| (g^T P^{-1} g) | energy    |
| (T(t))         | control   |
| (R(t))         | stability |

---

## 🔥 Final Insight

> **Optimization is not about moving parameters.
> It is about controlling energy in a curved space.**

---

# 9. References

### Optimization and Preconditioning

* Kingma & Ba. *Adam: A Method for Stochastic Optimization* (2014)
* Loshchilov & Hutter. *Decoupled Weight Decay Regularization* (2017)
* Martens. *Hessian-Free Optimization* (2010)
* Grosse & Martens. *K-FAC* (2016)

---

### Noise and Scaling

* Smith et al. *Don’t Decay the Learning Rate, Increase the Batch Size* (2017)
* Smith & Dherin. *On the Origin of Implicit Regularization in SGD* (2020)
* Kaplan et al. *Scaling Laws for Neural Language Models* (2020)
* Hoffmann et al. *Chinchilla* (2022)

---

### Spectral Methods

* Trefethen & Bau. *Numerical Linear Algebra*
* Golub & Van Loan. *Matrix Computations*
* Saad. *Iterative Methods for Sparse Linear Systems*

---

### Spectral Stability

* Miyato et al. *Spectral Normalization for GANs* (2018)

---

### Weight Decay / Hyperball

* [https://whenwen.github.io/wd_blog/public/weight-decay-part-2.html](https://whenwen.github.io/wd_blog/public/weight-decay-part-2.html)

---

# 🎯 Closing

> SCT v2 is not a tweak to Adam.
> It is a shift from **parameter-space heuristics → geometry-consistent control**.

---
