# Conditional Rectified Flow Matching for Paired Relighting

This document compiles the **implementation-level design** for switching from a one-step transformer relighting model to a **conditional flow-matching (rectified flow) model** that transports

`scene_lightA → scene_lightB`

using explicit lighting information.

The goal is to be **drop-in compatible** with an existing transformer-based token updater, while gaining better stability, generalization, and controllability.

---

## 1. Problem Setup

We assume **paired relighting data**:

- Image (or scene tokens) under light A: `x_A`
- Image (or scene tokens) under light B: `x_B`
- Lighting descriptors: `ℓ_A`, `ℓ_B`

After tokenization:

```
z_A = Enc(x_A)   # (B, N, D)
z_B = Enc(x_B)   # (B, N, D)
```

We want to learn a conditional transport:

```
z_A  --Δlight-->  z_B
```

where

```
Δlight = LightEnc(ℓ_B) - LightEnc(ℓ_A)
```

This is a **deterministic, paired, low-entropy mapping**, which makes it ideal for **rectified flow matching**.

---

## 2. Why Flow Matching (Not One-Step Regression)

### One-step regression
```
z_B = f(z_A, Δlight)
```
- brittle
- overfits to seen lights
- poor extrapolation
- no controllable strength

### Flow matching
```
dz/dt = vθ(z_t, t, Δlight)
```
- decomposes large changes into local updates
- smooth trajectories
- better extrapolation
- more stable for video / strong lights

---

## 3. Method Choice

We use:

> **Conditional Rectified Flow Matching (CRFM)**

because:
- paired data
- near-deterministic mapping
- low curvature path
- identity preservation is critical

---

## 4. Mathematical Formulation

### Interpolant (training path)

```
z_t = (1 - t) z_A + t z_B + σ(t) ε
```

with small noise (optional):

```
σ(t) = σ0 · t · (1 - t),   σ0 ∈ [0.03, 0.08]
```

### Target velocity (rectified)

```
v* = z_B - z_A
```

### Conditional vector field

```
vθ(z_t, t, Δlight)
```

### Loss

```
L = E || vθ(z_t, t, Δlight) - (z_B - z_A) ||²
```

---

## 5. Model Architecture

### 5.1 Core principle

Reuse your **existing transformer** as a **velocity-field network**.

Instead of predicting the final tokens, it predicts **token-wise velocity**.

---

### 5.2 Recommended architecture (minimal disruption)

```
VFieldTransformer(
    d = 768,
    d_head = 64,
    n_layers = 12 (or 8),
)
```

#### Inputs
- `z_t` : (B, N, 768)
- `t` : scalar time
- `Δlight` : (B, C)

#### Output
- `v` : (B, N, 768)

---

### 5.3 Conditioning injection (important)

Use **AdaLN / FiLM** in every transformer block.

```
cond = MLP(t_embed) + MLP(Δlight)
h = AdaLN(h, cond)
```

Why:
- relighting is global
- more stable than condition tokens
- avoids attention collapse

---

### 5.4 Time embedding

```
t_embed = MLP(sinusoidal(t))  # -> (B, 768)
```

Always pass `t`, even for rectified flow.

---

## 6. Training Loop (Pseudo-code)

```python
t = rand_uniform(0,1)

zA = Enc(xA)
zB = Enc(xB)

zt = (1-t)*zA + t*zB + sigma(t)*noise
v_target = zB - zA

v_pred = v_theta(zt, t, Δlight)
loss = mse(v_pred, v_target)

loss.backward(); optimizer.step()
```

---

## 7. Inference (ODE Integration)

We solve:

```
dz/dt = vθ(z, t, Δlight)
```

starting from `z(0) = z_A`.

### Heun solver (recommended)

```python
z = zA
for i in range(steps):
    t = i / steps
    v1 = v_theta(z, t, Δlight)
    z_pred = z + dt * v1
    v2 = v_theta(z_pred, t+dt, Δlight)
    z = z + 0.5 * dt * (v1 + v2)
```

Then decode:

```
x̂_B = Dec(z)
```

### Defaults
- steps = 8–16
- solver = Heun
- deterministic

---

## 8. Stability Rules (Important)

Flow matching is **more stable than one-step** if:

1. Use Heun (not Euler)
2. Steps ≥ 5
3. Always condition on `t`
4. Keep σ(t) small

Failure cases come from solver misuse, not the method.

---

## 9. Initialization Trick (Strongly Recommended)

Initialize v-field from your one-step transformer:

- copy all transformer weights
- replace output head
- add conditioning MLPs

This often makes training converge **very fast**.

---

## 10. When Flow Matching Is Better Than One-Step

| Aspect | One-step | Flow matching |
|------|----------|---------------|
| Identity preservation | ❌ | ✔ |
| Strong light changes | ❌ | ✔ |
| Extrapolation | ❌ | ✔ |
| Video consistency | ❌ | ✔ |
| Control strength | ❌ | ✔ |
| Inference speed | ✔ | slightly slower |

---

## 11. Optional Extensions

### 11.1 Distill to 1–2 steps
Train FM → distill into single-step model for speed.

### 11.2 Predict Δz instead of v
Equivalent for rectified flow; sometimes numerically simpler.

### 11.3 Add light-consistency regularizer
If you have a light estimator, add:

```
|| LightHead(Dec(ẑ_B)) - ℓ_B ||
```

---

## 12. Correct Terminology (for paper)

Use:

> Conditional Rectified Flow Matching in Token Space

Avoid saying “diffusion” or “ODE” alone — reviewers care about precision.

---

## 13. Summary (TL;DR)

- Keep your transformer
- Make it predict velocity
- Add t + Δlight conditioning via AdaLN
- Train with rectified flow loss
- Integrate with Heun (8 steps)

You get **better stability, better generalization, and controllable relighting** with minimal code changes.

---

If you want, next I can:
- generate a **full PyTorch module skeleton**
- help you write the **method section** of the paper
- help you plan **ablations reviewers expect**
- help you integrate this into your existing codebase

Just tell me what you want next.

