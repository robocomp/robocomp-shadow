# Active Inference for Room-Robot Estimation

## Mathematical Formulation

This document describes the mathematical foundations of the Active Inference approach used for joint room geometry and robot pose estimation.

---

## 1. State Space

### State Vector
$$\mathbf{s} = (x, y, \theta, W, L)$$

Where:
- $(x, y)$: Robot position in room frame (meters)
- $\theta$: Robot heading angle (radians), $\theta=0$ means facing $+Y$
- $(W, L)$: Room dimensions - width and length (meters)

### Coordinate Convention
- **Room frame**: Origin at room center
  - $X$-axis: Right (lateral), walls at $x = \pm W/2$
  - $Y$-axis: Forward, walls at $y = \pm L/2$
- **Robot frame**: 
  - $X$-axis: Right
  - $Y$-axis: Forward (direction robot is facing)

---

## 2. Generative Model

### Observation Model (LIDAR)

Given robot pose $(x, y, \theta)$ and room dimensions $(W, L)$, LIDAR points $\mathbf{p}_i^{robot}$ in robot frame are transformed to room frame:

$$\mathbf{p}_i^{room} = R(\theta) \cdot \mathbf{p}_i^{robot} + \begin{pmatrix} x \\ y \end{pmatrix}$$

Where the rotation matrix is:
$$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

### Signed Distance Function (SDF)

For a rectangular room, the SDF measures distance to the nearest wall:

$$\text{SDF}(\mathbf{p}) = \min\left( |p_x + W/2|, |p_x - W/2|, |p_y + L/2|, |p_y - L/2| \right)$$

For points on walls (ideal LIDAR): $\text{SDF}(\mathbf{p}) \approx 0$

### Motion Model

Robot pose evolves according to velocity commands:

$$\mathbf{s}_{t+1}^{pred} = f(\mathbf{s}_t, \mathbf{u}_t) = \begin{pmatrix} x_t + v_x^{world} \cdot \Delta t \\ y_t + v_y^{world} \cdot \Delta t \\ \theta_t + \omega \cdot \Delta t \end{pmatrix}$$

Where world velocities are transformed from robot frame:
$$\begin{pmatrix} v_x^{world} \\ v_y^{world} \end{pmatrix} = R(\theta_t) \cdot \begin{pmatrix} v_x^{robot} \\ v_y^{robot} \end{pmatrix}$$

---

## 3. Two-Phase Estimation

### Phase 1: Initialization (Static Robot)

**Objective**: Estimate full state $\mathbf{s} = (x, y, \theta, W, L)$ from accumulated LIDAR points.

#### Prior Distribution
Gaussian priors on room dimensions:
$$p(W) = \mathcal{N}(\mu_W = 6, \sigma_W^2 = 1)$$
$$p(L) = \mathcal{N}(\mu_L = 4, \sigma_L^2 = 1)$$

#### Likelihood (SDF Error)
$$p(\mathbf{z} | \mathbf{s}) \propto \exp\left( -\frac{1}{2\sigma_{sdf}^2} \sum_{i=1}^{N} \text{SDF}(\mathbf{p}_i^{room})^2 \right)$$

#### Optimization
Minimize negative log-posterior:
$$\mathcal{L}_{init} = \underbrace{\frac{1}{N}\sum_{i=1}^{N} \text{SDF}^2}_{\text{Likelihood}} + \underbrace{\lambda_{reg} \|\mathbf{s} - \mathbf{s}_0\|^2}_{\text{Regularization}}$$

Solved via gradient descent using PyTorch autograd.

---

### Phase 2: Tracking (Moving Robot)

**Objective**: Update pose $\mathbf{s}_{pose} = (x, y, \theta)$ with fixed room $(W, L)$.

#### Prediction Step (EKF-style)

Predicted state:
$$\mathbf{s}_{t|t-1} = f(\mathbf{s}_{t-1}, \mathbf{u}_{t-1})$$

Jacobian of motion model:
$$F_t = \frac{\partial f}{\partial \mathbf{s}} = \begin{pmatrix} 1 & 0 & -v_y^{world} \Delta t \\ 0 & 1 & v_x^{world} \Delta t \\ 0 & 0 & 1 \end{pmatrix}$$

Predicted covariance:
$$\Sigma_{t|t-1} = F_t \Sigma_{t-1} F_t^T + Q_t$$

Where process noise scales with speed:
$$Q_t = Q_{base} \cdot \Delta t \cdot \max(0.1, \|\mathbf{v}\| / 0.2)$$

#### Correction Step (Free Energy Minimization)

The Free Energy combines likelihood and prior:
$$\mathcal{F} = \mathcal{F}_{likelihood} + \lambda \cdot \mathcal{F}_{prior}$$

**Likelihood term** (LIDAR fit):
$$\mathcal{F}_{likelihood} = \frac{1}{N}\sum_{i=1}^{N} \text{SDF}(\mathbf{p}_i)^2$$

**Prior term** (motion model):
$$\mathcal{F}_{prior} = \frac{1}{2}(\mathbf{s} - \mathbf{s}_{pred})^T \Sigma_{pred}^{-1} (\mathbf{s} - \mathbf{s}_{pred})$$

**Adaptive Prior Weight** $\lambda$:
Based on the innovation (prediction error):
$$d_{Mahal} = \sqrt{(\mathbf{s}^* - \mathbf{s}_{pred})^T \Sigma_{pred}^{-1} (\mathbf{s}^* - \mathbf{s}_{pred})}$$

$$\lambda_{target} = \lambda_{base} + (\lambda_{max} - \lambda_{base}) \cdot e^{-d_{Mahal}/\tau}$$

With exponential moving average:
$$\lambda_t = (1-\alpha)\lambda_{t-1} + \alpha \cdot \lambda_{target}$$

Parameters: $\lambda_{base}=0.05$, $\lambda_{max}=0.3$, $\tau=0.5$, $\alpha=0.2$

---

## 4. Posterior Covariance (Laplace Approximation)

The posterior covariance is estimated using the Hessian of the cost function at the optimum:

$$\Sigma_{post} = \sigma^2 \cdot H^{-1}$$

Where:
- $\sigma^2 = \frac{1}{N}\sum_i \text{SDF}(\mathbf{p}_i)^2$ is the residual variance
- $H = \nabla^2 \mathcal{L}$ is the Hessian of the loss

The Hessian is computed via PyTorch autograd:
```python
grad = torch.autograd.grad(loss, pose, create_graph=True)
for i in range(3):
    hessian[i] = torch.autograd.grad(grad[i], pose, retain_graph=True)
```

Robust inversion with Cholesky decomposition and regularization ensures numerical stability.

---

## 5. Coordinate Transformations

### Robot Frame → Room Frame
For a point $\mathbf{p}^{robot} = (p_x, p_y)$:
$$\mathbf{p}^{room} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} p_x \\ p_y \end{pmatrix} + \begin{pmatrix} x \\ y \end{pmatrix}$$

### Velocity Transformation
Robot velocity $(v_x^{robot}, v_y^{robot})$ to world velocity:
$$\begin{pmatrix} v_x^{world} \\ v_y^{world} \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} v_x^{robot} \\ v_y^{robot} \end{pmatrix}$$

---

## 6. Algorithm Summary

```
INITIALIZATION PHASE:
1. Collect LIDAR points while robot is static
2. Estimate initial state using PCA + Bayesian priors
3. Optimize s = (x, y, θ, W, L) minimizing SDF error
4. When SDF error < threshold, freeze (W, L) and switch to tracking

TRACKING PHASE (each timestep):
1. PREDICT:
   - s_pred = f(s_prev, u)
   - Σ_pred = F @ Σ_prev @ F^T + Q

2. CORRECT:
   - Minimize: F = F_likelihood + λ * F_prior
   - Update adaptive weight λ based on innovation

3. UPDATE COVARIANCE:
   - Compute Hessian H of likelihood at optimum
   - Σ_post = σ² * H⁻¹

4. OUTPUT:
   - Pose estimate (x, y, θ)
   - Covariance matrix Σ
   - Innovation and prior weight for diagnostics
```

---

## 7. Performance Characteristics

| Metric | Typical Value |
|--------|---------------|
| Room estimation error | ~3 cm |
| Pose error (mean) | ~8-10 cm |
| Angle error | ~1-2° |
| SDF error | ~0.06-0.08 m |
| Computation time | ~10 ms/step |

---

## 8. References

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Buckley, C. L., et al. (2017). The free energy principle for action and perception.
- Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics. MIT Press.
