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

**Objective**: Estimate full state $\mathbf{s} = (x, y, \theta, W, L)$ from accumulated LIDAR points **without ground truth**.

#### GT-Free Initial State Estimation

The initial state is estimated using geometric analysis of LIDAR points via Oriented Bounding Box (OBB):

1. **Boundary Point Extraction**: For each angular sector, take the farthest point (wall detection)
2. **Statistical Outlier Removal**: Remove points with abnormal neighbor distances
3. **Convex Hull**: Compute hull of boundary points using Graham Scan
4. **Minimum-Area OBB**: Find optimal bounding box using Rotating Calipers algorithm

The OBB provides:
- **Room center in robot frame**: $\mathbf{c}_{OBB} = (c_x, c_y)$
- **Room dimensions**: $(W, L) = (\text{width}, \text{height})$ with $W \geq L$
- **Room orientation**: $\phi_{OBB}$

**Robot pose from OBB**:
$$x = -c_x, \quad y = -c_y, \quad \theta = -\phi_{OBB}$$

#### Prior Distribution (Regularization)
Gaussian priors on room dimensions:
$$p(W) = \mathcal{N}(\mu_W = 6, \sigma_W^2 = 1)$$
$$p(L) = \mathcal{N}(\mu_L = 4, \sigma_L^2 = 1)$$

#### Bayesian Fusion
OBB estimates are fused with priors:
$$\hat{W} = \frac{\sigma_{OBB}^2 \mu_W + \sigma_{prior}^2 W_{OBB}}{\sigma_{prior}^2 + \sigma_{OBB}^2}$$

#### Likelihood (SDF Error)
$$p(\mathbf{z} | \mathbf{s}) \propto \exp\left( -\frac{1}{2\sigma_{sdf}^2} \sum_{i=1}^{N} \text{SDF}(\mathbf{p}_i^{room})^2 \right)$$

#### Optimization
Minimize negative log-posterior:
$$\mathcal{L}_{init} = \underbrace{\frac{1}{N}\sum_{i=1}^{N} \text{SDF}^2}_{\text{Likelihood}} + \underbrace{\lambda_{reg} \|\mathbf{s} - \mathbf{s}_0\|^2}_{\text{Regularization}}$$

Solved via gradient descent using PyTorch autograd, initialized with GT-free OBB estimates.

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
$$\mathcal{F} = \mathcal{F}_{likelihood} + \pi_{prior} \cdot \mathcal{F}_{prior}$$

**Likelihood term** (LIDAR fit) - displayed as `F_like` in UI:
$$\mathcal{F}_{likelihood} = \frac{1}{N}\sum_{i=1}^{N} \text{SDF}(\mathbf{p}_i)^2$$

This is the **accuracy** term in Active Inference. Lower values indicate better fit between transformed LIDAR points and room walls. Typical values: 0.001-0.01 (good), 0.01-0.05 (acceptable), >0.05 (poor).

**Prior term** (motion model) - displayed as `F_prior` in UI:
$$\mathcal{F}_{prior} = \frac{1}{2}(\mathbf{s} - \mathbf{s}_{pred})^T \Sigma_{pred}^{-1} (\mathbf{s} - \mathbf{s}_{pred})$$

This is the **complexity** term in Active Inference. It measures the Mahalanobis distance between optimized pose and motion model prediction. Lower values indicate the motion model predicted well. Typical values: 0.0-0.1 (good), 0.1-0.5 (moderate correction), >0.5 (large correction needed).

**Variational Free Energy (VFE)** - displayed as `VFE` in UI:
$$\mathcal{F} = \mathcal{F}_{likelihood} + \pi_{prior} \cdot \mathcal{F}_{prior}$$

The total Free Energy after optimization. This is the objective that the optimizer minimizes. Lower VFE indicates a better balance between explaining observations (accuracy) and staying close to predictions (complexity).

**Adaptive Prior Precision** $\pi_{prior}$:

In Active Inference, **precision** represents confidence or inverse variance. The prior precision $\pi_{prior}$ controls how much the agent trusts its motion model predictions relative to sensory observations (LIDAR).

- High $\pi_{prior}$: Trust motion model more (proprioceptive precision)
- Low $\pi_{prior}$: Trust LIDAR observations more (exteroceptive precision)

The precision adapts based on the innovation (prediction error):
$$d_{Mahal} = \sqrt{(\mathbf{s}^* - \mathbf{s}_{pred})^T \Sigma_{pred}^{-1} (\mathbf{s}^* - \mathbf{s}_{pred})}$$

$$\pi_{target} = \pi_{base} + (\pi_{max} - \pi_{base}) \cdot e^{-d_{Mahal}/\tau}$$

With exponential moving average for smooth adaptation:
$$\pi_t = (1-\alpha)\pi_{t-1} + \alpha \cdot \pi_{target}$$

Parameters: $\pi_{base}=0.05$, $\pi_{max}=0.3$, $\tau=0.5$, $\alpha=0.2$

This implements **precision-weighted prediction error minimization**: when predictions are accurate (low innovation), precision increases, trusting the motion model more. When predictions fail (high innovation), precision decreases, relying more on sensory correction.

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
   - Minimize: F = F_likelihood + π_prior * F_prior
   - Update adaptive precision π based on innovation

3. UPDATE COVARIANCE:
   - Compute Hessian H of likelihood at optimum
   - Σ_post = σ² * H⁻¹

4. OUTPUT:
   - Pose estimate (x, y, θ)
   - Covariance matrix Σ
   - Innovation and prior precision for diagnostics
```

---

## 7. Velocity-Adaptive Precision Weighting

In Active Inference, **precision** represents the confidence or inverse variance of predictions. We extend the standard precision-weighting to incorporate **velocity-dependent precision** for each state variable, enabling the system to focus optimization effort on parameters most likely to change given the current motion profile.

### Motivation

When a robot is:
- **Rotating**: The heading angle $\theta$ changes rapidly while position $(x, y)$ remains relatively stable
- **Translating**: Position $(x, y)$ changes while $\theta$ should remain stable
- **Stationary**: All parameters should converge gradually with uniform precision

### Velocity-Adaptive Precision Weights

Define the **velocity precision weights** $\mathbf{w} = (w_x, w_y, w_\theta)$ as state-specific precision multipliers:

$$\mathbf{w} = \mathcal{W}(v_{linear}, \omega)$$

Where:
- $v_{linear} = \|\mathbf{v}\| = \sqrt{v_x^2 + v_y^2}$ is linear speed
- $\omega$ is angular velocity

### Motion Profile Classification

Based on velocity thresholds $\tau_v$ (linear) and $\tau_\omega$ (angular):

**Case 1: Pure Rotation** ($|\omega| > \tau_\omega$ and $v_{linear} < \tau_v$)
$$\mathbf{w} = (w_{reduce}, w_{reduce}, w_{boost})$$

The agent increases precision (confidence) for $\theta$ estimates while reducing precision for $(x, y)$, reflecting that rotation primarily affects heading.

**Case 2: Pure Translation** ($v_{linear} > \tau_v$ and $|\omega| < \tau_\omega$)
$$\mathbf{w} = (w_x^*, w_y^*, w_{reduce})$$

Where the position weights depend on motion direction:
- Forward/backward motion ($|v_y| > |v_x|$): $w_y^* = w_{boost}$, $w_x^* = w_{base}$
- Lateral motion ($|v_x| > |v_y|$): $w_x^* = w_{boost}$, $w_y^* = w_{base}$

**Case 3: Combined Motion** (both $v_{linear} > \tau_v$ and $|\omega| > \tau_\omega$)
$$\mathbf{w} = (1.2, 1.2, 1.2)$$

Moderate boost to all parameters during complex maneuvers.

**Case 4: Stationary** ($v_{linear} < \tau_v$ and $|\omega| < \tau_\omega$)
$$\mathbf{w} = (w_{base}, w_{base}, w_{base}) = (1, 1, 1)$$

Uniform precision allows gradual convergence of all estimates.

### Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Linear threshold | $\tau_v$ | 0.05 m/s | Below this, robot is "not translating" |
| Angular threshold | $\tau_\omega$ | 0.05 rad/s | Below this, robot is "not rotating" |
| Base weight | $w_{base}$ | 1.0 | Neutral precision multiplier |
| Boost factor | $w_{boost}$ | 2.0 | Increased precision for relevant parameters |
| Reduction factor | $w_{reduce}$ | 0.5 | Decreased precision for irrelevant parameters |

### Application to Optimization

The velocity weights modify the optimization in two ways:

#### 1. Weighted Prior Covariance

The prediction covariance is scaled inversely by the weights:
$$\Sigma_{pred}^{weighted} = D_w^{-1} \Sigma_{pred} D_w^{-1}$$

Where $D_w = \text{diag}(w_x, w_y, w_\theta)$

This means **higher weight = lower uncertainty = stronger constraint** on that parameter.

#### 2. Weighted Gradient Descent

During optimization, gradients are scaled by the velocity weights:
$$\nabla \mathcal{F}_{weighted} = \mathbf{w} \odot \nabla \mathcal{F}$$

Where $\odot$ denotes element-wise multiplication.

Higher weights result in larger gradient steps for those parameters, effectively **focusing optimization effort** on the most relevant state variables.

### Temporal Smoothing

To avoid abrupt changes, weights are smoothed with an exponential moving average:
$$\mathbf{w}_t = (1 - \alpha)\mathbf{w}_{t-1} + \alpha \mathbf{w}_{target}$$

With $\alpha = 0.3$ for responsive but smooth transitions.

### Active Inference Interpretation

In the Active Inference framework, this mechanism can be interpreted as **action-conditional precision**:

- The agent's actions (velocity commands) induce expected changes in specific state variables
- The precision allocated to each state variable reflects the **expected precision** of predictions given the current action
- This is consistent with the Free Energy Principle, where precision weighting optimizes the balance between prior predictions and sensory evidence

Formally, if we denote the action as $\mathbf{a} = (v_x, v_y, \omega)$, the precision matrix becomes action-dependent:
$$\Pi(\mathbf{a}) = \text{diag}(\pi_x(\mathbf{a}), \pi_y(\mathbf{a}), \pi_\theta(\mathbf{a}))$$

This **action-conditional precision** allows the agent to adaptively weight prediction errors, allocating more precision (and thus more corrective weight) to state variables expected to change given the current action.

---

## 8. Uncertainty-Based Speed Modulation

The robot's velocity is modulated based on pose uncertainty, implementing **precision-weighted action** from Active Inference.

### Motivation

When pose uncertainty is high:
- Risk of collision increases (uncertain position)
- Slower movement allows more observations for uncertainty reduction
- Cautious behavior is adaptive

### Speed Factor Computation

From the pose covariance matrix $\Sigma = \begin{pmatrix} \sigma_x^2 & \cdot & \cdot \\ \cdot & \sigma_y^2 & \cdot \\ \cdot & \cdot & \sigma_\theta^2 \end{pmatrix}$, we compute:

**Positional uncertainty:**
$$\sigma_{pos} = \sqrt{\sigma_x^2 + \sigma_y^2}$$

**Speed factor (exponential decay):**
$$f_{speed} = f_{min} + (1 - f_{min}) \cdot \exp(-\lambda \cdot \max(0, \sigma_{pos} - \tau))$$

Where:
- $f_{min} = 0.2$: Minimum speed factor (20% of commanded speed)
- $\lambda = 10$: Uncertainty sensitivity
- $\tau = 0.02$ m: Threshold below which no reduction occurs

### Application to Velocity Commands

$$\mathbf{v}_{actual} = f_{speed} \cdot \mathbf{v}_{commanded}$$

Applied to all velocity components: $(v_x, v_y, \omega)$.

### Temporal Smoothing

To avoid jerky motion, the factor is smoothed with EMA:
$$f_{speed}(t) = (1-\alpha) \cdot f_{speed}(t-1) + \alpha \cdot f_{target}$$

Where $\alpha = 0.3$ is the smoothing factor.

### Active Inference Interpretation

This mechanism implements **precision-weighted action selection**:

- High precision (low $\sigma_{pos}$) → confident actions at full speed
- Low precision (high $\sigma_{pos}$) → cautious actions at reduced speed

This is consistent with the Free Energy Principle: the agent acts to minimize expected free energy, and when uncertainty is high, exploratory (slow, information-gathering) behavior is preferred over exploitative (fast, goal-directed) behavior.

---

## 9. Performance Characteristics

| Metric | Typical Value |
|--------|---------------|
| Room estimation error | ~3 cm |
| Pose error (mean) | ~8-10 cm |
| Angle error | ~1-2° |
| SDF error | ~0.06-0.08 m |
| Computation time | ~10 ms/step |

---

## 10. References

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Buckley, C. L., et al. (2017). The free energy principle for action and perception.
- Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics. MIT Press.
