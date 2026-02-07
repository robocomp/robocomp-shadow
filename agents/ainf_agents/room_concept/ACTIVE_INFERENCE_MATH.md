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

### Velocity Auto-Calibration

The commanded velocities may not match actual robot motion due to wheel slip, calibration errors, or mechanical issues. We learn a **velocity scale factor** $k$ online:

$$\mathbf{v}_{calibrated} = k \cdot \mathbf{v}_{commanded}$$

**Learning from Innovation**: The innovation $\boldsymbol{\epsilon} = \mathbf{s}_{optimized} - \mathbf{s}_{predicted}$ tells us how much the LIDAR-corrected pose differs from the motion model prediction.

**Accumulation**: Over a window of $N$ samples with significant motion:
$$\Delta_{innov} = \sum_{i=1}^{N} \boldsymbol{\epsilon}_i, \quad \Delta_{cmd} = \sum_{i=1}^{N} \mathbf{v}_{cmd,i} \cdot \Delta t$$

**Scale Factor Update**: Project innovation onto motion direction and compute correction:
$$k_{correction} = \frac{\Delta_{innov} \cdot \hat{d}_{motion}}{|\Delta_{cmd}|}$$

Where $\hat{d}_{motion}$ is the unit vector in the direction of commanded motion.

**Exponential Moving Average Update**:
$$k_{new} = (1 - \alpha) \cdot k_{old} + \alpha \cdot (k_{old} + k_{correction})$$

With $\alpha = 0.1$ (learning rate) and $k$ clamped to $[0.5, 1.5]$.

**Interpretation**: If the robot consistently moves less than commanded ($\epsilon < 0$ along motion), $k$ decreases. If it moves more ($\epsilon > 0$), $k$ increases.

### Testing Auto-Calibration

A **simulated velocity error** can be enabled for testing:
```python
self._simulated_velocity_error = 0.85  # Robot moves at 85% of commanded speed
self._velocity_error_enabled = True
```

This applies a scaling factor to the actual velocity sent to the robot, while the estimator receives the original commanded velocity. The auto-calibration should learn to compensate by adjusting $k$ toward 0.85.

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

#### Hierarchical Pose Search

After OBB estimation, a **multi-scale hierarchical search** refines the pose to handle ambiguities (especially the 180° symmetry in rectangular rooms):

##### Level 1 - Coarse Grid Search
- **Spatial grid**: 5×5 positions around OBB estimate (±0.8m)
- **Angular grid**: 16 angles covering full circle (every 22.5°)
- **Candidates evaluated**: ~400 (25 positions × 16 angles)
- **Selection**: Top-10 candidates by combined score

##### Level 2 - Medium Grid Refinement
- **Spatial grid**: 5×5 positions around each top candidate (±0.2m)
- **Angular grid**: 5 angles around each candidate (±15°)
- **Candidates evaluated**: ~1250 (10 × 125 per candidate)
- **Selection**: Top-5 candidates by combined score

##### Level 3 - Fine Local Optimization
- **Method**: Adam optimizer with lr=0.05
- **Iterations**: Up to 30 with early stopping (1e-6 tolerance)
- **Input**: Each of top-5 candidates from Level 2
- **Output**: Best optimized pose

##### Combined Score Function
$$\text{Score}(\mathbf{s}) = \bar{d}_{SDF} \cdot (1 + 0.5 \cdot (1 - r_{inlier}))$$

Where:
- $\bar{d}_{SDF} = \frac{1}{N}\sum_i \text{SDF}(\mathbf{p}_i)$ is mean SDF error
- $r_{inlier} = \frac{|\{i : \text{SDF}(\mathbf{p}_i) < 0.15\}|}{N}$ is inlier ratio

This penalizes poses with many points far from walls (low inlier ratio).

#### Prior Distribution (Regularization)
Gaussian priors on room dimensions:
$$p(W) = \mathcal{N}(\mu_W = 6, \sigma_W^2 = 1)$$
$$p(L) = \mathcal{N}(\mu_L = 4, \sigma_L^2 = 1)$$

#### Bayesian Fusion
OBB estimates are fused with priors:
$$\hat{W} = \frac{\sigma_{OBB}^2 \mu_W + \sigma_{prior}^2 W_{OBB}}{\sigma_{prior}^2 + \sigma_{OBB}^2}$$

#### Likelihood (SDF Error)
$$p(\mathbf{z} | \mathbf{s}) \propto \exp\left( -\frac{1}{2\sigma_{sdf}^2} \sum_{i=1}^{N} \text{SDF}(\mathbf{p}_i^{room})^2 \right)$$

#### Final Optimization
After hierarchical search, minimize negative log-posterior:
$$\mathcal{L}_{init} = \underbrace{\frac{1}{N}\sum_{i=1}^{N} \text{SDF}^2}_{\text{Likelihood}} + \underbrace{\lambda_{reg} \|\mathbf{s} - \mathbf{s}_0\|^2}_{\text{Regularization}}$$

Solved via gradient descent using PyTorch autograd, initialized with the best pose from hierarchical search.

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

The Free Energy objective follows the standard Active Inference formulation (see `main.tex` Eq. obstacle_objective):

$$\mathcal{F}(\mathbf{s}) = \underbrace{\frac{1}{2\sigma_{sdf}^2} \sum_{i=1}^{N} \text{SDF}(\mathbf{p}_i)^2}_{\text{Prediction Error (Accuracy)}} + \underbrace{\frac{1}{2}(\mathbf{s} - \mathbf{s}_{pred})^T \Sigma_{pred}^{-1} (\mathbf{s} - \mathbf{s}_{pred})}_{\text{Complexity (Prior)}}$$

**Key insight**: The balance between likelihood and prior comes naturally from:
- $\sigma_{sdf}^2$: Sensor noise variance (higher → less trust in LIDAR)
- $\Sigma_{pred}^{-1}$: Prior precision matrix (inverse of motion model covariance)

There is **no additional weighting factor** - this is the correct Bayesian formulation where both terms are properly scaled negative log-probabilities.

**Likelihood term** (LIDAR fit) - displayed as `F_like` in UI:
$$\mathcal{F}_{likelihood} = \frac{1}{2\sigma_{sdf}^2 N}\sum_{i=1}^{N} \text{SDF}(\mathbf{p}_i)^2$$

This is the **accuracy** term (prediction error) in Active Inference. It measures how well the LIDAR points fit the room walls given the current pose estimate. The normalization by $N$ provides numerical stability. Parameters:
- $\sigma_{sdf} = 0.15$ m (accounts for sensor noise and dynamic effects)

Typical values: 0.01-0.1 (good fit), 0.1-0.5 (moderate), >0.5 (poor fit).

**Prior term** (motion model) - displayed as `F_prior` in UI:
$$\mathcal{F}_{prior} = \frac{1}{2}(\mathbf{s} - \mathbf{s}_{pred})^T \Sigma_{pred}^{-1} (\mathbf{s} - \mathbf{s}_{pred})$$

This is the **complexity** term in Active Inference. It measures the Mahalanobis distance between optimized pose and motion model prediction. The precision $\Sigma_{pred}^{-1}$ is the inverse of the predicted covariance from the motion model.

Typical values: 0.0-0.5 (good prediction), 0.5-2.0 (moderate correction), >2.0 (large innovation).

**Variational Free Energy (VFE)** - displayed as `VFE` in UI:
$$\text{VFE} = \mathcal{F}_{likelihood} + \mathcal{F}_{prior}$$

The total Free Energy after optimization. This is the objective that the optimizer minimizes. Lower VFE indicates a better balance between explaining observations (accuracy) and staying close to predictions (complexity).

---

## 4. Diagnostic Precision (Innovation Monitor)

The **diagnostic precision** $\pi_{diag}$ shown in the UI indicates how well the motion model predicted the optimized pose. It does **not** affect the optimization - it is purely for monitoring.

**Innovation** (prediction error):
$$\boldsymbol{\epsilon} = \mathbf{s}^* - \mathbf{s}_{pred}$$

**Mahalanobis distance**:
$$d_{Mahal} = \sqrt{\boldsymbol{\epsilon}^T \Sigma_{pred}^{-1} \boldsymbol{\epsilon}}$$

**Diagnostic precision** (exponential decay):
$$\pi_{diag} = \exp(-d_{Mahal} / 2)$$

Interpretation:
- $\pi_{diag} \approx 1.0$: Motion model predicted well (innovation ≈ 0)
- $\pi_{diag} \approx 0.5$: Moderate innovation (~1.4σ)
- $\pi_{diag} \approx 0.05$: Large innovation (~3σ, poor prediction)

---

## 5. Posterior Covariance (Laplace Approximation)

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

## 6. Coordinate Transformations

### Robot Frame → Room Frame
For a point $\mathbf{p}^{robot} = (p_x, p_y)$:
$$\mathbf{p}^{room} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} p_x \\ p_y \end{pmatrix} + \begin{pmatrix} x \\ y \end{pmatrix}$$

### Velocity Transformation
Robot velocity $(v_x^{robot}, v_y^{robot})$ to world velocity:
$$\begin{pmatrix} v_x^{world} \\ v_y^{world} \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} v_x^{robot} \\ v_y^{robot} \end{pmatrix}$$

---

## 7. Algorithm Summary

```
INITIALIZATION PHASE:
1. Collect LIDAR points while robot performs small exploration movements
2. Exploration sequence helps escape local minima:
   - First rotation sweep: ±23° (different wall angles)
   - Small advance: 15cm (different distance to walls)
   - Second rotation sweep: ±29° (more viewpoints)
   - Return to start position
3. Estimate room dimensions using OBB (Oriented Bounding Box)
4. HIERARCHICAL POSE SEARCH:
   a. Level 1 (Coarse): 5×5 grid × 16 angles → Top-10 candidates
   b. Level 2 (Medium): ±0.2m × ±15° around Top-10 → Top-5 candidates
   c. Level 3 (Fine): Local optimization from Top-5 → Best pose
5. Optimize s = (x, y, θ, W, L) starting from hierarchical best
6. If SDF error > 0.2 after max iterations, restart search (up to 3 times)
7. When SDF error < 0.10 AND uncertainty < 0.1, switch to tracking

TRACKING PHASE (each timestep):
1. PREDICT:
   - s_pred = f(s_prev, u)           # Motion model prediction
   - Σ_pred = F @ Σ_prev @ F^T + Q   # Covariance propagation

2. PREDICTION-BASED EARLY EXIT (CPU optimization):
   - Compute SDF at predicted pose: SDF_pred = SDF(s_pred)
   - If mean(SDF_pred) < σ_sdf × trust_factor:
     - Accept s_pred as estimate (skip optimization)
     - Return immediately with 0 iterations
   - Else: proceed to correction step

3. CORRECT (Free Energy Minimization):
   - F_likelihood = Σ SDF² / (2σ²N)  # Normalized likelihood
   - F_prior = ½(s - s_pred)ᵀ Σ⁻¹ (s - s_pred)  # Mahalanobis distance
   - Minimize: F = F_likelihood + F_prior  # No additional weighting!

4. UPDATE COVARIANCE:
   - Compute Hessian H of likelihood at optimum
   - Σ_post = σ² × H⁻¹

5. DIAGNOSTICS:
   - Innovation ε = s* - s_pred
   - π_diag = exp(-‖ε‖_Mahal / 2)  # Monitor motion model quality

6. OUTPUT:
   - Pose estimate (x, y, θ)
   - Covariance matrix Σ
   - Free energy components for UI
```

### Prediction-Based Early Exit

When the motion model prediction already explains observations well (low SDF error), optimization is unnecessary. This implements **trusting the prior** in Active Inference:

$$\text{Skip optimization if: } \bar{d}_{SDF}(\mathbf{s}_{pred}) < \sigma_{sdf} \cdot \tau_{trust} \text{ AND } \text{tr}(\Sigma_{xy}) < \sigma_{max}^2$$

Where:
- $\bar{d}_{SDF}(\mathbf{s}_{pred}) = \frac{1}{N}\sum_i \text{SDF}(\mathbf{p}_i | \mathbf{s}_{pred})$ is mean SDF at predicted pose
- $\tau_{trust} = 0.5$ is the trust factor (default)
- $\text{tr}(\Sigma_{xy})$ is the trace of position covariance (uncertainty in x, y)
- $\sigma_{max}^2 = 0.1$ is the maximum allowed uncertainty for early exit

**Covariance constraint**: Even if SDF is low, if the pose uncertainty grows too large (due to repeated early exits propagating covariance without correction), the system forces a full optimization step to reduce uncertainty.

**Stabilization requirement**: Early exit is only enabled after a minimum number of tracking steps (default: 50) to ensure the room estimation has stabilized before trusting predictions.

**Benefits**:
- Significant CPU savings when robot moves smoothly
- Prior (motion model) is trusted when it's already accurate
- Bounded uncertainty growth through covariance check
- Consistent with Free Energy minimization: if VFE is already low, no correction needed

### Key Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| SDF noise std | $\sigma_{sdf}$ | 0.15 m | Sensor noise + dynamic effects |
| Process noise | $Q$ | diag(0.002, 0.002, 0.001) | Motion model uncertainty (base) |
| Init threshold | - | 0.10 m | SDF error to switch to tracking (stricter) |
| Max init restarts | - | 3 | Maximum restarts if init fails |
| Adam lr (tracking) | - | 0.01 | Optimizer learning rate |
| Min iterations | - | 25 | Minimum optimizer iterations |
| Inlier threshold | - | 0.15 m | SDF threshold for inlier counting |
| Trust factor | $\tau_{trust}$ | 0.5 | Prediction early exit threshold factor |
| Min tracking steps | - | 50 | Steps before enabling early exit |
| Max uncertainty | $\sigma_{max}^2$ | 0.1 | Max position covariance for early exit |

---

## 8. Adaptive LIDAR Decimation

LIDAR point density is dynamically adjusted based on estimation quality to balance CPU usage and accuracy.

### Quality-Based Decimation

The decimation factor $d$ (1 = all points, 4 = 1 of 4 points) is computed from three quality metrics:

$$d = f(\text{SDF}, \text{Cov}, \text{VFE})$$

**Quality Scores** (0 = poor, 1 = excellent):
- SDF score: Based on mean SDF error vs thresholds (0.05, 0.075, 0.125 m)
- Covariance score: Based on $\text{tr}(\Sigma_{xy})$ vs thresholds (0.02, 0.05, 0.1)
- VFE score: Based on VFE vs thresholds (0.1, 0.3, 0.5)

**Combined Quality**: $q = \min(\text{SDF}_{score}, \text{Cov}_{score}, \text{VFE}_{score})$

The worst metric dominates (conservative approach).

### Decimation Mapping

| Quality $q$ | Decimation $d$ | Description |
|-------------|----------------|-------------|
| $q \geq 0.75$ | 4 | Excellent - high subsampling |
| $q \geq 0.5$ | 3 | Good - moderate subsampling |
| $q \geq 0.25$ | 2 | Degraded - minimal subsampling |
| $q < 0.25$ | 1 | Poor - use all points |

### Asymmetric Response

- **Quality degradation**: Immediate reduction in decimation (responsive)
- **Quality improvement**: Gradual increase in decimation (stable)

This asymmetry ensures fast response to problems (e.g., during rotation) while avoiding oscillation when quality improves.

### Active Inference Interpretation

This implements **precision-weighted sensing**: when prediction errors are high (high VFE), the agent allocates more sensory resources (more LIDAR points) to reduce uncertainty. This is consistent with the Free Energy Principle where precision modulates the influence of prediction errors.

---

## 9. Velocity-Adaptive Precision Weighting

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

## 9. Uncertainty-Based Speed Modulation

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
- $f_{min} = 0.3$: Minimum speed factor (30% of commanded speed)
- $\lambda = 1.0$: Uncertainty sensitivity (reduced to avoid over-reaction)
- $\tau = 0.2$ m: High threshold - only reduces speed when uncertainty > 20cm

**Note**: The threshold is set high to avoid interfering with normal operation. The covariance naturally grows during prediction and shrinks during correction. Speed modulation should only activate when there's a genuine localization problem.

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

## 10. Performance Characteristics

| Metric | Typical Value |
|--------|---------------|
| Room estimation error | ~3 cm |
| Pose error (mean) | ~8-10 cm |
| Angle error | ~1-2° |
| SDF error | ~0.06-0.08 m |
| Computation time | ~10 ms/step |

---

## 11. References

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Buckley, C. L., et al. (2017). The free energy principle for action and perception.
- Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics. MIT Press.
