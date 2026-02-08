# Object Inference using Active Inference

## Mathematical Formalism

This document describes the complete mathematical framework for estimating 3D objects (boxes, tables, chairs) using Active Inference, including **Bayesian Model Selection** for inferring object type as a discrete latent variable.

**Key components:**
1. **Continuous state inference**: Position, orientation, and dimensions via VFE minimization
2. **Discrete model selection**: Object type (table vs chair) via Bayesian model comparison
3. **Historical evidence**: Temporal Consistency Error (TCE) for weighting past observations

---

## 1. Problem Statement

A mobile robot equipped with a 3D LIDAR sensor observes its environment and must infer:
1. **What type** of object is present (table, chair, box, etc.)
2. **Where** it is located (position, orientation)
3. **What shape** it has (dimensions)

The robot's pose in the room frame is uncertain, characterized by a covariance matrix. The goal is to maintain probabilistic beliefs over both discrete model identity and continuous parameters.

---

## 2. State Space

### 2.1 Object State Vectors

Each object type has a specific state vector encoding position, dimensions, and orientation.

#### 2.1.1 Box (6 parameters)

$$
\mathbf{s}_{\text{box}} = (c_x, c_y, w, h, d, \theta)^\top
$$

| Parameter | Description | Units |
|-----------|-------------|-------|
| $c_x, c_y$ | Center position in room frame | meters |
| $w$ | Width (X dimension) | meters |
| $h$ | Height (Y dimension in XY plane) | meters |
| $d$ | Depth (Z dimension, vertical) | meters |
| $\theta$ | Orientation angle around Z axis | radians |

#### 2.1.2 Table (7 parameters)

$$
\mathbf{s}_{\text{table}} = (c_x, c_y, w, h, h_{\text{table}}, l_{\text{leg}}, \theta)^\top
$$

| Parameter | Description | Units |
|-----------|-------------|-------|
| $c_x, c_y$ | Center position in room frame | meters |
| $w$ | Table top width (X dimension) | meters |
| $h$ | Table top depth (Y dimension) | meters |
| $h_{\text{table}}$ | Height of table surface from floor | meters |
| $l_{\text{leg}}$ | Length of legs | meters |
| $\theta$ | Orientation angle around Z axis | radians |

**Fixed constants:**
- Top thickness: $t_{\text{top}} = 0.03$ m
- Leg radius: $r_{\text{leg}} = 0.025$ m

#### 2.1.3 Chair (7 parameters)

$$
\mathbf{s}_{\text{chair}} = (c_x, c_y, w_s, d_s, h_s, h_b, \theta)^\top
$$

| Parameter | Description | Units |
|-----------|-------------|-------|
| $c_x, c_y$ | Center position in room frame | meters |
| $w_s$ | Seat width | meters |
| $d_s$ | Seat depth | meters |
| $h_s$ | Seat height from floor | meters |
| $h_b$ | Backrest height above seat | meters |
| $\theta$ | Orientation angle around Z axis | radians |

**Fixed constants:**
- Seat thickness: $t_{\text{seat}} = 0.05$ m
- Backrest thickness: $t_{\text{back}} = 0.05$ m

**Convention:** The backrest is positioned at $+Y$ in the object's local frame.

#### 2.1.4 TV (6 parameters)

$$
\mathbf{s}_{\text{tv}} = (c_x, c_y, w, h, z_{\text{base}}, \theta)^\top
$$

| Parameter | Description | Units |
|-----------|-------------|-------|
| $c_x, c_y$ | Center position in room frame (floor projection) | meters |
| $w$ | Screen width (horizontal extent) | meters |
| $h$ | Screen height (vertical extent) | meters |
| $z_{\text{base}}$ | Height from floor to bottom of TV (mount height) | meters |
| $\theta$ | Orientation angle around Z axis | radians |

**Fixed constants:**
- Depth (thickness): $d_{\text{tv}} = 0.05$ m (not optimized)

**Key characteristics:**
- TVs are **thin panels**: $w \gg d_{\text{tv}}$
- Screen aspect ratio typically $\approx 2:1$ (width/height)
- Usually **wall-mounted**: $z_{\text{base}} \in [0.5, 1.5]$ m
- Strong **angle alignment** with walls (0°, 90°, ±180°)

### 2.2 Robot State

The robot pose in the room frame:

$$
\mathbf{r}_t = (r_x, r_y, r_\theta)^\top
$$

with associated covariance:

$$
\Sigma_{\text{robot}}(t) \in \mathbb{R}^{3 \times 3}
$$

---

## 3. Signed Distance Functions (SDF)

The SDF returns the signed distance from a point to the object's surface:
- $\text{SDF} > 0$: Point is **outside** the object
- $\text{SDF} = 0$: Point is **on the surface**
- $\text{SDF} < 0$: Point is **inside** the object

### 3.1 Box SDF

For a 3D oriented box:

1. **Transform point to box-local frame:**
$$
\mathbf{p}_{\text{local}} = R(-\theta)^\top (\mathbf{p} - \mathbf{c})
$$

2. **Compute distance to box surface:**
$$
\mathbf{q} = |\mathbf{p}_{\text{local}}| - \frac{1}{2}\begin{pmatrix} w \\ h \\ d \end{pmatrix}
$$

3. **SDF value:**
$$
\text{SDF}_{\text{box}}(\mathbf{p}, \mathbf{s}) = \|\max(\mathbf{q}, 0)\| + \min(\max(q_x, q_y, q_z), 0)
$$

The first term handles points outside, the second handles points inside.

### 3.2 Table SDF

The table consists of a **box** (top) and **4 cylinders** (legs).

```
        ┌─────────────────────────┐  ← Top (box)
        │                         │     z = h_table
        └─────────────────────────┘
          │                     │
          │                     │    ← Legs (cylinders)
          │                     │
          ▼                     ▼
        ─────                 ─────   z = 0 (floor)
```

**Top box SDF:**
$$
\text{SDF}_{\text{top}}(\mathbf{p}) = \text{SDF}_{\text{box}}\left(\mathbf{p}, (c_x, c_y, z_{\text{top}}), (w, h, t_{\text{top}})\right)
$$
where $z_{\text{top}} = h_{\text{table}} - t_{\text{top}}/2$

**Leg cylinder SDF** (for leg $i$ at position $(x_i, y_i)$):
$$
\text{SDF}_{\text{leg}_i}(\mathbf{p}) = \text{SDF}_{\text{cylinder}}\left(\mathbf{p}, (x_i, y_i), r_{\text{leg}}, l_{\text{leg}}\right)
$$

Leg positions (in local frame):
$$
(x_i, y_i) \in \left\{ \left(\pm\frac{w-r}{2}, \pm\frac{h-r}{2}\right) \right\}
$$
where $r = 2 \cdot r_{\text{leg}}$ is the inset from edges.

**Combined Table SDF (union):**
$$
\text{SDF}_{\text{table}}(\mathbf{p}) = \min\left( \text{SDF}_{\text{top}}(\mathbf{p}), \min_{i=1}^{4} \text{SDF}_{\text{leg}_i}(\mathbf{p}) \right)
$$

### 3.3 Chair SDF

The chair consists of a **seat** (box) and a **backrest** (box).

```
     ┌─────────────┐  ← Backrest
     │             │     z = h_s + h_b/2
     │             │
     └─────────────┘
     ┌─────────────┐  ← Seat
     │             │     z = h_s - t_seat/2
     └─────────────┘
           │
         floor        z = 0
         
    Top view:
    ┌─────────────┐
    │   Backrest  │  +Y (back)
    ├─────────────┤
    │             │
    │    Seat     │
    │             │
    └─────────────┘  -Y (front)
```

**Seat box SDF:**
$$
\text{SDF}_{\text{seat}}(\mathbf{p}) = \text{SDF}_{\text{box}}\left(\mathbf{p}, (c_x, c_y, z_s), (w_s, d_s, t_{\text{seat}})\right)
$$
where $z_s = h_s - t_{\text{seat}}/2$

**Backrest box SDF:**
$$
\text{SDF}_{\text{back}}(\mathbf{p}) = \text{SDF}_{\text{box}}\left(\mathbf{p}, (c_x, c_y + y_{\text{off}}, z_b), (w_s, t_{\text{back}}, h_b)\right)
$$
where:
- $z_b = h_s + h_b/2$ (center height of backrest)
- $y_{\text{off}} = d_s/2 - t_{\text{back}}/2$ (offset to back edge of seat)

**Combined Chair SDF (union):**
$$
\text{SDF}_{\text{chair}}(\mathbf{p}) = \min\left( \text{SDF}_{\text{seat}}(\mathbf{p}), \text{SDF}_{\text{back}}(\mathbf{p}) \right)
$$

### 3.4 TV SDF

A TV is modeled as a **thin rectangular panel** mounted on a wall at height $z_{\text{base}}$.

```
Side view:                   Front view:
     │                       ┌─────────────────┐
     │ ← wall                │                 │
┌────┤                       │       TV        │  h (screen height)
│ TV │ ← depth (fixed 5cm)   │                 │
└────┤                       └─────────────────┘
     │                              w (screen width)
     │
  ───┴─── floor (z=0)         z_base = mount height
```

**TV SDF:**

1. **Transform to TV-local frame** (same as box):
$$
\mathbf{p}_{\text{local}} = R(-\theta)^\top (\mathbf{p} - \mathbf{c})
$$

2. **Compute distance to panel surface:**
$$
\mathbf{q} = \begin{pmatrix}
|p_x^{\text{local}}| - w/2 \\
|p_y^{\text{local}}| - d_{\text{tv}}/2 \\
|p_z - (z_{\text{base}} + h/2)| - h/2
\end{pmatrix}
$$

3. **SDF value** (same formula as box):
$$
\text{SDF}_{\text{tv}}(\mathbf{p}) = \|\max(\mathbf{q}, 0)\| + \min(\max(q_x, q_y, q_z), 0)
$$

**Key difference from box:** The depth $d_{\text{tv}} = 0.05$ m is **fixed**, not optimized. This reduces the state dimension from 7 to 6.

### 3.5 Cylinder SDF (auxiliary)

For a vertical cylinder with center $(c_x, c_y)$, radius $r$, and height $h$:

$$
\text{SDF}_{\text{cylinder}}(\mathbf{p}) = \max\left( \sqrt{(p_x - c_x)^2 + (p_y - c_y)^2} - r, \quad |p_z - h/2| - h/2 \right)
$$

### 3.5 Smooth Minimum for Internal Points

**Problem:** The standard SDF uses a hard minimum for internal points. Gradients only flow to the dimension with smallest distance, preventing optimization of unseen faces.

**Solution:** Replace hard minimum with **log-sum-exp smooth minimum**:

$$
\text{smooth\_min}(q_x, q_y, q_z) \approx -k \cdot \log\left( e^{-q_x/k} + e^{-q_y/k} + e^{-q_z/k} \right)
$$

where $k = 0.02$ m is the smoothness parameter.

**Additionally:** Internal points are scaled by $\alpha_{\text{inside}} = 0.3$ to reduce their influence:

$$
\text{SDF}_{\text{inside}} = \alpha_{\text{inside}} \cdot \text{smooth\_min}(q_x, q_y, q_z)
$$

| Constant | Description | Value |
|----------|-------------|-------|
| `SDF_SMOOTH_K` | Smoothness parameter $k$ | 0.02 m |
| `SDF_INSIDE_SCALE` | Internal point scale $\alpha_{\text{inside}}$ | 0.3 |

---

## 4. Generative Model

### 4.1 Observation Model

LIDAR points are generated by the surfaces of objects. For a point $\mathbf{p}$ observed in the robot frame, we transform it to the room frame:

$$
\mathbf{p}_{\text{room}} = R(r_\theta) \mathbf{p}_{\text{robot}} + \begin{pmatrix} r_x \\ r_y \end{pmatrix}
$$

The generative model assumes that observed surface points have zero signed distance to the object surface, corrupted by Gaussian noise:

$$
p(\mathbf{p} | \mathbf{s}) = \mathcal{N}\left( \text{SDF}(\mathbf{p}, \mathbf{s}) \mid 0, \sigma_o^2 \right)
$$

where $\text{SDF}(\mathbf{p}, \mathbf{s})$ is the signed distance function for the object type (see Section 3).

### 4.2 Prior Model (Temporal Dynamics)

Objects are assumed static. The prior at time $t$ is the posterior from time $t-1$ with added process noise:

$$
p(\mathbf{s}_t) = \mathcal{N}(\mathbf{s}_t \mid \mathbf{\mu}_{t-1}, \Sigma_{t-1} + \Sigma_{\text{process}})
$$

where:
$$
\Sigma_{\text{process}} = \text{diag}(\sigma_{xy}^2, \sigma_{xy}^2, \sigma_{\text{size}}^2, ..., \sigma_{\theta}^2)
$$

### 4.2.1 State Prior as Regularizer

For **static objects**, the prior acts as a soft regularizer penalizing state changes:

$$
F_{\text{prior}} = \frac{\lambda_{\text{pos}}}{2} \|\mathbf{s}_{xy} - \mathbf{s}_{xy}^{\text{prev}}\|^2 + \frac{\lambda_{\text{size}}}{2} \|\mathbf{s}_{\text{dim}} - \mathbf{s}_{\text{dim}}^{\text{prev}}\|^2 + \frac{\lambda_{\theta}}{2} (\theta - \theta^{\text{prev}})^2
$$

**Current implementation values:**
| Parameter | Value | Effect |
|-----------|-------|--------|
| $\lambda_{\text{pos}}$ | 0.05 | Weak position regularization |
| $\lambda_{\text{size}}$ | 0.02 | Very weak size regularization |
| $\lambda_{\theta}$ | 0.01 | Very weak angle regularization |

### 4.2.2 Frame Transformation for Prior

**Critical:** The optimization operates in the **robot frame**, but the previous state $\mathbf{s}^{\text{prev}}$ is stored in the **room frame**. 

To compare correctly, we transform the previous state to robot frame:

$$
\mathbf{s}^{\text{prev}}_{\text{robot}} = T_{\text{room} \to \text{robot}}(\mathbf{s}^{\text{prev}}_{\text{room}}, \mathbf{r}_t)
$$

where $T_{\text{room} \to \text{robot}}$ transforms:
- **Position:** $\mathbf{c}^{\text{robot}} = R(-r_\theta) (\mathbf{c}^{\text{room}} - \mathbf{r}_{xy})$
- **Dimensions:** unchanged
- **Angle:** $\theta^{\text{robot}} = \theta^{\text{room}} - r_\theta$

**Implementation (in `belief.compute_prior_term()`):**
```python
# Transform self.mu (room) to robot frame
mu_prev_robot = transform_object_to_robot_frame(self.mu, robot_pose)

# Compare with mu (robot frame)
diff_pos = mu[:2] - mu_prev_robot[:2]
prior_pos = lambda_pos * torch.sum(diff_pos ** 2)
```

### 4.3 Prior for Dimensions

New objects are initialized with priors favoring typical dimensions:

$$
p(w), p(h), p(d) \sim \mathcal{N}(\mu_{\text{size}}, \sigma_{\text{size}}^2)
$$

with $\mu_{\text{size}} = 0.5$ m (typical object size).

### 4.4 Prior for Angle Alignment

Objects in indoor environments tend to align with room axes (walls). We model this with a **mixture of Gaussians prior** centered at aligned angles:

$$
p(\theta) \propto \sum_{k \in \{0, \pm\pi/2\}} \exp\left( -\frac{(\theta - \mu_k)^2}{2\sigma_\theta^2} \right)
$$

where:
- $\mu_k \in \{0, \pi/2, -\pi/2\}$: aligned angles (0°, 90°, -90°)
- $\sigma_\theta$: standard deviation (default 0.1 rad ≈ 6°)
- $\lambda_\theta = 1/\sigma_\theta^2$: precision

**Prior Energy Term:**

$$
E_\theta(\theta) = \frac{\lambda_\theta}{2} \min_k (\theta - \mu_k)^2
$$

This term:
- **Equals zero** when $\theta$ is aligned (0° or 90°)
- **Increases quadratically** as $\theta$ deviates from alignment
- Acts as a **soft constraint** that biases objects toward room-aligned orientations

---

## 5. Variational Free Energy

### 5.1 Definition

Following the Active Inference framework, we approximate the posterior $p(\mathbf{s} | \mathbf{o})$ with a Gaussian recognition density:

$$
q(\mathbf{s}) = \mathcal{N}(\mathbf{s} \mid \mathbf{\mu}, \Sigma)
$$

The Variational Free Energy (VFE) to minimize is:

$$
F[q, \mathbf{o}] = D_{KL}[q(\mathbf{s}) \| p(\mathbf{s})] - \mathbb{E}_q[\ln p(\mathbf{o} | \mathbf{s})]
$$

### 5.2 Decomposition

$$
F = \underbrace{D_{KL}[q(\mathbf{s}) \| p(\mathbf{s})]}_{\text{Complexity}} - \underbrace{\mathbb{E}_q[\ln p(\mathbf{o} | \mathbf{s})]}_{\text{Accuracy}}
$$

- **Complexity**: Penalizes deviation from prior (Occam's razor)
- **Accuracy**: Rewards explaining observations

### 4.3 Laplace Approximation

Under Gaussian assumptions and point estimation at the mode:

$$
F(\mathbf{s}) \approx \underbrace{\frac{1}{2\sigma_o^2} \sum_{i=1}^{N} \text{SDF}(\mathbf{p}_i, \mathbf{s})^2}_{\text{Likelihood (prediction error)}} + \underbrace{\frac{1}{2} (\mathbf{s} - \mathbf{\mu}_{\text{prior}})^\top \Sigma_{\text{prior}}^{-1} (\mathbf{s} - \mathbf{\mu}_{\text{prior}})}_{\text{Prior regularization}}
$$

---

## 6. Historical Points and Evidence Accumulation

### 5.1 Motivation

Single-frame observations only capture visible surfaces. To build a complete model, we accumulate evidence from multiple viewpoints as the robot moves.

### 5.2 Historical Point Storage

Each historical point is stored with:
- Position in room frame: $\mathbf{p}_{\text{room}} \in \mathbb{R}^3$
- Capture covariance: $\Sigma_{\text{capture}}$
- TCE score: accumulated consistency measure

#### Capture Covariance Formulation

The capture covariance combines two independent sources of uncertainty:

$
\Sigma_{\text{capture}} = \Sigma_{\text{robot}}(t_0) + \sigma_{\text{sdf}}^2 \mathbf{I}
$

where:
- $\Sigma_{\text{robot}}(t_0)$: Robot localization uncertainty at capture time
- $\sigma_{\text{sdf}}^2 = \beta \cdot \text{SDF}(\mathbf{p}, \mathbf{s})^2$: Measurement uncertainty from SDF

**Physical interpretation:**
- $\Sigma_{\text{robot}}(t_0)$: "How well localized was the robot when this point was captured?"
- $\sigma_{\text{sdf}}^2$: "How well did this point fit the model at capture time?"

These are **independent error sources**, hence they add:

$
\sigma_{\text{sdf}}^2 = \beta \cdot \text{SDF}(\mathbf{p}, \mathbf{s}_{t_0})^2
$

The SDF value at capture time modulates the point's reliability:
- $\text{SDF} \approx 0$: Point exactly on surface → low $\sigma_{\text{sdf}}^2$ → high confidence
- $\text{SDF} > 0$: Point outside model → high $\sigma_{\text{sdf}}^2$ → low confidence
- $\text{SDF} < 0$: Point inside model → high $\sigma_{\text{sdf}}^2$ → very low confidence (inconsistent)

**Note:** The capture covariance is **fixed** once the point is stored. It represents the uncertainty at the moment of measurement, not an accumulated uncertainty.

### 5.3 Using Historical Points

When using a historical point at time $t$, we must account for **two sources of uncertainty**:

1. **Capture uncertainty** ($\Sigma_{\text{capture}}$): Fixed at storage time
2. **Current transformation uncertainty**: From room frame to robot frame

#### Why Covariance Composition is Required

Historical points are stored in the **room frame**, but the SDF optimization operates in the **robot frame**. The transformation from room to robot frame depends on the current robot pose, which has uncertainty $\Sigma_{\text{robot}}(t)$.

**Key insight:** The point position is fixed in the room frame, but our knowledge of *where that point is in the robot frame* depends on how well we know the current robot pose.

#### Transformation and Covariance Propagation

1. **Transform to robot frame:**
$$
\mathbf{p}_{\text{robot}}(t) = R(-r_\theta(t))^\top (\mathbf{p}_{\text{room}} - \mathbf{r}_{xy}(t))
$$

2. **Propagate covariance through the transformation:**
$$
\Sigma_{\text{total}} = \Sigma_{\text{capture}} + J(t) \cdot \Sigma_{\text{robot}}(t) \cdot J(t)^\top
$$

where $J(t)$ is the Jacobian of the room-to-robot transformation with respect to robot pose:

$$
J = \frac{\partial \mathbf{p}_{\text{robot}}}{\partial \mathbf{r}} = 
\begin{pmatrix}
-\cos(r_\theta) & -\sin(r_\theta) & -(p_y - r_y)\cos(r_\theta) + (p_x - r_x)\sin(r_\theta) \\
\sin(r_\theta) & -\cos(r_\theta) & -(p_y - r_y)\sin(r_\theta) - (p_x - r_x)\cos(r_\theta)
\end{pmatrix}
$$

3. **Compute weight for optimization:**
$$
w = \frac{1}{1 + \text{tr}(\Sigma_{\text{total}}) + \text{TCE}}
$$

#### Implications

| Robot state at capture ($t_0$) | Robot state now ($t$) | Point weight |
|-------------------------------|----------------------|--------------|
| Well localized (small $\Sigma$) | Well localized | **High** |
| Well localized | Poorly localized | Low |
| Poorly localized | Well localized | Low |
| Poorly localized | Poorly localized | **Very low** |

This ensures that historical points only contribute strongly when **both** the capture and current localization are reliable.

### 5.4 Temporal Consistency Error (TCE)

Inspired by Expected Free Energy (EFE) for future planning, we introduce **Temporal Consistency Error (TCE)** for retrospective evidence evaluation:

$$
\text{TCE}_i = \sum_{\tau=t_0}^{t} \alpha^{t-\tau} \cdot w_\tau \cdot \text{SDF}(\mathbf{p}_i, \mathbf{s}_\tau)^2
$$

where:
- $\alpha \in (0,1)$: temporal decay factor
- $w_\tau = 1 / (1 + \text{tr}(\Sigma_{\text{robot}}(\tau)))$: weight based on robot certainty

**Interpretation:**
- $\text{TCE} \approx 0$: Point consistently on surface → **high evidence value**
- $\text{TCE} \gg 0$: Point inconsistent with model → **likely noise, low value**

### 5.5 Temporal Symmetry: EFE and TCE

Active Inference provides a principled framework for action selection via Expected Free Energy (EFE), which evaluates the quality of future policies. TCE complements this by evaluating the quality of past evidence.

| Concept | Direction | Question Answered |
|---------|-----------|-------------------|
| **EFE** | Future → Present | "What actions will reduce my uncertainty?" |
| **TCE** | Past → Present | "What evidence has been consistently reliable?" |

Together, they form a **temporally symmetric** view of inference:

$$
\text{Present belief} = f(\underbrace{\text{TCE-weighted past evidence}}_{\text{memory}}, \underbrace{\text{EFE-guided future actions}}_{\text{planning}})
$$

**Key insight:** Memory in Active Inference is not passive storage, but **evidence weighted by historical consistency and certainty**. Points with low TCE are "trusted memories" that anchor the belief, while points with high TCE are "unreliable memories" that are downweighted or forgotten.

### 5.6 TCE Integration with Uniform Surface Coverage

Historical points are organized in spatial bins for uniform surface coverage:
- **24 angular bins** around the object (XY plane)
- **10 height bins** (Z axis, 10cm each)
- **Maximum points per bin**: `max_historical / (24 × 10)`

This ensures that points are distributed uniformly across the visible surface, preventing over-representation of frequently observed areas.

#### Quality Metric for Bin Selection

Within each bin, points compete based on a **quality score** that prioritizes geometric information content:

$$
Q_i = \text{tr}(\Sigma_{\text{capture},i}) + \text{TCE}_i - \gamma \cdot E_i
$$

where:
- $\text{tr}(\Sigma_{\text{capture},i})$: trace of capture covariance (uncertainty)
- $\text{TCE}_i$: Temporal Consistency Error (historical consistency)
- $E_i \in [0, 1]$: **edge/corner score** (geometric value)
- $\gamma$: edge bonus weight (default 0.3)

**Key insight:** Lower $Q$ = better point. The edge bonus $\gamma \cdot E_i$ is **subtracted**, giving priority to edge/corner points.

#### Edge/Corner Detection Algorithm

A point's edge score is computed by checking proximity to multiple object faces:

```python
def _compute_edge_score(points):
    # Transform to object-local frame
    # Compute distance to each face
    dist_x = |local_x| - half_w  # Distance to X faces
    dist_y = |local_y| - half_h  # Distance to Y faces  
    dist_z = |local_z| - half_d  # Distance to Z faces
    
    # Count faces within threshold (0.05m)
    close_x = dist_x < threshold
    close_y = dist_y < threshold
    close_z = dist_z < threshold
    faces_close = close_x + close_y + close_z
    
    # Base score: 0=flat, 0.5=edge, 1.0=corner
    edge_score = (faces_close - 1) / 2
    
    # Proximity bonus for being very close
    min_dist = min(dist_x, dist_y, dist_z)
    proximity_bonus = exp(-min_dist / 0.02) × 0.2
    
    return clamp(edge_score + proximity_bonus, 0, 1)
```

| Location | Faces Close | Edge Score E | Geometric Value |
|----------|-------------|--------------|-----------------|
| Flat face | 1 | 0.0 | Low - redundant |
| Edge | 2 | 0.5 | Medium - constrains 2 dimensions |
| Corner | 3 | 1.0 | **High** - constrains all dimensions |

**Rationale:** Corner and edge points provide stronger constraints on object dimensions because they simultaneously touch multiple faces. A single corner observation constrains width, height, and depth, whereas a flat face point only confirms one dimension.

#### Effect of Quality Factors on Point Retention

| Σ_capture | TCE | Edge Score | Q | Outcome |
|-----------|-----|------------|---|---------|
| Low | Low | High (corner) | **Very Low** | **Kept** - best anchor |
| Low | Low | Low (flat) | Low | Kept - trusted |
| Low | High | High | Medium | May be replaced |
| High | Low | High | Medium | May be replaced |
| High | High | Low | **High** | **Replaced first** |

This ensures:
1. **Spatial uniformity**: Points cover all visible faces
2. **Temporal consistency**: Reliable points accumulate, noisy points are forgotten
3. **Geometric priority**: Edge/corner points are preserved over flat-face points
4. **Capture quality**: Well-localized measurements are preferred

---

## 7. Complete Optimization Objective

### 6.1 Full Free Energy with Historical Points

$$
F(\mathbf{s}) = \underbrace{\frac{1}{2\sigma_o^2} \sum_{i=1}^{N_{\text{current}}} \text{SDF}(\mathbf{p}_i^{\text{curr}}, \mathbf{s})^2}_{\text{Current observations}}
+ \underbrace{\frac{1}{2} \sum_{j=1}^{N_{\text{hist}}} w_j \cdot \text{SDF}(\mathbf{p}_j^{\text{hist}}, \mathbf{s})^2}_{\text{Historical evidence}}
+ \underbrace{F_{\text{prior}}(\mathbf{s}, \mathbf{s}^{\text{prev}})}_{\text{State regularization}}
$$

where:
- $w_j = 1 / (1 + \text{tr}(\Sigma_{\text{total},j}) + \text{TCE}_j)$: weight for historical point $j$

**State Prior Term (for static objects):**

$$
F_{\text{prior}} = \frac{\lambda_{\text{pos}}}{2} \|\mathbf{s}_{xy} - \mathbf{s}_{xy}^{\text{prev}}\|^2 + \frac{\lambda_{\text{size}}}{2} \|\mathbf{s}_{\text{dim}} - \mathbf{s}_{\text{dim}}^{\text{prev}}\|^2
$$

**Important:** Both $\mathbf{s}$ and $\mathbf{s}^{\text{prev}}$ must be in the same frame (robot frame) for the comparison. The previous state is transformed from room frame using:

$$
\mathbf{s}^{\text{prev}}_{\text{robot}} = T_{\text{room} \to \text{robot}}(\mathbf{s}^{\text{prev}}_{\text{room}}, \mathbf{r}_t)
$$

### 6.2 Gradient-Based Optimization

The posterior mode is found via gradient descent:

$$
\mathbf{s}^{(k+1)} = \mathbf{s}^{(k)} - \eta \nabla_{\mathbf{s}} F(\mathbf{s}^{(k)})
$$

Gradients are computed via automatic differentiation (PyTorch).

### 6.3 Posterior Covariance

After convergence, the posterior covariance is approximated by the inverse Hessian:

$$
\Sigma_{\text{post}}^{-1} = \nabla^2_{\mathbf{s}} F(\mathbf{s}) \big|_{\mathbf{s} = \mathbf{\mu}_{\text{post}}}
$$

---

## 8. Bayesian Model Selection for Object Types

When a new cluster is detected, its geometric type (table, chair, box, etc.) is unknown. We treat **model identity as a discrete latent variable** within the Active Inference framework. This allows uncertainty over object structure to be inferred, propagated, and resolved in a principled manner.

### 7.1 Extended Generative Model

The generative model includes a discrete variable $m \in \mathcal{M}$ for object type:

$$
p(o, s, \theta, m) = p(o | s, \theta_m, m) \cdot p(\theta_m | m) \cdot p(m) \cdot p(s)
$$

where:
- $m \in \mathcal{M} = \{\text{table}, \text{chair}, \text{box}, ...\}$ — object type (discrete)
- $\theta_m$ — model-specific continuous parameters (position, dimensions, angle)
- $p(m)$ — prior over object types (e.g., uniform)
- $p(\theta_m | m)$ — prior over parameters given model type

### 7.2 Variational Posterior (Structured Factorization)

The approximate posterior factorizes as:

$$
q(s, \theta, m) = q(s) \cdot q(m) \cdot q(\theta_m | m)
$$

where:
- $q(m)$ is a **categorical distribution** over model types
- $q(\theta_m | m)$ are model-conditional Gaussian posteriors

This factorization allows the system to maintain uncertainty over both:
1. **Which model is correct** → $q(m)$
2. **What its parameters are** → $q(\theta_m | m)$

### 7.3 Model Evidence via VFE

For each model $m$, we compute its Variational Free Energy:

$$
F_m = \mathbb{E}_{q(\theta_m|m)} \left[ \ln q(\theta_m|m) - \ln p(o, s, \theta_m | m) \right]
$$

In practice, $F_m$ is the final loss from optimizing each model's belief:

$$
F_m = \underbrace{\frac{1}{N}\sum_{i=1}^{N} w_i \cdot \text{SDF}_m(\mathbf{p}_i, \theta_m)^2}_{\text{Likelihood}} + \underbrace{\frac{\lambda}{2}\|\theta_m - \theta_m^{\text{prev}}\|^2}_{\text{Prior}}
$$

**Key insight:** The model with lower VFE better explains the observations with less complexity.

### 7.4 Posterior over Model Identity

The posterior over which model is correct follows from Bayesian model comparison:

$$
q(m) \propto p(m) \cdot \exp(-F_m / T)
$$

where $T$ is a temperature parameter (default 1.0). Using **softmax normalization**:

$$
q(m_i) = \frac{p(m_i) \cdot \exp(-F_{m_i}/T)}{\sum_j p(m_j) \cdot \exp(-F_{m_j}/T)}
$$

**Interpretation:**
- Lower VFE → Higher probability
- This is Bayesian Occam's razor: simpler models that explain the data are preferred

### 7.5 Entropy as Uncertainty Measure

The entropy of $q(m)$ measures uncertainty over model identity:

$$
H[q(m)] = -\sum_m q(m) \ln q(m)
$$

| $q(\text{table})$ | $q(\text{chair})$ | $H[q(m)]$ | Interpretation |
|-------------------|-------------------|-----------|----------------|
| 0.50 | 0.50 | 0.69 nats | Maximum uncertainty |
| 0.70 | 0.30 | 0.61 nats | Moderate uncertainty |
| 0.85 | 0.15 | 0.42 nats | High confidence |
| 0.95 | 0.05 | 0.20 nats | Very confident |
| 0.99 | 0.01 | 0.06 nats | Almost certain |

### 7.6 Commitment Criteria

A belief **commits** to a single model when uncertainty is resolved. We use three criteria:

**Criterion 1: Posterior concentration**
$$
\max_m q(m) > 1 - \epsilon \quad (\text{e.g., } > 0.85)
$$

**Criterion 2: Entropy threshold**
$$
H[q(m)] < H_{\min} \quad (\text{e.g., } < 0.3 \text{ nats})
$$

**Criterion 3: Hysteresis (stability)**
Same model must win for $N$ consecutive frames (e.g., $N=10$).

Commitment occurs when: `(Criterion 1 OR Criterion 2) AND Criterion 3`

### 7.7 Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Frame t: New cluster detected                              │
│                                                             │
│  1. Create MultiModelBelief with:                          │
│     - TableBelief (θ_table initialized from cluster)       │
│     - ChairBelief (θ_chair initialized from cluster)       │
│     - q(m) = [0.5, 0.5] (uniform prior)                    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Frame t+1, t+2, ...: Update                               │
│                                                             │
│  For each model m:                                         │
│    1. Optimize θ_m via VFE minimization → F_m              │
│    2. Update historical points                             │
│    3. Update TCE                                           │
│                                                             │
│  Then:                                                     │
│    4. Update q(m) = softmax(-F_m) · p(m)                   │
│    5. Compute H[q(m)]                                       │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Evaluate Commitment                                        │
│                                                             │
│  if (H[q(m)] < 0.3 OR max(q) > 0.85) AND hysteresis_ok:   │
│      → Discard losing model                                │
│      → Continue with committed model only                  │
│  else:                                                      │
│      → Maintain both models active                         │
└─────────────────────────────────────────────────────────────┘
```

### 7.8 Implementation

The model selection algorithm is implemented in two files:

**`model_selector.py`** - Core algorithm:

| Class | Description |
|-------|-------------|
| `MultiModelBelief` | Maintains parallel hypotheses for one object |
| `ModelHypothesis` | Single model hypothesis with VFE tracking |
| `ModelSelector` | Factory for creating multi-model beliefs |
| `ModelSelectorConfig` | Configuration parameters |

**`multi_model_manager.py`** - Integration with perception:

| Class | Description |
|-------|-------------|
| `MultiModelManager` | BeliefManager using ModelSelector |

### 7.9 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_priors` | 0.5/0.5 | Prior $p(m)$ for each type |
| `vfe_temperature` | 1.0 | Temperature $T$ in softmax |
| `entropy_threshold` | 0.3 | $H_{\min}$ for commitment |
| `concentration_threshold` | 0.85 | $1-\epsilon$ for commitment |
| `hysteresis_frames` | 10 | Consecutive frames needed |
| `min_frames_before_commit` | 20 | Minimum observation period |
| `debug_interval` | 20 | Frames between debug output |

### 7.10 Advantages of This Approach

1. **Principled**: Follows directly from Active Inference theory
2. **Soft fusion**: Maintains uncertainty while useful
3. **Automatic Occam's razor**: VFE penalizes unnecessary complexity
4. **Non-heuristic**: Selection emerges from the mathematics
5. **Extensible**: Easy to add more object types

---

## 9. Data Association

### 8.1 Clustering

LIDAR points are clustered using DBSCAN with parameters:
- $\epsilon$: neighborhood radius (meters)
- $\text{minPts}$: minimum points per cluster

### 8.2 Cost Matrix

For cluster $k$ and belief $j$, the association cost is:

$$
C_{kj} = \frac{1}{|\mathcal{C}_k|} \sum_{\mathbf{p} \in \mathcal{C}_k} \text{SDF}(\mathbf{p}, \mathbf{\mu}_j)^2
$$

### 8.3 Hungarian Algorithm

Optimal assignment minimizes total cost:

$$
\mathcal{A}^* = \arg\min_{\mathcal{A}} \sum_{(k,j) \in \mathcal{A}} C_{kj}
$$

subject to one-to-one matching constraints.

---

## 10. Belief Lifecycle

### 10.1 Initialization

New beliefs are created from unmatched clusters:
- Position: cluster centroid
- Dimensions: from cluster extent, regularized by prior $\mathcal{N}(0.5, 0.2^2)$
- Covariance: initial uncertainty values

### 10.2 Update

Matched beliefs are updated by minimizing VFE with current and historical observations.

### 10.3 Confidence Dynamics

The confidence score $\kappa \in [0, 1]$ tracks belief reliability:

$$
\kappa_{t+1} = \begin{cases}
\min(\kappa_t + \Delta\kappa, 1) & \text{if matched (observed)} \\
\gamma_{\text{confirmed}} \cdot \kappa_t & \text{if unmatched AND confirmed AND } (t - t_{\text{last}}) > T_{\text{grace}} \\
\gamma \cdot \kappa_t & \text{if unmatched AND not confirmed}
\end{cases}
$$

where:
- $\Delta\kappa = 0.15$: confidence boost on observation
- $\gamma = 0.90$: decay factor for unconfirmed beliefs
- $\gamma_{\text{confirmed}} = 0.98$: **slower** decay for confirmed beliefs
- $T_{\text{grace}} = 50$: grace frames before confirmed beliefs start decaying

**Confirmation:** A belief becomes **confirmed** when $\kappa \geq \kappa_{\text{confirmed}} = 0.70$

### 10.4 Confirmed vs Unconfirmed Beliefs

| Property | Unconfirmed | Confirmed |
|----------|-------------|-----------|
| Decay rate | $\gamma = 0.90$ | $\gamma_{\text{confirmed}} = 0.98$ |
| Grace period | None | $T_{\text{grace}} = 50$ frames |
| Removal threshold | $\kappa < 0.20$ | $\kappa < 0.20$ |
| Typical lifetime (unseen) | ~15 frames | ~200+ frames |

**Rationale:** Confirmed objects (e.g., a TV that has been observed many times) should not disappear immediately when temporarily occluded. The grace period and slower decay allow the robot to pass in front of an object without losing track of it.

### 10.5 Lifecycle Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Confidence boost | $\Delta\kappa$ | 0.15 | Increase on observation |
| Decay (unconfirmed) | $\gamma$ | 0.90 | Decay factor when not observed |
| Decay (confirmed) | $\gamma_{\text{confirmed}}$ | 0.98 | Slower decay for confirmed objects |
| Removal threshold | $\kappa_{\text{threshold}}$ | 0.20 | Below this, belief is removed |
| Confirmation threshold | $\kappa_{\text{confirmed}}$ | 0.70 | Above this, belief is confirmed |
| Grace frames | $T_{\text{grace}}$ | 50 | Frames before confirmed beliefs decay |
| Initial confidence | $\kappa_0$ | 0.30 | Confidence at creation |

### 10.6 Association Distance

Each object type has a **maximum association distance** that determines how far a cluster can be from a belief's predicted position to be considered a match:

| Object Type | Max Association Distance | Rationale |
|-------------|-------------------------|-----------|
| Table | 0.8 m | Standard objects |
| Chair | 0.8 m | Standard objects |
| TV | **1.5 m** | Wall-mounted, viewpoint changes more dramatically |

**Why TV has larger association distance:** When the robot moves around a room, wall-mounted objects like TVs can appear to shift significantly due to perspective changes. A larger association radius prevents losing track of the TV during robot motion.

### 10.7 Removal

Beliefs with $\kappa < \kappa_{\text{threshold}}$ are removed from the active set.

---

## 11. Coordinate Frames

### 9.1 Frame Definitions

1. **Robot frame**: Origin at robot, X+ right, Y+ forward, Z+ up
2. **Room frame**: Origin at room center, fixed orientation
3. **Box frame**: Origin at box center, Y+ toward robot (dynamic)

### 9.2 Transformations

**Robot → Room:**
$$
\mathbf{p}_{\text{room}} = R(r_\theta) \mathbf{p}_{\text{robot}} + \begin{pmatrix} r_x \\ r_y \\ 0 \end{pmatrix}
$$

**Room → Robot:**
$$
\mathbf{p}_{\text{robot}} = R(-r_\theta)^\top (\mathbf{p}_{\text{room}} - \mathbf{r})
$$

---

## 12. Summary of Key Equations

| Concept | Equation |
|---------|----------|
| State vector (box) | $\mathbf{s} = (c_x, c_y, w, h, d, \theta)^\top$ |
| State vector (table) | $\mathbf{s} = (c_x, c_y, w, h, h_{\text{table}}, l_{\text{leg}}, \theta)^\top$ |
| State vector (chair) | $\mathbf{s} = (c_x, c_y, w_s, d_s, h_s, h_b, \theta)^\top$ |
| State vector (TV) | $\mathbf{s} = (c_x, c_y, w, h, z_{\text{base}}, \theta)^\top$ |
| **Total VFE** | $F = F_{\text{present}} + F_{\text{past}} + F_{\text{prior}}$ |
| Present (current LIDAR) | $F_{\text{present}} = \frac{1}{N}\sum_{i} \text{SDF}(\mathbf{p}_i^{\text{now}}, \mathbf{s})^2$ |
| Past (historical points) | $F_{\text{past}} = \sum_{j} w_j \cdot \text{SDF}(\mathbf{p}_j^{\text{hist}}, \mathbf{s})^2$ |
| Historical weight | $w_j = 1 / (1 + \text{tr}(\Sigma_{\text{total},j}) + \text{TCE}_j)$ |
| State prior | $F_{\text{prior}} = \frac{\lambda_{\text{pos}}}{2}\|\Delta\mathbf{c}\|^2 + \frac{\lambda_{\text{size}}}{2}\|\Delta\mathbf{d}\|^2 + \frac{\lambda_{\theta}}{2}\Delta\theta^2$ |
| Capture covariance | $\Sigma_{\text{capture}} = \Sigma_{\text{robot}}(t_0) + \beta \cdot \text{SDF}^2 \cdot \mathbf{I}$ |
| Propagated covariance | $\Sigma_{\text{total}} = \Sigma_{\text{capture}} + J \Sigma_{\text{robot}}(t) J^\top$ |
| TCE (temporal consistency) | $\text{TCE}_j = \sum_{\tau} \alpha^{t-\tau} w_\tau \text{SDF}(\mathbf{p}_j, \mathbf{s}_\tau)^2$ |
| Frame transform | $\mathbf{s}^{\text{prev}}_{\text{robot}} = T(\mathbf{s}^{\text{prev}}_{\text{room}}, \mathbf{r}_t)$ |
| **Confidence update (observed)** | $\kappa_{t+1} = \min(\kappa_t + \Delta\kappa, 1)$ |
| **Confidence decay (unconfirmed)** | $\kappa_{t+1} = \gamma \cdot \kappa_t$ |
| **Confidence decay (confirmed)** | $\kappa_{t+1} = \gamma_{\text{confirmed}} \cdot \kappa_t$ (after $T_{\text{grace}}$) |

**Terminology:**
- **TCE (Temporal Consistency Error)**: Measures how consistently a historical point has been on the model surface over time. Also known as RFE (Remembered Free Energy) in the code.
- Low TCE → point consistently on surface → high weight
- High TCE → point inconsistent → low weight (or removed)

---

## 13. Supported Object Types

The framework supports multiple object types, each with its own SDF and prior functions. All objects share the same Active Inference optimization framework.

### 11.1 Box (6 parameters)

**State:** $\mathbf{s} = (c_x, c_y, w, h, d, \theta)$

| Parameter | Description |
|-----------|-------------|
| $c_x, c_y$ | Center position in XY plane |
| $w$ | Width (X dimension) |
| $h$ | Height (Y dimension) |
| $d$ | Depth (Z dimension) |
| $\theta$ | Rotation angle around Z |

**Files:** `box_belief.py`, `box_manager.py`

### 11.2 Table (7 parameters)

**State:** $\mathbf{s} = (c_x, c_y, w, h, h_{\text{table}}, l_{\text{leg}}, \theta)$

| Parameter | Description |
|-----------|-------------|
| $c_x, c_y$ | Center position in XY plane |
| $w, h$ | Table top dimensions |
| $h_{\text{table}}$ | Height of table surface from floor |
| $l_{\text{leg}}$ | Length of legs |
| $\theta$ | Rotation angle |

**Fixed constants:**
- Top thickness: 0.03 m
- Leg radius: 0.025 m

**SDF:** Union of box (top) and 4 cylinders (legs)

**Files:** `table_belief.py`, `table_manager.py`

### 11.3 Chair (7 parameters)

**State:** $\mathbf{s} = (c_x, c_y, w_s, d_s, h_s, h_b, \theta)$

| Parameter | Description |
|-----------|-------------|
| $c_x, c_y$ | Center position |
| $w_s, d_s$ | Seat width and depth |
| $h_s$ | Seat height from floor |
| $h_b$ | Backrest height above seat |
| $\theta$ | Rotation angle |

**Fixed constants:**
- Seat thickness: 0.05 m
- Backrest thickness: 0.05 m

**SDF:** Union of box (seat) and box (backrest)

**Files:** `chair_belief.py`, `chair_manager.py`

### 13.4 TV (6 parameters)

**State:** $\mathbf{s} = (c_x, c_y, w, h, z_{\text{base}}, \theta)$

| Parameter | Description |
|-----------|-------------|
| $c_x, c_y$ | Center position (floor projection) |
| $w$ | Screen width |
| $h$ | Screen height |
| $z_{\text{base}}$ | Height from floor to bottom of TV |
| $\theta$ | Rotation angle |

**Fixed constants:**
- Depth (thickness): 0.05 m

**SDF:** Standard 3D box SDF with fixed depth

**Priors:**
- Screen aspect ratio: $w/h \approx 2.0$
- Strong angle alignment with walls

**Files:** `tv/belief.py`, `tv/sdf.py`

### 13.5 Cylinder (4 parameters)

**State:** $\mathbf{s} = (c_x, c_y, r, h)$

| Parameter | Description |
|-----------|-------------|
| $c_x, c_y$ | Center position |
| $r$ | Radius |
| $h$ | Height |

**SDF:** Standard cylinder SDF (no angle prior, rotationally symmetric)

### 13.6 Adding New Object Types

See `ADDING_NEW_OBJECTS.md` for detailed instructions on implementing new object types.

---

## 14. Implementation Notes

### 14.1 Numerical Stability

- SDF gradients can be unstable near corners; smoothing helps
- Covariance matrices must remain positive definite
- Use Cholesky decomposition for matrix inversions

### 14.2 Computational Efficiency

- Batch SDF computation on GPU (PyTorch CUDA)
- Limit historical points per belief (max 500)
- Uniform surface coverage via angular/height binning

### 14.3 Hyperparameters

| Parameter | Symbol | Typical Value |
|-----------|--------|---------------|
| Observation noise | $\sigma_o$ | 0.05 m |
| Process noise (position) | $\sigma_{xy}$ | 0.02 m |
| Process noise (size) | $\sigma_{\text{size}}$ | 0.01 m |
| Prior size mean | $\mu_{\text{size}}$ | 0.5 m |
| Prior size std | $\sigma_{\text{size}}$ | 0.2 m |
| Confidence decay (unconfirmed) | $\gamma$ | 0.90 |
| Confidence decay (confirmed) | $\gamma_{\text{confirmed}}$ | 0.98 |
| Confidence boost | $\Delta\kappa$ | 0.15 |
| Confidence threshold | $\kappa_{\text{threshold}}$ | 0.20 |
| Confirmation threshold | $\kappa_{\text{confirmed}}$ | 0.70 |
| Grace frames (confirmed) | $T_{\text{grace}}$ | 50 |
| Prior precision (position) | $\lambda_{\text{pos}}$ | 0.05 |
| Prior precision (size) | $\lambda_{\text{size}}$ | 0.02 |
| Prior precision (angle) | $\lambda_{\theta}$ | 0.01 |
| SDF smooth parameter | $k$ | 0.02 m |
| Inside point scale | $\alpha_{\text{inside}}$ | 0.5 |
| Optimization iterations | - | 10 |
| Learning rate | $\eta$ | 0.05 |
| Gradient clip | - | 2.0 |
| Angle gradient multiplier (chair) | - | 100.0 |

### 14.4 Historical Points Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_historical_points` | Maximum stored points per belief | 600 |
| `num_angle_bins` | Angular bins for uniform coverage | 24 |
| `num_z_bins` | Height bins | 10 |
| `edge_proximity_threshold` | Distance to consider "near face" | 0.05 m |
| `edge_bonus_weight` | Weight for edge/corner priority | 0.3 |
| `rfe_alpha` | RFE temporal decay | 0.98 |
| `rfe_max_threshold` | Max RFE before point removal | 2.0 |
| `sdf_threshold_for_storage` | Max SDF to store point | 0.08 m |

---

## 15. References

1. Friston, K. (2010). The free-energy principle: a unified brain theory?
2. Lanillos, P. et al. (2021). Active Inference in Robotics and Artificial Agents
3. Da Costa, L. et al. (2022). Active Inference on Discrete State-Spaces

---

*Document version: 2.2*
*Last updated: February 8, 2026*
*Renamed from ACTIVE_INFERENCE_MATH.md to OBJECT_INFERENCE_MATH.md*

**Changes in v2.2:**
- Added **TV object type** (Section 2.1.4, 3.4, 13.4) with fixed depth parameter
- **Updated Belief Lifecycle** (Section 10) with new confidence dynamics:
  - Differentiated decay rates for confirmed vs unconfirmed beliefs
  - Added grace period ($T_{\text{grace}} = 50$) before confirmed beliefs decay
  - Slower decay ($\gamma_{\text{confirmed}} = 0.98$) for confirmed objects
- Added **association distance** per object type (Section 10.6)
- Updated hyperparameters table (Section 14.3) with new lifecycle values
- Clarified fixed vs free parameters concept (TV depth example)

**Changes in v2.1:**
- Added complete State Space section (2) for all three object types (Box, Table, Chair)
- Added detailed SDF section (3) with mathematical formulations for each object type
- Included ASCII diagrams for Table and Chair geometry
- Renumbered all sections for consistency

**Changes in v2.0:**
- Renamed document to reflect broader scope (not just boxes)
- Expanded Section 8: Bayesian Model Selection with detailed math
- Added entropy table for uncertainty interpretation
- Added execution flow diagram
- Added implementation details for model_selector.py and multi_model_manager.py
