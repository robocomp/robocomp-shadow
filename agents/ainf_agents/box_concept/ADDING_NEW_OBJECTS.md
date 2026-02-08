# Adding New Objects to the Active Inference System

This guide documents the process of adding a new object type (e.g., TV, chair, table) to the multi-model belief system.

---

## Overview

Each object type requires:
1. **SDF (Signed Distance Function)** - defines the object's geometry
2. **Belief class** - manages the probabilistic state estimation
3. **Manager class** - handles lifecycle and DSR integration (optional)
4. **Visualizer support** - Qt3D rendering

---

## Directory Structure

Create a new folder under `src/objects/`:

```
src/objects/
├── sdf_constants.py        # Shared constants (SDF_SMOOTH_K, etc.)
├── your_object/
│   ├── __init__.py
│   ├── sdf.py              # SDF and prior functions
│   ├── belief.py           # Belief class and config
│   └── manager.py          # Manager class (optional)
├── table/
├── chair/
└── tv/
```

---

## Step 1: Define the State Space

Decide on the parameters that define your object.

### Example - TV (6 parameters):
```python
# State: [cx, cy, width, height, z_base, theta]
# - cx, cy: center position in XY plane
# - width: horizontal extent
# - height: vertical extent  
# - z_base: height from floor (for wall-mounted objects)
# - theta: rotation angle
```

### Example - Table (7 parameters):
```python
# State: [cx, cy, width, depth, table_height, leg_length, theta]
```

### Example - Chair (7 parameters):
```python
# State: [cx, cy, seat_width, seat_depth, seat_height, back_height, theta]
```

**Guidelines:**
- Keep state dimension minimal (6-8 parameters typical)
- **Fix parameters that don't need optimization** (e.g., TV depth = 5cm fixed)
- Include position (cx, cy), dimensions, and angle (theta) at minimum
- Angle (theta) should be the **last** parameter

---

## Step 2: Create the SDF (`sdf.py`)

### 2.1 Define Constants

```python
#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
YourObject SDF and Prior Functions

State: [cx, cy, width, height, z_base, theta] (6 parameters)
- cx, cy: center position
- width, height: dimensions
- z_base: height from floor
- theta: rotation angle
"""

import torch
import numpy as np
from src.objects.sdf_constants import SDF_SMOOTH_K, SDF_INSIDE_SCALE

# Object parameters
YOUR_OBJECT_PARAM_COUNT = 6
YOUR_OBJECT_PARAM_NAMES = ['cx', 'cy', 'width', 'height', 'z_base', 'theta']

# Constraints
YOUR_OBJECT_MIN_WIDTH = 0.4
YOUR_OBJECT_MAX_WIDTH = 1.2
YOUR_OBJECT_MIN_HEIGHT = 0.2
YOUR_OBJECT_MAX_HEIGHT = 0.7
# ... etc
```

### 2.2 Implement `compute_<object>_sdf()`

```python
def compute_your_object_sdf(points_xyz: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    Compute SDF for points relative to the object.
    
    Args:
        points_xyz: [N, 3] points (x, y, z)
        params: [D] tensor with object parameters
        
    Returns:
        [N] SDF values (positive=outside, negative=inside, zero=surface)
    """
    # 1. Extract parameters
    cx, cy = params[0], params[1]
    width, height = params[2], params[3]
    z_base = params[4]
    theta = params[5]
    
    # 2. Transform points to object local frame
    cos_t = torch.cos(-theta)
    sin_t = torch.sin(-theta)
    px = points_xyz[:, 0] - cx
    py = points_xyz[:, 1] - cy
    local_x = px * cos_t - py * sin_t
    local_y = px * sin_t + py * cos_t
    
    # 3. Compute half dimensions
    half_w = width / 2
    half_h = height / 2
    
    # 4. Compute distance to each face
    dx = torch.abs(local_x) - half_w
    dy = torch.abs(local_y) - half_depth
    
    # For Z: object at z_base, center at z_base + half_h
    z_center = z_base + half_h
    local_z = points_xyz[:, 2] - z_center
    dz = torch.abs(local_z) - half_h
    
    # 5. Outside: Euclidean distance to surface
    outside = torch.linalg.norm(torch.stack([
        torch.clamp(dx, min=0),
        torch.clamp(dy, min=0),
        torch.clamp(dz, min=0)
    ], dim=-1), dim=-1)
    
    # 6. Inside: smooth min approximation
    k = SDF_SMOOTH_K
    inside_scale = SDF_INSIDE_SCALE
    
    is_inside = (dx < 0) & (dy < 0) & (dz < 0)
    inside_term = torch.zeros_like(dx)
    
    if is_inside.any():
        dx_in, dy_in, dz_in = dx[is_inside], dy[is_inside], dz[is_inside]
        stacked = torch.stack([dx_in / k, dy_in / k, dz_in / k], dim=-1)
        smooth_min = k * torch.logsumexp(-stacked, dim=-1) * (-1)
        inside_term[is_inside] = inside_scale * smooth_min
    
    return outside + inside_term
```

### 2.3 Implement `compute_<object>_priors()`

```python
def compute_your_object_priors(params: torch.Tensor, config, robot_pose=None) -> torch.Tensor:
    """Compute prior energy terms specific to this object type."""
    total_prior = torch.tensor(0.0, dtype=params.dtype, device=params.device)
    
    width, height = params[2], params[3]
    theta = params[5]
    
    # Transform angle to room frame if robot_pose provided
    if robot_pose is not None:
        theta_room = theta + robot_pose[2]
    else:
        theta_room = theta
    theta_room = torch.atan2(torch.sin(theta_room), torch.cos(theta_room))
    
    # 1. ANGLE ALIGNMENT PRIOR (objects align with walls: 0°, ±90°, 180°)
    sigma_angle = getattr(config, 'angle_alignment_sigma', 0.1)
    weight_angle = getattr(config, 'angle_alignment_weight', 1.0)
    precision_angle = 1.0 / (sigma_angle ** 2)
    
    dist_to_0 = theta_room ** 2
    dist_to_pos90 = (theta_room - np.pi/2) ** 2
    dist_to_neg90 = (theta_room + np.pi/2) ** 2
    dist_to_180 = (torch.abs(theta_room) - np.pi) ** 2
    
    min_angle_dist = torch.minimum(
        torch.minimum(dist_to_0, dist_to_180),
        torch.minimum(dist_to_pos90, dist_to_neg90)
    )
    total_prior = total_prior + weight_angle * 0.5 * precision_angle * min_angle_dist
    
    # 2. ASPECT RATIO PRIOR (optional)
    target_aspect = getattr(config, 'target_aspect', 2.0)
    current_aspect = width / (height + 0.01)
    aspect_error = (current_aspect - target_aspect) / target_aspect
    if torch.abs(aspect_error) > 0.3:
        total_prior = total_prior + 0.5 * (aspect_error ** 2)
    
    return total_prior
```

---

## Step 3: Create the Belief Class (`belief.py`)

### 3.1 Define Configuration

```python
from dataclasses import dataclass
from src.belief_core import BeliefConfig

@dataclass
class YourObjectBeliefConfig(BeliefConfig):
    """Configuration for your object beliefs."""
    
    # Size priors
    prior_width: float = 1.0
    prior_height: float = 0.5
    prior_size_std: float = 0.1
    
    # Dimension constraints
    min_width: float = 0.4
    max_width: float = 1.2
    min_height: float = 0.2
    max_height: float = 0.7
    min_z_base: float = 0.5
    max_z_base: float = 1.5
    
    # Prior weights
    angle_alignment_weight: float = 1.5
    angle_alignment_sigma: float = 0.08
    target_aspect: float = 2.0
    
    # Clustering
    cluster_eps: float = 0.20
    min_cluster_points: int = 15
    
    # Historical points
    max_historical_points: int = 500
    sdf_threshold_for_storage: float = 0.06
    beta_sdf: float = 1.0
    
    # RFE parameters
    rfe_alpha: float = 0.98
    rfe_max_threshold: float = 2.0
```

### 3.2 Implement Belief Class

```python
from typing import Optional, Tuple
import torch
import numpy as np
from src.belief_core import Belief, DEVICE, DTYPE
from src.objects.your_object.sdf import compute_your_object_sdf

class YourObjectBelief(Belief):
    """Gaussian belief over object parameters."""
    
    STATE_DIM = 6  # MUST match parameter count
    
    @property
    def state_dim(self) -> int:
        return self.STATE_DIM
    
    @property
    def position(self) -> Tuple[float, float]:
        return (self.mu[0].item(), self.mu[1].item())
    
    @property
    def cx(self) -> float:
        return self.mu[0].item()
    
    @property
    def cy(self) -> float:
        return self.mu[1].item()
    
    @property
    def width(self) -> float:
        return self.mu[2].item()
    
    @property
    def height(self) -> float:
        return self.mu[3].item()
    
    @property
    def z_base(self) -> float:
        return self.mu[4].item()
    
    @property
    def angle(self) -> float:
        return self.mu[5].item()  # Always last
    
    def to_dict(self) -> dict:
        """Convert to dictionary for visualization."""
        base_dict = super().to_dict()
        base_dict.update({
            'type': 'your_object',
            'cx': self.cx,
            'cy': self.cy,
            'width': self.width,
            'height': self.height,
            'z_base': self.z_base,
            'angle': self.angle,
        })
        return base_dict
    
    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """Compute SDF for points."""
        assert points.shape[1] == 3, f"Points must be 3D, got {points.shape}"
        return compute_your_object_sdf(points, self.mu)
    
    def compute_prior_term(self, mu: torch.Tensor, robot_pose=None) -> torch.Tensor:
        """Compute prior energy term."""
        # Implement inline or call your priors function
        total_prior = torch.tensor(0.0, dtype=mu.dtype, device=mu.device)
        # ... add priors
        return total_prior
    
    def _get_process_noise_variances(self) -> list:
        """Return process noise for each state dimension."""
        cfg = self.config
        return [
            cfg.sigma_process_xy**2,      # cx
            cfg.sigma_process_xy**2,      # cy
            cfg.sigma_process_size**2,    # width
            cfg.sigma_process_size**2,    # height
            cfg.sigma_process_size**2,    # z_base
            cfg.sigma_process_angle**2    # theta
        ]
    
    @classmethod
    def from_cluster(cls, belief_id: int, cluster: np.ndarray,
                     config, device=DEVICE) -> Optional['YourObjectBelief']:
        """Create belief from point cluster."""
        
        if len(cluster) < config.min_cluster_points:
            return None
        
        # 1. Compute centroid
        pts_xy = cluster[:, :2]
        centroid = np.mean(pts_xy, axis=0)
        
        # 2. Use PCA to find orientation and dimensions
        centered = pts_xy - centroid
        try:
            cov = np.cov(centered.T)
            if cov.ndim < 2:
                return None
            evals, evecs = np.linalg.eigh(cov)
            idx = np.argsort(evals)[::-1]
            evals, evecs = evals[idx], evecs[:, idx]
            
            # Compute angle from principal axis
            angle = np.arctan2(evecs[1, 0], evecs[0, 0])
        except Exception:
            return None
        
        # 3. Apply object-specific filters
        eigenratio = evals[1] / (evals[0] + 1e-6)
        # e.g., reject if too linear for a square object
        if eigenratio < 0.1:
            return None
        
        # 4. Compute dimensions from rotated points
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rotated = np.column_stack([
            centered[:, 0] * cos_a - centered[:, 1] * sin_a,
            centered[:, 0] * sin_a + centered[:, 1] * cos_a
        ])
        min_xy, max_xy = np.min(rotated, axis=0), np.max(rotated, axis=0)
        width = max(max_xy[0] - min_xy[0], 0.1)
        depth = max(max_xy[1] - min_xy[1], 0.1)
        
        # 5. Validate dimensions
        if width < config.min_width or width > config.max_width:
            return None
        
        # 6. Compute z_base and height from Z values
        z_values = cluster[:, 2]
        z_min, z_max = np.min(z_values), np.max(z_values)
        z_base = np.clip(z_min, config.min_z_base, config.max_z_base)
        height = np.clip(z_max - z_min + 0.05, config.min_height, config.max_height)
        
        # 7. Create state tensor
        mu = torch.tensor([centroid[0], centroid[1], width, height, z_base, angle],
                         dtype=DTYPE, device=device)
        
        # 8. Create covariance matrix (diagonal)
        Sigma = torch.diag(torch.tensor([
            config.initial_position_std**2,   # cx
            config.initial_position_std**2,   # cy
            config.initial_size_std**2,       # width
            config.initial_size_std**2,       # height
            config.initial_size_std**2,       # z_base
            config.initial_angle_std**2       # theta
        ], dtype=DTYPE, device=device))
        
        print(f"[YourObjectBelief] Created id={belief_id}: pos=({centroid[0]:.2f}, {centroid[1]:.2f}), "
              f"size=({width:.2f} x {height:.2f}), z_base={z_base:.2f}")
        
        return cls(belief_id, mu, Sigma, config, config.initial_confidence)
```

---

## Step 4: Register in Model Selector

Edit `src/model_selector.py`:

```python
from src.objects.your_object.belief import YourObjectBelief, YourObjectBeliefConfig

class ModelSelector:
    def __init__(self, model_types: List[str], ...):
        # ...
        
        # Add your object registration
        if 'your_object' in model_types:
            self.model_classes['your_object'] = (YourObjectBelief, YourObjectBeliefConfig())
```

---

## Step 5: Add Size Clamping in Optimizer

Edit `src/multi_model_manager.py` in `_optimize_hypothesis()`:

```python
# Clamp sizes using model-specific limits
config = belief.config

if model_type == 'tv':
    # TV: [cx, cy, width, height, z_base, theta]
    mu[2] = torch.clamp(mu[2], config.min_width, config.max_width)
    mu[3] = torch.clamp(mu[3], config.min_height, config.max_height)
    mu[4] = torch.clamp(mu[4], config.min_z_base, config.max_z_base)
elif model_type == 'your_object':
    # Your object: [cx, cy, ...]
    mu[2] = torch.clamp(mu[2], config.min_width, config.max_width)
    mu[3] = torch.clamp(mu[3], config.min_height, config.max_height)
    mu[4] = torch.clamp(mu[4], config.min_z_base, config.max_z_base)
else:
    # Generic fallback using min_size/max_size
    for i in range(2, len(mu) - 1):
        mu[i] = torch.clamp(mu[i], config.min_size, config.max_size)
```

---

## Step 6: Add Qt3D Visualization

Edit `src/qt3d_viewer.py`:

### 6.1 Create Method

```python
def _create_your_object(self, obj_id: int, cx: float, cy: float, 
                        width: float, height: float, z_base: float, theta: float):
    """Create your object entity in Qt3D."""
    print(f"[Qt3D] CREATING YOUR_OBJECT {obj_id} at ({cx:.2f}, {cy:.2f})")
    
    # Create parent entity
    entity = Qt3DCore.QEntity(self.root_entity)
    
    # Create transform
    transform = Qt3DCore.QTransform()
    transform.setTranslation(QVector3D(cx, cy, 0))
    transform.setRotationZ(math.degrees(theta))
    entity.addComponent(transform)
    
    # Create material
    material = Qt3DExtras.QPhongMaterial()
    material.setDiffuse(QColor(100, 100, 100))
    material.setAmbient(QColor(50, 50, 50))
    
    # Create mesh (e.g., cuboid)
    mesh = Qt3DExtras.QCuboidMesh()
    mesh.setXExtent(width)
    mesh.setYExtent(0.05)  # thin
    mesh.setZExtent(height)
    
    # Create sub-entity for the mesh
    mesh_entity = Qt3DCore.QEntity(entity)
    mesh_transform = Qt3DCore.QTransform()
    mesh_transform.setTranslation(QVector3D(0, 0, z_base + height / 2))
    mesh_entity.addComponent(mesh)
    mesh_entity.addComponent(mesh_transform)
    mesh_entity.addComponent(material)
    
    # Store references
    self.scene_objects[f'your_object_{obj_id}'] = {
        'entity': entity,
        'transform': transform,
        'mesh': mesh,
        'mesh_transform': mesh_transform,
        'material': material,
    }
    
    print(f"[Qt3D] YOUR_OBJECT {obj_id} created successfully")
    return entity
```

### 6.2 Update Method

```python
def _update_your_object(self, obj_id: int, cx: float, cy: float,
                        width: float, height: float, z_base: float, theta: float):
    """Update existing object."""
    key = f'your_object_{obj_id}'
    
    if key not in self.scene_objects:
        self._create_your_object(obj_id, cx, cy, width, height, z_base, theta)
        return
    
    obj = self.scene_objects[key]
    obj['transform'].setTranslation(QVector3D(cx, cy, 0))
    obj['transform'].setRotationZ(math.degrees(theta))
    obj['mesh'].setXExtent(width)
    obj['mesh'].setZExtent(height)
    obj['mesh_transform'].setTranslation(QVector3D(0, 0, z_base + height / 2))
```

### 6.3 Add Update Loop Method

```python
def update_your_objects(self, beliefs: list):
    """Update all your_object visualizations."""
    active_ids = set()
    
    for belief_dict in beliefs:
        model_sel = belief_dict.get('model_selection', {})
        state = model_sel.get('state', 'committed')
        if state != 'committed':
            continue
        
        if belief_dict.get('type') != 'your_object':
            continue
        
        obj_id = belief_dict.get('id', 0)
        active_ids.add(obj_id)
        
        cx = belief_dict.get('cx', 0)
        cy = belief_dict.get('cy', 0)
        width = belief_dict.get('width', 1.0)
        height = belief_dict.get('height', 0.5)
        z_base = belief_dict.get('z_base', 1.0)
        theta = belief_dict.get('angle', 0)
        
        self._update_your_object(obj_id, cx, cy, width, height, z_base, theta)
    
    self._remove_unused_your_objects(active_ids)

def _remove_unused_your_objects(self, active_ids: set):
    """Remove objects that are no longer active."""
    to_remove = [k for k in self.scene_objects.keys() 
                 if k.startswith('your_object_') and int(k.split('_')[-1]) not in active_ids]
    for key in to_remove:
        entity = self.scene_objects[key]['entity']
        entity.setParent(None)
        del self.scene_objects[key]
```

### 6.4 Call from Main Update

In the `update()` method, add:

```python
def update(self, beliefs, robot_pose, room_dims, ...):
    # ... existing updates ...
    self.update_your_objects(beliefs)
```

---

## Step 7: Add Ground Truth Config (Optional)

Edit `src/specificworker.py`:

```python
GT_CONFIG = {
    'table': {'gt_cx': 0.0, 'gt_cy': 0.0, 'gt_width': 0.8, ...},
    'chair': {'gt_cx': 0.0, 'gt_cy': 0.0, 'gt_seat_width': 0.45, ...},
    'tv': {'gt_cx': 0.0, 'gt_cy': 0.0, 'gt_width': 1.0, 'gt_height': 0.5},
    'your_object': {
        'gt_cx': 0.0,
        'gt_cy': 0.0,
        'gt_width': 1.0,
        'gt_height': 0.5,
        'gt_z_base': 1.0,
    },
}
```

---

## Step 8: DSR Integration (Optional)

Edit `src/dsr_integration.py` to store objects in the DSR graph:

```python
def _create_dsr_node(self, belief, obj_type):
    if obj_type == 'your_object':
        # Use 'container' as the DSR node type
        node = Node(self.g.get_agent_id(), 'container', f"your_object_{belief.id}")
        # Set attributes (in mm for positions)
        node.attrs['pos_x'] = Attribute(float(belief.cx * 1000), self.agent_id)
        node.attrs['pos_y'] = Attribute(float(belief.cy * 1000), self.agent_id)
        node.attrs['width'] = Attribute(float(belief.width * 1000), self.agent_id)
        # ... etc
        
        # Create RT edge to room
        # ...
```

---

## Checklist

- [ ] Create `src/objects/your_object/` directory
- [ ] Create `__init__.py` with exports
- [ ] Implement `sdf.py` with SDF and priors
- [ ] Implement `belief.py` with config and belief class
- [ ] Register in `src/model_selector.py`
- [ ] Add clamping in `src/multi_model_manager.py`
- [ ] Add Qt3D visualization in `src/qt3d_viewer.py`
- [ ] Update `src/specificworker.py` to include new model
- [ ] Test with single object scenario
- [ ] Test with multi-object scenario
- [ ] Add GT config for debugging (optional)
- [ ] Add DSR integration (optional)

---

## Common Issues

### 1. State dimension mismatch
Ensure `STATE_DIM`, covariance matrix size, and parameter count all match.

### 2. Index errors
When accessing `mu[i]`, ensure indices match your state layout. Remember:
- Position: `mu[0]`, `mu[1]`
- Dimensions: `mu[2]`, `mu[3]`, ...
- Angle: `mu[-1]` (always last)

### 3. SDF sign convention
- **Positive** = outside the object
- **Negative** = inside the object
- **Zero** = on the surface

### 4. Angle normalization
Always normalize angles to [-π, π] after updates:
```python
theta = torch.atan2(torch.sin(theta), torch.cos(theta))
```

### 5. Fixed vs free parameters
If a parameter is fixed (like TV depth), **don't include it in the state vector**. Instead:
- Provide it as a constant in `sdf.py`
- Return it as a property that reads the constant

### 6. `from_cluster` returning None
Check:
- `min_cluster_points` threshold
- Eigenratio filter
- Dimension validation

### 7. Qt3D object not visible
Common causes:
- Wrong Z position (object below floor)
- Zero dimensions
- Material not added to entity
- Entity not parented to root_entity

---

## Example State Vectors

| Object | State Vector | Params |
|--------|--------------|--------|
| **Box** | `[cx, cy, width, height, depth, theta]` | 6 |
| **Table** | `[cx, cy, width, depth, table_height, leg_length, theta]` | 7 |
| **Chair** | `[cx, cy, seat_width, seat_depth, seat_height, back_height, theta]` | 7 |
| **TV** | `[cx, cy, width, height, z_base, theta]` | 6 (depth fixed) |

---

## References

- `src/objects/tv/` - Example of object with fixed parameter (depth)
- `src/objects/table/` - Example of compound object (top + legs)
- `src/objects/chair/` - Example of complex multi-part object
- `src/belief_core.py` - Base Belief class
- `src/multi_model_manager.py` - Bayesian Model Selection
