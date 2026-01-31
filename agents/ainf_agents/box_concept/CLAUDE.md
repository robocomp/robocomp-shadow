# CLAUDE.md - Box Concept Agent

## Component Overview

This is `box_concept`, a RoboComp component that detects rectangular obstacles using **Active Inference** with SDF-based Variational Free Energy minimization. It reads room geometry and robot pose from the shared DSR graph and maintains Gaussian beliefs over box parameters.

**Related documents**: 
- `main.tex`: Full mathematical framework (Active Inference paper)
- `concept_rectangle.py`: Reference implementation for rectangle obstacles

## Architecture

```
┌─────────────────┐         ┌─────────────────┐
│   ainf_test     │         │  box_concept    │
│                 │         │                 │
│ Room + Pose     │◄───────►│ Box Detection   │
│ Estimation      │   DSR   │ using Active    │
│                 │  Graph  │ Inference       │
└─────────────────┘         └─────────────────┘
        │                           │
        ▼                           ▼
   Room node                  Box nodes
   RT edge (pose+cov)         (Gaussian beliefs)
```

## Active Inference Framework

### Variational Free Energy Objective

From paper Eq. obstacle_objective:
```
F(s) = (1/2σ²_o) Σᵢ dᵢ(s)² + (1/2)(s-μₚ)ᵀΣₚ⁻¹(s-μₚ)
```

Where:
- **Prediction Error** (negative accuracy): `(1/2σ²_o) Σᵢ dᵢ(s)²`
  - `dᵢ(s)` = SDF of point `pᵢ` to rectangle with state `s`
  - `σ_o` = observation noise standard deviation
  
- **Complexity** (prior regulariser): `(1/2)(s-μₚ)ᵀΣₚ⁻¹(s-μₚ)`
  - `μₚ` = prior mean from temporal propagation
  - `Σₚ` = prior covariance

### Belief Representation

Each obstacle is represented as a Gaussian belief:
```
q(s) = N(s | μ, Σ)
```

State vector `s = (cx, cy, w, h, d, θ)` - **6D**:
- `(cx, cy)`: center position in room frame (meters)
- `(w, h, d)`: width, height, depth (meters) - **3D dimensions**
- `θ`: orientation angle (radians)

### Size Prior

Dimensions are regularized toward typical box size:
```
F_size = (1/2σ²_size) [(w - 0.5)² + (h - 0.5)² + (d - 0.5)²]
```
- Prior mean μ_size = 0.5m (typical box)
- Prior std σ_size = 0.3m

### 3D SDF (Signed Distance Function)

The SDF supports both 2D and 3D points:
- **2D points** [N, 2]: Uses (w, h) for top-down projection
- **3D points** [N, 3]: Full 3D SDF with (w, h, d)

```python
# 3D SDF formula
dx = |local_x| - w/2
dy = |local_y| - h/2  
dz = |local_z| - d/2

# Outside: Euclidean distance to surface
outside = sqrt(max(dx,0)² + max(dy,0)² + max(dz,0)²)

# Inside: negative distance to nearest face
inside = max(dx, dy, dz)
```

### Perception Cycle (from paper)

1. **Transform** LIDAR points to room frame (Eq. lidar_room_transform)
2. **Cluster** points via DBSCAN-like algorithm (Sec. obstacle-dbscan)
3. **Filter** wall points using PCA linearity (Sec. obstacle-pca)
4. **Associate** clusters to beliefs via Hungarian algorithm (Sec. obstacle-association)
5. **Update** matched beliefs via VFE gradient descent (Sec. obstacle-update)
6. **Initialize** new beliefs for unmatched clusters (Sec. circle_fit adapted)
7. **Decay** unmatched beliefs (Sec. belief_decay)

### Belief Lifecycle

From paper:
- **Provisional**: New beliefs start with confidence κ = 0.4
- **Confidence boost**: κ += Δκ on each match
- **Confidence decay**: κ *= γ for unmatched beliefs  
- **Confirmed**: κ ≥ 0.7 → belief becomes permanent
- **Removal**: κ < 0.2 → belief is pruned (Occam's razor)

## Implementation Status

### Implemented Features
- [x] Gaussian beliefs with mean μ (6D) and covariance Σ (6x6)
- [x] 3D SDF-based likelihood model for oriented boxes
- [x] Size prior regularization toward 0.5m typical box size
- [x] VFE minimization via PyTorch gradient descent
- [x] Prior propagation with process noise
- [x] Hungarian algorithm for data association
- [x] Belief confidence tracking and decay
- [x] DSR graph integration

### Main Cycle (specificworker.py::compute())
1. Get LIDAR points (2D projection, meters)
2. Check room node exists, get dimensions
3. Get robot pose and covariance from Shadow node
4. Call `BoxManager.update()` with all data
5. BoxManager runs full Active Inference perception cycle

## DSR Graph Data (from ainf_test)

### Reading Room Geometry
```python
room_node = self.g.get_node("room")
if room_node:
    width_mm = room_node.attrs["room_width"].value   # Width in mm
    length_mm = room_node.attrs["room_length"].value # Length in mm
    width_m = width_mm / 1000.0
    length_m = length_mm / 1000.0
```

### Reading Robot Pose and Covariance
```python
robot_node = self.g.get_node("robot")
room_node = self.g.get_node("room")
if robot_node and room_node:
    rt_edge = self.g.get_edge(robot_node.id, room_node.id, "RT")
    if rt_edge:
        translation = rt_edge.attrs["rt_translation"].value  # [x, y, z] in mm
        rotation = rt_edge.attrs["rt_rotation_euler_xyz"].value  # [rx, ry, rz] in rad
        cov_flat = rt_edge.attrs["rt_se2_covariance"].value  # 9 elements (3x3 flattened)
        
        # Convert to meters and reshape
        x_m = translation[0] / 1000.0
        y_m = translation[1] / 1000.0
        theta = rotation[2]  # rz is the heading angle
        cov_matrix = np.array(cov_flat).reshape(3, 3)
```

## Coordinate Convention

Same as ainf_test:
- **Room frame**: Origin at room center
  - X-axis: Right (lateral), walls at x = ±W/2
  - Y-axis: Forward, walls at y = ±L/2
- **Robot frame**: 
  - X-axis: Right
  - Y-axis: Forward (direction robot is facing)
- **θ = 0**: Robot facing +Y direction

## SDF for Box Detection

For a box centered at $(b_x, b_y)$ with dimensions $(w, h)$:

```python
def sdf_box(points, box_x, box_y, box_w, box_h):
    """
    SDF for a rectangular box obstacle.
    Returns distance to box boundary (negative inside, positive outside).
    """
    # Translate points relative to box center
    px = points[:, 0] - box_x
    py = points[:, 1] - box_y
    
    # Distance to box edges
    dx = torch.abs(px) - box_w / 2
    dy = torch.abs(py) - box_h / 2
    
    # SDF: negative inside, positive outside
    outside = torch.sqrt(torch.relu(dx)**2 + torch.relu(dy)**2)
    inside = torch.min(torch.max(dx, dy), torch.zeros_like(dx))
    
    return outside + inside
```

## Active Inference Framework

Same principles as ainf_test:

1. **Free Energy**: F = F_likelihood + π × F_prior
2. **Likelihood**: SDF error for LIDAR points near boxes
3. **Prior**: Expected box locations (can be uniform or learned)
4. **Precision π**: Confidence in prior vs observations

## Box State Vector

For each detected box:
$$\mathbf{b} = (b_x, b_y, w, h)$$

Where:
- $(b_x, b_y)$: Box center in room frame (meters)
- $(w, h)$: Box width and height (meters)

## Implementation Steps

1. **Subscribe to DSR signals** to receive room/pose updates ✓ (already connected)
2. **Filter LIDAR points** that don't belong to walls (residuals from room SDF)
3. **Cluster non-wall points** to identify potential boxes
4. **Optimize box parameters** using SDF minimization
5. **Insert box nodes** into DSR graph

## DSR Box Node Creation

```python
from pydsr import Node, Edge, Attribute

def insert_box_node(self, box_id, box_x, box_y, box_w, box_h):
    box_node = Node(agent_id=self.agent_id, type="box", name=f"box_{box_id}")
    box_node.attrs["pos_x"] = Attribute(float(box_x * 1000), self.agent_id)  # mm
    box_node.attrs["pos_y"] = Attribute(float(box_y * 1000), self.agent_id)  # mm
    box_node.attrs["box_width"] = Attribute(float(box_w * 1000), self.agent_id)  # mm
    box_node.attrs["box_height"] = Attribute(float(box_h * 1000), self.agent_id)  # mm
    box_node.attrs["color"] = Attribute("Orange", self.agent_id)
    
    new_id = self.g.insert_node(box_node)
    return new_id
```

## Available Proxies

From `specificworker.py`:
```python
# Lidar3D proxy methods:
self.lidar3d_proxy.getColorCloudData()
self.lidar3d_proxy.getLidarData(name, start, len, decimationDegreeFactor)
self.lidar3d_proxy.getLidarDataArrayProyectedInImage(name)
self.lidar3d_proxy.getLidarDataByCategory(categories, timestamp)
self.lidar3d_proxy.getLidarDataProyectedInImage(name)
self.lidar3d_proxy.getLidarDataWithThreshold2d(name, distance, decimationDegreeFactor)
```

## Key Dependencies

- PyTorch for optimization
- pydsr for DSR graph access
- numpy for matrix operations
- PySide6 for Qt integration

## Files Structure

```
box_concept/
├── CLAUDE.md                  # This file
├── CONCEPT_BOX_TEMPLATE.md    # Original template
├── src/
│   ├── specificworker.py      # Main component logic
│   └── box_manager.py         # BoxManager class for box lifecycle
├── generated/
│   ├── box_concept.py         # Entry point
│   ├── genericworker.py       # Base worker class
│   └── interfaces.py          # Ice interfaces
└── etc/
    └── config                 # Configuration file
```

## References

- `ainf_test/ACTIVE_INFERENCE_MATH.md`: Full mathematical formulation
- `ainf_test/src/concept_room.py`: Reference implementation of SDF optimization
- CORTEX Python API: https://github.com/robocomp/cortex/blob/dev/python-signals/python_api_documentation.md
