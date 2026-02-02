# Adding New Object Types

This guide explains how to add a new object type (e.g., table, chair) to the Active Inference belief system.

## Project Structure

```
src/
├── belief_core.py       # Abstract base class for beliefs
├── belief_manager.py    # Generic belief lifecycle management
├── box_belief.py        # Box-specific belief
├── box_manager.py       # Box-specific manager
├── table_belief.py      # Table-specific belief (example of new object)
├── table_manager.py     # Table-specific manager
├── object_sdf_prior.py  # ALL SDF and prior functions for all objects
├── transforms.py        # Coordinate transformations
├── visualizer_3d.py     # Open3D visualization (supports box + table)
└── specificworker.py    # Main entry point
```

---

## Step-by-Step: Adding a Table Object

### Step 1: Define the State Vector

Decide on the parameters for your object. For a table:

```
Table State: [cx, cy, w, h, table_height, leg_length, theta]
- cx, cy: center position in XY plane
- w, h: width and depth of table top
- table_height: height of table surface from floor
- leg_length: length of legs (free parameter)
- theta: rotation angle

Fixed constants (in object_sdf_prior.py):
- TABLE_TOP_THICKNESS = 0.03m
- TABLE_LEG_RADIUS = 0.025m
```

### Step 2: Add SDF Function to `object_sdf_prior.py`

The SDF function computes the signed distance from points to the object surface.

```python
def compute_table_sdf(points_xyz: torch.Tensor, table_params: torch.Tensor) -> torch.Tensor:
    """
    Compute Signed Distance Function for a table.
    
    Args:
        points_xyz: [N, 3] points (x, y, z)
        table_params: [7] tensor [cx, cy, w, h, table_height, leg_length, theta]
    
    Returns:
        [N] SDF values (positive=outside, negative=inside)
    """
    # Extract parameters
    cx, cy = table_params[0], table_params[1]
    w, h = table_params[2], table_params[3]
    table_height = table_params[4]
    leg_length = table_params[5]
    theta = table_params[6]
    
    # Use module constants
    TOP_THICKNESS = TABLE_TOP_THICKNESS
    LEG_RADIUS = TABLE_LEG_RADIUS
    
    # Transform points to local frame
    cos_t = torch.cos(-theta)
    sin_t = torch.sin(-theta)
    px = points_xyz[:, 0] - cx
    py = points_xyz[:, 1] - cy
    local_x = px * cos_t - py * sin_t
    local_y = px * sin_t + py * cos_t
    local_z = points_xyz[:, 2]
    
    # Compute SDF for table top (box)
    # ... (see existing implementation)
    
    # Compute SDF for legs (cylinders)
    # ... (see existing implementation)
    
    # Return union (minimum of all parts)
    return torch.minimum(sdf_top, sdf_legs)
```

### Step 3: Add Prior Function to `object_sdf_prior.py`

The prior function computes energy terms for structural priors (e.g., angle alignment).

```python
def compute_table_priors(params: torch.Tensor, config, robot_pose: np.ndarray = None) -> torch.Tensor:
    """
    Compute prior energy for table parameters.
    
    Args:
        params: [7] tensor [cx, cy, w, h, table_height, leg_length, theta]
        config: Configuration with prior parameters
        robot_pose: Robot pose for angle transformation
    
    Returns:
        Total prior energy (scalar)
    """
    theta = params[6]
    
    # Transform to room frame if robot_pose provided
    if robot_pose is not None:
        theta_room = theta + robot_pose[2]
    else:
        theta_room = theta
    
    # Normalize angle
    while theta_room > np.pi/2: theta_room = theta_room - np.pi
    while theta_room < -np.pi/2: theta_room = theta_room + np.pi
    
    # Angle alignment prior (tables align with room axes)
    sigma = getattr(config, 'angle_alignment_sigma', 0.1)
    weight = getattr(config, 'angle_alignment_weight', 0.5)
    precision = 1.0 / (sigma ** 2)
    
    dist_to_0 = theta_room ** 2
    dist_to_pos90 = (theta_room - np.pi/2) ** 2
    dist_to_neg90 = (theta_room + np.pi/2) ** 2
    min_dist_sq = torch.minimum(dist_to_0, torch.minimum(dist_to_pos90, dist_to_neg90))
    
    return weight * 0.5 * precision * min_dist_sq
```

### Step 4: Register in OBJECT_REGISTRY

Add your object to the registry at the bottom of `object_sdf_prior.py`:

```python
OBJECT_REGISTRY: Dict[str, Dict[str, Any]] = {
    'box': { ... },
    'cylinder': { ... },
    'table': {
        'sdf': compute_table_sdf,
        'prior': compute_table_priors,
        'param_count': 7,
        'param_names': ['cx', 'cy', 'w', 'h', 'table_height', 'leg_length', 'theta'],
        'description': 'Table with box top and 4 cylindrical legs',
    },
    'chair': { ... },
}
```

### Step 5: Create Belief Class (e.g., `table_belief.py`)

Create a new file `table_belief.py` based on `box_belief.py`:

```python
from dataclasses import dataclass
from src.belief_core import Belief, BeliefConfig, DEVICE, DTYPE
from src.object_sdf_prior import compute_table_sdf

@dataclass
class TableBeliefConfig(BeliefConfig):
    """Configuration for table beliefs."""
    # Size priors
    prior_table_height: float = 0.75  # Standard table height
    prior_table_height_std: float = 0.1
    
    # ... other table-specific config
    
    # Angle alignment prior
    angle_alignment_weight: float = 0.5
    angle_alignment_sigma: float = 0.1


class TableBelief(Belief):
    """Gaussian belief over table parameters."""
    STATE_DIM = 7
    
    @property
    def state_dim(self) -> int:
        return self.STATE_DIM
    
    @property
    def position(self) -> Tuple[float, float]:
        return self.mu[0].item(), self.mu[1].item()
    
    @property
    def angle(self) -> float:
        return self.mu[6].item()  # theta is at index 6 for table
    
    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        return compute_table_sdf(points, self.mu)
    
    @classmethod
    def from_cluster(cls, belief_id, cluster, config, device):
        # Initialize table parameters from cluster
        # Use PCA for position/orientation
        # Initialize height, leg_length with priors
        ...
```

### Step 6: Create Manager (e.g., `table_manager.py`)

Create a thin wrapper like `box_manager.py`:

```python
from src.belief_manager import BeliefManager
from src.table_belief import TableBelief, TableBeliefConfig

class TableManager(BeliefManager):
    """Specialized BeliefManager for tables."""
    
    def __init__(self, g, agent_id: int, config: TableBeliefConfig = None):
        config = config or TableBeliefConfig()
        super().__init__(TableBelief, config, DEVICE)
        self.g = g
        self.agent_id = agent_id
```

### Step 7: Update Visualizer

Add table rendering to `visualizer_3d.py`:

```python
def _create_table_geometry(self, belief_dict):
    """Create Open3D geometry for a table belief."""
    # Create table top (box)
    # Create 4 legs (cylinders)
    # Combine and return
```

---

## Configuration Parameters

All objects share these base parameters from `BeliefConfig`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `min_size` | Minimum dimension | 0.10m |
| `max_size` | Maximum dimension | 2.0m |
| `confidence_decay` | Decay when not observed | 0.85 |
| `confidence_boost` | Increase on observation | 0.15 |
| `confidence_threshold` | Remove below this | 0.25 |

Object-specific configs extend `BeliefConfig` with additional parameters.

---

## Optimization Parameters (in `BeliefManager.__init__`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lambda_prior` | Weight for state prior | 0.3 |
| `optimization_iters` | Gradient descent iterations | 10 |
| `optimization_lr` | Learning rate | 0.05 |
| `grad_clip` | Gradient clipping | 2.0 |
| `merge_belief_dist` | Merge distance threshold | 0.3m |

---

## SDF Constants (in `object_sdf_prior.py`)

| Constant | Description | Default |
|----------|-------------|---------|
| `SDF_SMOOTH_K` | Smooth min parameter | 0.02m |
| `SDF_INSIDE_SCALE` | Inside point weight | 0.5 |
| `TABLE_TOP_THICKNESS` | Table top thickness | 0.03m |
| `TABLE_LEG_RADIUS` | Table leg radius | 0.025m |
| `CHAIR_SEAT_THICKNESS` | Chair seat thickness | 0.05m |

---

## Checklist for New Object

- [ ] Define state vector and parameter meanings
- [ ] Implement `compute_<object>_sdf()` in `object_sdf_prior.py`
- [ ] Implement `compute_<object>_priors()` in `object_sdf_prior.py`
- [ ] Add to `OBJECT_REGISTRY`
- [ ] Create `<object>_belief.py` with config and belief class
- [ ] Create `<object>_manager.py` wrapper
- [ ] Add visualization support in `visualizer_3d.py`
- [ ] Test with synthetic data
- [ ] Test with real LIDAR data
