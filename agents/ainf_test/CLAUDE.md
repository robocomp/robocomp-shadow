# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Component Overview

This is `ainf_test`, a RoboComp component that validates Active Inference for the room-robot concept problem. It estimates room geometry and robot pose using LIDAR data and a Signed Distance Function (SDF) minimization approach implemented in PyTorch.

**See `ACTIVE_INFERENCE_MATH.md` for the complete mathematical formulation.**

## Building and Running

### Build the Component

```bash
cmake .
make
```

This generates Ice interfaces in `generated/` and creates a symlink `bin/ainf_test` → `generated/ainf_test.py`.

### Running the Component

The recommended way to run the full system is using the subcognitive launcher:

```bash
python subcognitive.py sub.toml
```

This script:
- Checks if Webots simulator is running (starts it if not)
- Launches all components defined in `sub.toml` (bridge, lidar, joystick, etc.)
- Monitors component health with Ice pings and resource usage
- Displays a live status table

Required Webots world: `SimpleWorld.wbt`

### Running Component Standalone

```bash
bin/ainf_test etc/config
```

Note: Requires all dependencies (bridge, lidar, etc.) to be running separately.

## Architecture

### Component Structure (RoboComp Pattern)

RoboComp components follow a specific architecture:

1. **CDSL Definition** (`ainf_test.cdsl`): Declares interfaces the component requires/provides
   - `requires`: Lidar3D, OmniRobot (synchronous proxies)
   - `subscribesTo`: FullPoseEstimationPub (asynchronous topic)
   - `options dsr`: Enables DSR (Deep State Representation) graph integration

2. **Generated Code** (`generated/`):
   - `genericworker.py`: Auto-generated base class with Ice proxies and DSR graph
   - `interfaces.py`: Ice interface stubs
   - `ainf_test.py`: Main entry point that instantiates SpecificWorker

3. **User Code** (`src/`):
   - `specificworker.py`: Main component logic, inherits from GenericWorker
   - Core modules (described below)

### Core Modules

#### `src/specificworker.py`
Main component loop and coordination:
- **Compute Loop** (100ms period): Gets LIDAR, executes exploration, updates estimator
- **Exploration**: Trajectory-based square pattern around room center
- **Ground Truth Handling**: Receives poses via `FullPoseEstimationPub_newFullPose()` subscription
- **Coordinate Transform**: Negates simulator coordinates to match internal convention
- **Statistics**: Tracks pose/SDF errors, prints summary on completion
- **DSR Integration**: Connects to DSR graph signals for node/edge updates

#### `src/concept_room.py`
Active Inference room/pose estimator using PyTorch:
- **RoomBelief**: Gaussian belief over state (x, y, θ, W, L) with 5×5 covariance
- **RoomPoseEstimatorV2**: Two-phase estimator
  - **INIT Phase**: Robot static, collects LIDAR (500+ points), minimizes SDF over full state using gradient descent
  - **TRACKING Phase**: Robot moving, fixed room dimensions (W, L), propagates pose with velocity (dead reckoning), corrects with LIDAR via SDF minimization
- **SDF Function**: `sdf_rect()` computes signed distance to rectangular room walls
- **Optimization**: PyTorch autograd for gradients, custom learning rates per phase
- Uses GPU if available (falls back to CPU)

#### `src/room_viewer.py`
Real-time visualization using DearPyGui:
- **Observer Pattern**: `RoomSubject` notifies `RoomViewerDPG` observers
- **ViewerData**: Dataclass packaging estimated pose, ground truth, LIDAR, room dimensions
- **Dual Display**: Shows estimated robot (blue) vs ground truth (red)
- **Live LIDAR**: Renders current LIDAR points
- **Stats Panel**: Displays phase, errors, uncertainties, step count
- **DSR Panel**: Integrated DSR graph viewer (right side)

#### `src/dsr_graph_viewer.py`
DSR graph visualization:
- Renders nodes (root, room, robot, shadow) with type-based colors
- Shows RT (RigidTransform) edges
- Force-directed layout for node positioning
- Integrated into main viewer canvas

### Coordinate Systems

**Internal Convention** (estimator, viewer, LIDAR):
- x+ = right (lateral)
- y+ = forward (front)
- θ = heading angle (radians), where **θ=0 means facing +y (forward)**

**Rotation Matrix** (robot frame → room frame):
For a point (px, py) in robot frame with robot at position (x, y) and heading θ:
```
room_x = cos(θ) * px - sin(θ) * py + x
room_y = sin(θ) * px + cos(θ) * py + y
```
This is the standard 2D rotation matrix.

**Motion Model** (velocity transform):
For robot velocity (vx_robot, vy_robot) in robot frame:
```
vx_world = cos(θ) * vx_robot - sin(θ) * vy_robot
vy_world = sin(θ) * vx_robot + cos(θ) * vy_robot
```

**Simulator Convention** (Webots via FullPoseEstimationPub):
- Negated from internal
- Transform in `FullPoseEstimationPub_newFullPose()`: negate x, y, and θ

**LIDAR Data**:
- Points in mm, converted to meters
- In robot frame: [x, y] = [right, forward]
- Retrieved via `getLidarDataWithThreshold2d("helios", 8000, 4)`

### DSR (Deep State Representation) Graph

- Initialized via `genericworker.py` using config from `insight_dsr.json`
- Provides shared knowledge graph across components
- Nodes: world, robot, room, shadow, etc.
- Edges: RT (rigid transforms), semantic relations
- Signal handlers in `specificworker.py`: `update_node()`, `update_edge()`, etc.

### Trajectory System

Exploration uses declarative trajectory definitions in `_init_trajectory()`:
- Actions: `('turn', degrees)`, `('forward', meters)`, `('backward', meters)`, `('wait', seconds)`
- Default: 3 loops of 1m square pattern
- Executor calculates steps from velocities and period

## Configuration Files

- **`etc/config`**: Ice endpoints, proxies, agent ID/name, DSR config path, compute period
- **`sub.toml`**: Component launcher configuration (paths, commands, Ice names)
- **`insight_dsr.json`**: DSR graph structure and node definitions

## Key Dependencies

- **RoboComp**: `/opt/robocomp/lib` (pydsr)
- **PyTorch**: For Active Inference optimization (GPU optional)
- **DearPyGui**: Real-time visualization
- **Ice/ZeroC**: Middleware for component communication
- **PySide6**: Qt integration for component lifecycle
- **Rich**: Terminal UI for subcognitive monitor
- **psutil**: Process monitoring

## Development Notes

### Active Inference Parameters

Located in `RoomPoseEstimatorV2.__init__()`:
- `sigma_sdf`: 0.15m (15cm) - LIDAR measurement uncertainty during motion
- `Q_pose`: Process noise [0.01, 0.01, 0.05] for [x, y, θ]
- `lr_init`: 0.5 - learning rate for initialization phase
- `lr_tracking`: 0.1 - learning rate for tracking phase
- `min_points_for_init`: 500 - required LIDAR points before initialization
- `init_convergence_threshold`: 0.05m - SDF error threshold to exit INIT

### EKF-style Tracking Phase

The tracking phase uses a proper predict-update cycle:

**Prediction Step (Motion Model):**
```python
s_pred = s_prev + [vx_world, vy_world, omega] * dt
Sigma_pred = F @ Sigma_prev @ F.T + Q
```
Where F is the Jacobian of the motion model.

**Correction Step (Free Energy Minimization):**
```python
F = F_likelihood + prior_weight * F_prior
F_likelihood = mean(SDF²)  # LIDAR fit
F_prior = 0.5 * (s - s_pred)ᵀ Σ⁻¹ (s - s_pred)  # Motion model prior
```

**Adaptive Prior Precision:**
The prior_precision adapts based on the innovation (prediction error):
- `innovation = optimized_pose - predicted_pose`
- `innovation_mahal = sqrt(innovation @ Σ⁻¹ @ innovation)` (Mahalanobis distance)
- `precision = base + (max - base) * exp(-innovation_mahal / scale)`
- Parameters: `base_precision=0.05`, `max_precision=0.3`, `scale=0.5`

In Active Inference, **precision** represents confidence (inverse variance).
High precision → trust motion model (proprioceptive). Low precision → trust LIDAR (exteroceptive).

**Posterior Covariance (Laplace Approximation):**
```python
H = Hessian of mean(SDF²) at optimum
Sigma_post = σ² * H⁻¹
```
Where σ² is the residual variance `mean(SDF²)`. This scales the geometric uncertainty by the actual fit quality.

### Viewer UI Elements

The room viewer (`room_viewer.py`) displays:

**ROOM Section:**
- `Est: W x L m` - Estimated room dimensions
- `GT: W x L m` - Ground truth room dimensions

**MOTION MODEL Section:**
- `dx: X cm` - Innovation in x (prediction error vs optimized)
- `dy: X cm` - Innovation in y
- `dθ: X°` - Innovation in theta
- `π: 0.XXX` - Adaptive prior precision (color coded)

**ERRORS (vs GT) Section:**
- `X err: X cm` - Position error in x vs ground truth
- `Y err: X cm` - Position error in y vs ground truth
- `θ err: X°` - Angle error vs ground truth
- `Pose: X cm` - Total pose error
- `SDF: X m` - Mean SDF error (LIDAR fit quality)

### Typical Performance

With circular arc trajectory (~2 full circles):
- Room error: ~3cm
- Pose error mean: ~8-10cm
- Angle error: ~1-2°
- SDF error: ~0.06-0.08m

### Modifying Exploration Behavior

Edit `_init_trajectory()` in `specificworker.py`:
- Change square side length
- Add/remove loops
- Adjust speeds: `self.advance_speed` (mm/s), `self.rotation_speed` (rad/s)

### Ice Interface Changes

After modifying `.idsl` files:
1. Edit `ainf_test.cdsl` imports if needed
2. Update `CMakeLists.txt` ROBOCOMP_IDSL_TO_ICE line
3. Run `cmake .` to regenerate Ice files
4. Run `make`

### Adding DSR Nodes/Edges

DSR operations in `specificworker.py`:
- Read: `self.g.get_node(name)`, `self.g.get_edge(from, to, type)`
- Create: `self.g.insert_node(node)`, `self.g.insert_or_assign_edge(edge)`
- Update: Modify node/edge attributes, call `g.update_node()`
- Signals automatically trigger `update_node_att()`, `update_edge()`, etc.

### Debugging

Component logs (when using subcognitive.py):
- stdout: `~/.local/logs/ainf_test.out`
- stderr: `~/.local/logs/ainf_test.err`

Statistics printed every 20 steps to console.
Final summary printed when trajectory completes.
