# CLAUDE.md - Template for concept-box agent

Copy this file to the concept-box agent directory and rename to CLAUDE.md.

## Component Overview

This is `concept-box`, a RoboComp component that detects boxes on the floor using Active Inference with SDF minimization. It reads room geometry and robot pose from the shared DSR graph (populated by `ainf_test`) and estimates box positions and dimensions.

**Related agent**: See `ainf_test/ACTIVE_INFERENCE_MATH.md` for the mathematical framework.

## Architecture

```
┌─────────────────┐         ┌─────────────────┐
│   ainf_test     │         │  concept-box    │
│                 │         │                 │
│ Room + Pose     │◄───────►│ Box Detection   │
│ Estimation      │   DSR   │ using SDF       │
│                 │  Graph  │                 │
└─────────────────┘         └─────────────────┘
        │                           │
        ▼                           ▼
   Room node                  Box nodes
   RT edge (pose+cov)         (detected boxes)
```

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

## Suggested Implementation Steps

1. **Subscribe to DSR signals** to receive room/pose updates
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

## Key Dependencies

Same as ainf_test:
- PyTorch for optimization
- pydsr for DSR graph access
- numpy for matrix operations

## References

- `ainf_test/ACTIVE_INFERENCE_MATH.md`: Full mathematical formulation
- `ainf_test/src/concept_room.py`: Reference implementation of SDF optimization
