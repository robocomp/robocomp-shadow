#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Object SDF and Prior Functions

This module centralizes all Signed Distance Functions (SDFs) and Prior functions
for different object types. When adding a new object type, add:

1. SDF function: compute_<object>_sdf(points, params) -> sdf_values
2. Prior function: compute_<object>_priors(params, config) -> prior_energy
3. Register in OBJECT_REGISTRY

=============================================================================
ADDING A NEW OBJECT TYPE - CHECKLIST:
=============================================================================

1. Define the SDF function:
   def compute_myobject_sdf(points_xyz: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
       '''Compute SDF for myobject. Returns [N] distances.'''
       ...

2. Define the prior function:
   def compute_myobject_priors(params: torch.Tensor, config) -> torch.Tensor:
       '''Compute prior energy for myobject parameters. Returns scalar.'''
       ...

3. Add to OBJECT_REGISTRY:
   OBJECT_REGISTRY['myobject'] = {
       'sdf': compute_myobject_sdf,
       'prior': compute_myobject_priors,
       'param_count': N,
       'param_names': ['cx', 'cy', ...],
   }

=============================================================================
"""

import torch
import numpy as np
from typing import Dict, Callable, Any


# =============================================================================
# SDF CONFIGURATION CONSTANTS
# =============================================================================
# These can be tuned to adjust SDF behavior

# Smooth minimum parameter for internal points (meters)
# Smaller = closer to hard min, larger = smoother gradients
SDF_SMOOTH_K = 0.02

# Scale factor for internal points (0-1)
# Reduces influence of internal points which are less reliable
SDF_INSIDE_SCALE = 0.3

# Table fixed dimensions
TABLE_TOP_THICKNESS = 0.03  # 3cm
TABLE_LEG_RADIUS = 0.025    # 2.5cm

# Chair fixed dimension
CHAIR_SEAT_THICKNESS = 0.05  # 5cm


# =============================================================================
# BOX (6 parameters)
# =============================================================================
# State: [cx, cy, w, h, d, theta]
# - cx, cy: center position in XY plane
# - w, h: width and height in XY plane
# - d: depth (vertical extent, Z axis)
# - theta: rotation angle around Z axis

def compute_box_sdf(points_xyz: torch.Tensor, box_params: torch.Tensor) -> torch.Tensor:
    """
    Compute Signed Distance Function for an oriented 3D box.

    The box sits on the floor (z=0) and extends upward to z=d.

    Uses smooth minimum for inside points so gradients flow to ALL dimensions,
    not just the closest face. This allows proper optimization when LIDAR
    only sees some faces of the box.

    Args:
        points_xyz: [N, 3] points (x, y, z)
        box_params: [6] tensor [cx, cy, w, h, d, theta]

    Returns:
        [N] SDF values (positive=outside, negative=inside, zero=surface)
    """
    cx, cy, w, h, d, theta = box_params[0], box_params[1], box_params[2], \
                              box_params[3], box_params[4], box_params[5]

    # Transform to local box frame
    cos_t = torch.cos(-theta)
    sin_t = torch.sin(-theta)

    px = points_xyz[:, 0] - cx
    py = points_xyz[:, 1] - cy

    local_x = px * cos_t - py * sin_t
    local_y = px * sin_t + py * cos_t

    # Half dimensions
    half_w, half_h, half_d = w / 2, h / 2, d / 2

    # Distance to faces (negative = inside, positive = outside)
    dx = torch.abs(local_x) - half_w
    dy = torch.abs(local_y) - half_h

    # Z: box sits on floor, center at half_d
    local_z = points_xyz[:, 2] - half_d
    dz = torch.abs(local_z) - half_d

    # Outside: standard Euclidean distance
    outside = torch.linalg.norm(torch.stack([
        torch.clamp(dx, min=0),
        torch.clamp(dy, min=0),
        torch.clamp(dz, min=0)
    ], dim=-1), dim=-1)

    # Inside: use smooth approximation instead of hard min
    # This ensures gradients flow to ALL dimensions, not just the closest
    # Smooth min: -softplus(-dx, -dy, -dz) ≈ min(dx, dy, dz) but differentiable everywhere
    # We use log-sum-exp: smooth_min(a,b,c) ≈ -k * log(exp(-a/k) + exp(-b/k) + exp(-c/k))
    # where k controls smoothness (smaller k = closer to hard min)
    k = SDF_SMOOTH_K
    inside_scale = SDF_INSIDE_SCALE

    # Only compute smooth min for inside points (where all d < 0)
    is_inside = (dx < 0) & (dy < 0) & (dz < 0)

    # For inside points: smooth minimum of (dx, dy, dz)
    # Using negative log-sum-exp trick
    inside_term = torch.zeros_like(dx)
    if is_inside.any():
        dx_in = dx[is_inside]
        dy_in = dy[is_inside]
        dz_in = dz[is_inside]

        # Smooth min using log-sum-exp
        # smooth_min = -k * log(exp(-dx/k) + exp(-dy/k) + exp(-dz/k))
        stacked = torch.stack([dx_in / k, dy_in / k, dz_in / k], dim=-1)
        smooth_min = k * torch.logsumexp(-stacked, dim=-1) * (-1)
        inside_term[is_inside] = inside_scale * smooth_min  # Scale down internal points

    return outside + inside_term


def compute_box_priors(params: torch.Tensor, config, robot_pose: np.ndarray = None) -> torch.Tensor:
    """
    Compute prior energy terms for box parameters.

    Includes:
    - Angle alignment prior: boxes tend to align with room axes (0°, 90°)

    Args:
        params: [6] tensor [cx, cy, w, h, d, theta]
        config: BoxBeliefConfig with prior parameters
        robot_pose: [x, y, theta] robot pose (to transform angle to room frame)

    Returns:
        Total prior energy (scalar tensor)
    """
    theta = params[5]

    # Transform to room frame if robot_pose provided
    if robot_pose is not None:
        theta_room = theta + robot_pose[2]
    else:
        theta_room = theta

    # Normalize to [-π/2, π/2]
    while theta_room > np.pi/2: theta_room = theta_room - np.pi
    while theta_room < -np.pi/2: theta_room = theta_room + np.pi

    # Angle alignment prior
    sigma = getattr(config, 'angle_alignment_sigma', 0.1)
    weight = getattr(config, 'angle_alignment_weight', 0.5)
    precision = 1.0 / (sigma ** 2)

    # Distance to nearest aligned angle (0, ±π/2)
    dist_to_0 = theta_room ** 2
    dist_to_pos90 = (theta_room - np.pi/2) ** 2
    dist_to_neg90 = (theta_room + np.pi/2) ** 2
    min_dist_sq = torch.minimum(dist_to_0, torch.minimum(dist_to_pos90, dist_to_neg90))

    # Prior energy: (λ/2) * (θ - μ_nearest)²
    angle_prior = weight * 0.5 * precision * min_dist_sq

    return angle_prior


# =============================================================================
# CYLINDER (4 parameters)
# =============================================================================
# State: [cx, cy, r, h]
# - cx, cy: center position
# - r: radius
# - h: height (vertical extent)

def compute_cylinder_sdf(points_xyz: torch.Tensor, cyl_params: torch.Tensor) -> torch.Tensor:
    """
    Compute Signed Distance Function for a vertical cylinder.

    Args:
        points_xyz: [N, 3] points (x, y, z)
        cyl_params: [4] tensor [cx, cy, r, h]

    Returns:
        [N] SDF values
    """
    cx, cy, r, h = cyl_params[0], cyl_params[1], cyl_params[2], cyl_params[3]

    # Distance from axis
    dx = points_xyz[:, 0] - cx
    dy = points_xyz[:, 1] - cy
    dist_xy = torch.sqrt(dx**2 + dy**2)

    d_radial = dist_xy - r

    # Vertical distance
    half_h = h / 2
    local_z = points_xyz[:, 2] - half_h
    d_vertical = torch.abs(local_z) - half_h

    # Combine
    outside_radial = torch.clamp(d_radial, min=0)
    outside_vertical = torch.clamp(d_vertical, min=0)

    outside = torch.sqrt(outside_radial**2 + outside_vertical**2)
    inside = torch.minimum(torch.maximum(d_radial, d_vertical), torch.zeros_like(d_radial))

    return outside + inside


def compute_cylinder_priors(params: torch.Tensor, config, robot_pose: np.ndarray = None) -> torch.Tensor:
    """
    Compute prior energy for cylinder parameters.

    Cylinders have no orientation prior (rotationally symmetric).

    Returns:
        Zero tensor (no priors for cylinder)
    """
    return torch.tensor(0.0, device=params.device)


# =============================================================================
# TABLE (7 parameters)
# =============================================================================
# State: [cx, cy, w, h, table_height, leg_length, theta]
# - cx, cy: center position
# - w, h: width and depth of table top
# - table_height: height of table surface from floor
# - leg_length: length of legs (free parameter)
# - theta: rotation angle
#
# Fixed constants:
# - TOP_THICKNESS = 0.03m (3cm)
# - LEG_RADIUS = 0.025m (2.5cm)

def compute_table_sdf(points_xyz: torch.Tensor, table_params: torch.Tensor) -> torch.Tensor:
    """
    Compute Signed Distance Function for a table (box top + 4 cylindrical legs).

    Args:
        points_xyz: [N, 3] points (x, y, z)
        table_params: [7] tensor [cx, cy, w, h, table_height, leg_length, theta]

    Returns:
        [N] SDF values
    """
    # Use module-level constants
    TOP_THICKNESS = TABLE_TOP_THICKNESS
    LEG_RADIUS = TABLE_LEG_RADIUS

    cx, cy = table_params[0], table_params[1]
    w, h = table_params[2], table_params[3]
    table_height = table_params[4]
    leg_length = table_params[5]
    theta = table_params[6]

    # Transform to local frame
    cos_t = torch.cos(-theta)
    sin_t = torch.sin(-theta)

    px = points_xyz[:, 0] - cx
    py = points_xyz[:, 1] - cy

    local_x = px * cos_t - py * sin_t
    local_y = px * sin_t + py * cos_t
    local_z = points_xyz[:, 2]

    # TABLE TOP
    half_w, half_h = w / 2, h / 2
    half_t = TOP_THICKNESS / 2
    top_center_z = table_height - half_t

    top_local_z = local_z - top_center_z

    dx_top = torch.abs(local_x) - half_w
    dy_top = torch.abs(local_y) - half_h
    dz_top = torch.abs(top_local_z) - half_t

    # Outside: standard Euclidean distance
    outside_top = torch.linalg.norm(torch.stack([
        torch.clamp(dx_top, min=0),
        torch.clamp(dy_top, min=0),
        torch.clamp(dz_top, min=0)
    ], dim=-1), dim=-1)

    # Inside: use smooth minimum like box SDF
    k = SDF_SMOOTH_K
    inside_scale = SDF_INSIDE_SCALE

    is_inside_top = (dx_top < 0) & (dy_top < 0) & (dz_top < 0)
    inside_top = torch.zeros_like(dx_top)
    if is_inside_top.any():
        dx_in = dx_top[is_inside_top]
        dy_in = dy_top[is_inside_top]
        dz_in = dz_top[is_inside_top]
        stacked = torch.stack([dx_in / k, dy_in / k, dz_in / k], dim=-1)
        smooth_min = k * torch.logsumexp(-stacked, dim=-1) * (-1)
        inside_top[is_inside_top] = inside_scale * smooth_min

    sdf_top = outside_top + inside_top

    # LEGS (4 cylinders at corners)
    leg_inset = LEG_RADIUS + 0.02
    leg_positions = [
        (half_w - leg_inset, half_h - leg_inset),
        (-half_w + leg_inset, half_h - leg_inset),
        (-half_w + leg_inset, -half_h + leg_inset),
        (half_w - leg_inset, -half_h + leg_inset),
    ]

    leg_center_z = leg_length / 2
    leg_half_h = leg_length / 2

    sdf_legs = torch.full_like(sdf_top, float('inf'))

    for leg_x, leg_y in leg_positions:
        dx_leg = local_x - leg_x
        dy_leg = local_y - leg_y
        dist_xy = torch.sqrt(dx_leg**2 + dy_leg**2)

        d_radial = dist_xy - LEG_RADIUS

        leg_local_z = local_z - leg_center_z
        d_vertical = torch.abs(leg_local_z) - leg_half_h

        outside_radial = torch.clamp(d_radial, min=0)
        outside_vertical = torch.clamp(d_vertical, min=0)

        outside_leg = torch.sqrt(outside_radial**2 + outside_vertical**2)
        inside_leg = torch.minimum(torch.maximum(d_radial, d_vertical), torch.zeros_like(d_radial))

        sdf_leg = outside_leg + inside_leg
        sdf_legs = torch.minimum(sdf_legs, sdf_leg)

    return torch.minimum(sdf_top, sdf_legs)


def compute_table_priors(params: torch.Tensor, config, robot_pose: np.ndarray = None) -> torch.Tensor:
    """
    Compute prior energy for table parameters.

    Includes:
    - Angle alignment prior: tables align with room axes

    Args:
        params: [7] tensor [cx, cy, w, h, table_height, leg_length, theta]
        config: Configuration with prior parameters
        robot_pose: Robot pose for angle transformation

    Returns:
        Total prior energy
    """
    theta = params[6]

    if robot_pose is not None:
        theta_room = theta + robot_pose[2]
    else:
        theta_room = theta

    while theta_room > np.pi/2: theta_room = theta_room - np.pi
    while theta_room < -np.pi/2: theta_room = theta_room + np.pi

    sigma = getattr(config, 'angle_alignment_sigma', 0.1)
    weight = getattr(config, 'angle_alignment_weight', 0.5)
    precision = 1.0 / (sigma ** 2)

    dist_to_0 = theta_room ** 2
    dist_to_pos90 = (theta_room - np.pi/2) ** 2
    dist_to_neg90 = (theta_room + np.pi/2) ** 2
    min_dist_sq = torch.minimum(dist_to_0, torch.minimum(dist_to_pos90, dist_to_neg90))

    return weight * 0.5 * precision * min_dist_sq


# =============================================================================
# CHAIR (8 parameters)
# =============================================================================
# State: [cx, cy, seat_w, seat_d, seat_h, back_h, back_thickness, theta]
# - cx, cy: center position
# - seat_w, seat_d: seat width and depth
# - seat_h: seat height from floor
# - back_h: backrest height above seat
# - back_thickness: backrest thickness
# - theta: rotation angle

def compute_chair_sdf(points_xyz: torch.Tensor, chair_params: torch.Tensor) -> torch.Tensor:
    """
    Compute Signed Distance Function for a chair (seat + backrest).

    Args:
        points_xyz: [N, 3] points (x, y, z)
        chair_params: [8] tensor [cx, cy, seat_w, seat_d, seat_h, back_h, back_thickness, theta]

    Returns:
        [N] SDF values
    """
    SEAT_THICKNESS = CHAIR_SEAT_THICKNESS

    cx, cy = chair_params[0], chair_params[1]
    seat_w, seat_d = chair_params[2], chair_params[3]
    seat_h = chair_params[4]
    back_h = chair_params[5]
    back_thickness = chair_params[6]
    theta = chair_params[7]

    # Transform to local frame
    cos_t = torch.cos(-theta)
    sin_t = torch.sin(-theta)

    px = points_xyz[:, 0] - cx
    py = points_xyz[:, 1] - cy

    local_x = px * cos_t - py * sin_t
    local_y = px * sin_t + py * cos_t
    local_z = points_xyz[:, 2]

    # SEAT
    seat_center_z = seat_h - SEAT_THICKNESS / 2

    dx_seat = torch.abs(local_x) - seat_w / 2
    dy_seat = torch.abs(local_y) - seat_d / 2
    dz_seat = torch.abs(local_z - seat_center_z) - SEAT_THICKNESS / 2

    qmax_seat = torch.maximum(dx_seat, torch.maximum(dy_seat, dz_seat))
    outside_seat = torch.linalg.norm(torch.stack([
        torch.clamp(dx_seat, min=0),
        torch.clamp(dy_seat, min=0),
        torch.clamp(dz_seat, min=0)
    ], dim=-1), dim=-1)
    sdf_seat = outside_seat + torch.minimum(qmax_seat, torch.zeros_like(qmax_seat))

    # BACKREST
    back_center_y = -seat_d / 2 + back_thickness / 2
    back_center_z = seat_h + back_h / 2

    dx_back = torch.abs(local_x) - seat_w / 2
    dy_back = torch.abs(local_y - back_center_y) - back_thickness / 2
    dz_back = torch.abs(local_z - back_center_z) - back_h / 2

    qmax_back = torch.maximum(dx_back, torch.maximum(dy_back, dz_back))
    outside_back = torch.linalg.norm(torch.stack([
        torch.clamp(dx_back, min=0),
        torch.clamp(dy_back, min=0),
        torch.clamp(dz_back, min=0)
    ], dim=-1), dim=-1)
    sdf_back = outside_back + torch.minimum(qmax_back, torch.zeros_like(qmax_back))

    return torch.minimum(sdf_seat, sdf_back)


def compute_chair_priors(params: torch.Tensor, config, robot_pose: np.ndarray = None) -> torch.Tensor:
    """
    Compute prior energy for chair parameters.

    Chairs typically align with room axes or tables.

    Returns:
        Angle alignment prior energy
    """
    theta = params[7]

    if robot_pose is not None:
        theta_room = theta + robot_pose[2]
    else:
        theta_room = theta

    while theta_room > np.pi/2: theta_room = theta_room - np.pi
    while theta_room < -np.pi/2: theta_room = theta_room + np.pi

    sigma = getattr(config, 'angle_alignment_sigma', 0.15)  # Chairs are less strict
    weight = getattr(config, 'angle_alignment_weight', 0.3)
    precision = 1.0 / (sigma ** 2)

    dist_to_0 = theta_room ** 2
    dist_to_pos90 = (theta_room - np.pi/2) ** 2
    dist_to_neg90 = (theta_room + np.pi/2) ** 2
    min_dist_sq = torch.minimum(dist_to_0, torch.minimum(dist_to_pos90, dist_to_neg90))

    return weight * 0.5 * precision * min_dist_sq


# =============================================================================
# OBJECT REGISTRY
# =============================================================================

OBJECT_REGISTRY: Dict[str, Dict[str, Any]] = {
    'box': {
        'sdf': compute_box_sdf,
        'prior': compute_box_priors,
        'param_count': 6,
        'param_names': ['cx', 'cy', 'w', 'h', 'd', 'theta'],
        'description': 'Oriented 3D box sitting on floor',
    },
    'cylinder': {
        'sdf': compute_cylinder_sdf,
        'prior': compute_cylinder_priors,
        'param_count': 4,
        'param_names': ['cx', 'cy', 'r', 'h'],
        'description': 'Vertical cylinder sitting on floor',
    },
    'table': {
        'sdf': compute_table_sdf,
        'prior': compute_table_priors,
        'param_count': 7,
        'param_names': ['cx', 'cy', 'w', 'h', 'table_height', 'leg_length', 'theta'],
        'description': 'Table with box top and 4 cylindrical legs',
    },
    'chair': {
        'sdf': compute_chair_sdf,
        'prior': compute_chair_priors,
        'param_count': 8,
        'param_names': ['cx', 'cy', 'seat_w', 'seat_d', 'seat_h', 'back_h', 'back_thickness', 'theta'],
        'description': 'Chair with seat and backrest',
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_sdf_function(object_type: str) -> Callable:
    """Get the SDF function for a given object type."""
    if object_type not in OBJECT_REGISTRY:
        raise ValueError(f"Unknown object type: {object_type}. "
                        f"Available: {list(OBJECT_REGISTRY.keys())}")
    return OBJECT_REGISTRY[object_type]['sdf']


def get_prior_function(object_type: str) -> Callable:
    """Get the prior function for a given object type."""
    if object_type not in OBJECT_REGISTRY:
        raise ValueError(f"Unknown object type: {object_type}. "
                        f"Available: {list(OBJECT_REGISTRY.keys())}")
    return OBJECT_REGISTRY[object_type]['prior']


def get_object_info(object_type: str) -> Dict[str, Any]:
    """Get full info for an object type."""
    if object_type not in OBJECT_REGISTRY:
        raise ValueError(f"Unknown object type: {object_type}. "
                        f"Available: {list(OBJECT_REGISTRY.keys())}")
    return OBJECT_REGISTRY[object_type]


def list_object_types() -> list:
    """List all available object types."""
    return list(OBJECT_REGISTRY.keys())


def print_object_summary():
    """Print summary of all registered object types."""
    print("\n" + "="*70)
    print("REGISTERED OBJECT TYPES")
    print("="*70)
    for name, info in OBJECT_REGISTRY.items():
        print(f"\n{name.upper()} ({info['param_count']} parameters)")
        print(f"  Description: {info['description']}")
        print(f"  Parameters:  {info['param_names']}")
    print("\n" + "="*70 + "\n")
