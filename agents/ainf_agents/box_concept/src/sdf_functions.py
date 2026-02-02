#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
SDF Functions - Signed Distance Functions for various object geometries.

Each SDF function computes the distance from points to the object surface:
- Positive: point is outside the object
- Negative: point is inside the object
- Zero: point is on the surface

These are used as the likelihood model in Active Inference:
p(observation | state) ∝ exp(-SDF²/2σ²)

Adding a new object type:
1. Implement compute_X_sdf(points, params) function
2. Define the parameter vector structure in comments
"""

import torch
import numpy as np
from typing import Tuple

from src.belief_core import DTYPE


def compute_box_sdf(points_xyz: torch.Tensor, box_params: torch.Tensor) -> torch.Tensor:
    """
    Compute Signed Distance Function for an oriented 3D box.

    The box sits on the floor (z=0) and extends upward to z=d.
    Center is at (cx, cy, d/2) in world coordinates.

    Args:
        points_xyz: [N, 3] points (x, y, z)
        box_params: [6] tensor [cx, cy, w, h, d, theta]
            - cx, cy: center position in XY plane
            - w: width (along rotated X axis)
            - h: height (along rotated Y axis)
            - d: depth (vertical extent, along Z)
            - theta: rotation angle around Z axis

    Returns:
        [N] SDF values for each point
    """
    cx, cy, w, h, d, theta = box_params[0], box_params[1], box_params[2], \
                              box_params[3], box_params[4], box_params[5]

    # Transform points to local box frame (rotation around z-axis)
    cos_t = torch.cos(-theta)
    sin_t = torch.sin(-theta)

    # Translate to box center (XY)
    px = points_xyz[:, 0] - cx
    py = points_xyz[:, 1] - cy

    # Rotate to align with box axes (rotation in XY plane)
    local_x = px * cos_t - py * sin_t
    local_y = px * sin_t + py * cos_t

    # Half dimensions
    half_w = w / 2
    half_h = h / 2
    half_d = d / 2

    # Distance to box faces in local frame
    dx = torch.abs(local_x) - half_w
    dy = torch.abs(local_y) - half_h

    # Z: box sits on floor (z=0) and extends to z=d
    # Center of box in Z is at half_d
    local_z = points_xyz[:, 2] - half_d
    dz = torch.abs(local_z) - half_d

    # Standard 3D box SDF formula
    qmax = torch.maximum(dx, torch.maximum(dy, dz))
    outside = torch.linalg.norm(torch.stack([
        torch.clamp(dx, min=0),
        torch.clamp(dy, min=0),
        torch.clamp(dz, min=0)
    ], dim=-1), dim=-1)

    inside_term = torch.minimum(qmax, torch.zeros_like(qmax))
    sdf = outside + inside_term

    return sdf


def compute_cylinder_sdf(points_xyz: torch.Tensor, cyl_params: torch.Tensor) -> torch.Tensor:
    """
    Compute Signed Distance Function for a vertical cylinder.

    The cylinder sits on the floor (z=0) and extends upward to z=h.

    Args:
        points_xyz: [N, 3] points (x, y, z)
        cyl_params: [4] tensor [cx, cy, r, h]
            - cx, cy: center position in XY plane
            - r: radius
            - h: height (vertical extent)

    Returns:
        [N] SDF values for each point
    """
    cx, cy, r, h = cyl_params[0], cyl_params[1], cyl_params[2], cyl_params[3]

    # Distance from central axis in XY
    dx = points_xyz[:, 0] - cx
    dy = points_xyz[:, 1] - cy
    dist_xy = torch.sqrt(dx**2 + dy**2)

    # Distance from cylinder surface in XY
    d_radial = dist_xy - r

    # Distance from top/bottom caps
    half_h = h / 2
    local_z = points_xyz[:, 2] - half_h
    d_vertical = torch.abs(local_z) - half_h

    # Combine radial and vertical distances
    outside_radial = torch.clamp(d_radial, min=0)
    outside_vertical = torch.clamp(d_vertical, min=0)

    outside = torch.sqrt(outside_radial**2 + outside_vertical**2)
    inside = torch.minimum(torch.maximum(d_radial, d_vertical), torch.zeros_like(d_radial))

    return outside + inside


def compute_table_sdf(points_xyz: torch.Tensor, table_params: torch.Tensor) -> torch.Tensor:
    """
    Compute Signed Distance Function for a table (box top + 4 legs).

    Simplified model: thin box for the top surface.
    Full model would include legs as separate boxes/cylinders.

    Args:
        points_xyz: [N, 3] points (x, y, z)
        table_params: [7] tensor [cx, cy, w, h, table_height, top_thickness, theta]
            - cx, cy: center position in XY plane
            - w, h: width and height of table top
            - table_height: height of table surface from floor
            - top_thickness: thickness of table top
            - theta: rotation angle

    Returns:
        [N] SDF values for each point
    """
    cx, cy = table_params[0], table_params[1]
    w, h = table_params[2], table_params[3]
    table_height = table_params[4]
    top_thickness = table_params[5]
    theta = table_params[6]

    # Table top as a box
    # Position the top box so it sits at table_height
    top_center_z = table_height - top_thickness / 2

    # Transform points to local frame
    cos_t = torch.cos(-theta)
    sin_t = torch.sin(-theta)

    px = points_xyz[:, 0] - cx
    py = points_xyz[:, 1] - cy

    local_x = px * cos_t - py * sin_t
    local_y = px * sin_t + py * cos_t
    local_z = points_xyz[:, 2] - top_center_z

    # Half dimensions of table top
    half_w = w / 2
    half_h = h / 2
    half_t = top_thickness / 2

    # SDF for table top
    dx = torch.abs(local_x) - half_w
    dy = torch.abs(local_y) - half_h
    dz = torch.abs(local_z) - half_t

    qmax = torch.maximum(dx, torch.maximum(dy, dz))
    outside = torch.linalg.norm(torch.stack([
        torch.clamp(dx, min=0),
        torch.clamp(dy, min=0),
        torch.clamp(dz, min=0)
    ], dim=-1), dim=-1)

    inside = torch.minimum(qmax, torch.zeros_like(qmax))

    return outside + inside


def compute_chair_sdf(points_xyz: torch.Tensor, chair_params: torch.Tensor) -> torch.Tensor:
    """
    Compute Signed Distance Function for a chair (seat + backrest).

    Simplified model: two boxes (seat and backrest).

    Args:
        points_xyz: [N, 3] points (x, y, z)
        chair_params: [8] tensor [cx, cy, seat_w, seat_d, seat_h, back_h, back_thickness, theta]
            - cx, cy: center position
            - seat_w, seat_d: seat width and depth
            - seat_h: seat height from floor
            - back_h: backrest height above seat
            - back_thickness: backrest thickness
            - theta: rotation angle

    Returns:
        [N] SDF values for each point
    """
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

    # Seat SDF (thin box)
    seat_thickness = 0.05  # 5cm thick seat
    seat_center_z = seat_h - seat_thickness / 2

    dx_seat = torch.abs(local_x) - seat_w / 2
    dy_seat = torch.abs(local_y) - seat_d / 2
    dz_seat = torch.abs(local_z - seat_center_z) - seat_thickness / 2

    qmax_seat = torch.maximum(dx_seat, torch.maximum(dy_seat, dz_seat))
    outside_seat = torch.linalg.norm(torch.stack([
        torch.clamp(dx_seat, min=0),
        torch.clamp(dy_seat, min=0),
        torch.clamp(dz_seat, min=0)
    ], dim=-1), dim=-1)
    sdf_seat = outside_seat + torch.minimum(qmax_seat, torch.zeros_like(qmax_seat))

    # Backrest SDF (positioned at back of seat)
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

    # Union of seat and backrest (min of SDFs)
    return torch.minimum(sdf_seat, sdf_back)


# Registry of available SDF functions
SDF_REGISTRY = {
    'box': compute_box_sdf,
    'cylinder': compute_cylinder_sdf,
    'table': compute_table_sdf,
    'chair': compute_chair_sdf,
}


def get_sdf_function(object_type: str):
    """
    Get the SDF function for a given object type.

    Args:
        object_type: Name of the object type ('box', 'cylinder', 'table', 'chair')

    Returns:
        SDF function

    Raises:
        ValueError if object type is not registered
    """
    if object_type not in SDF_REGISTRY:
        raise ValueError(f"Unknown object type: {object_type}. "
                        f"Available: {list(SDF_REGISTRY.keys())}")
    return SDF_REGISTRY[object_type]
