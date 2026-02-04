#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Chair SDF and Prior Functions

State: [cx, cy, seat_w, seat_d, seat_h, back_h, theta] (7 parameters)
- cx, cy: center position
- seat_w, seat_d: seat width and depth
- seat_h: seat height from floor
- back_h: backrest height above seat
- theta: rotation angle

Fixed constants:
- SEAT_THICKNESS = 0.05m (5cm)
- BACK_THICKNESS = 0.05m (5cm)
- LEG_RADIUS = 0.02m (2cm)
"""

import torch
import numpy as np

# Chair parameters
CHAIR_PARAM_COUNT = 7
CHAIR_PARAM_NAMES = ['cx', 'cy', 'seat_w', 'seat_d', 'seat_h', 'back_h', 'theta']

# Fixed dimensions
CHAIR_SEAT_THICKNESS = 0.05  # 5cm
CHAIR_BACK_THICKNESS = 0.05  # 5cm
CHAIR_LEG_RADIUS = 0.02      # 2cm
CHAIR_LEG_INSET = 0.03       # 3cm inset from seat edge


def _sdf_cylinder_z(local_x: torch.Tensor, local_y: torch.Tensor, local_z: torch.Tensor,
                    cx: float, cy: float, z_min: float, z_max: float, radius: float) -> torch.Tensor:
    """
    Compute SDF for a vertical cylinder (along Z axis).

    Args:
        local_x, local_y, local_z: Point coordinates in local frame
        cx, cy: Cylinder center in XY plane
        z_min, z_max: Cylinder Z extent
        radius: Cylinder radius

    Returns:
        SDF values for each point
    """
    # Distance from cylinder axis in XY plane
    dx = local_x - cx
    dy = local_y - cy
    dist_xy = torch.sqrt(dx**2 + dy**2)

    # Distance from cylinder surface in XY
    d_radial = dist_xy - radius

    # Distance from cylinder caps in Z
    z_center = (z_min + z_max) / 2
    half_height = (z_max - z_min) / 2
    d_z = torch.abs(local_z - z_center) - half_height

    # Combine: outside if either radial or Z distance is positive
    outside = torch.sqrt(torch.clamp(d_radial, min=0)**2 + torch.clamp(d_z, min=0)**2)
    inside = torch.minimum(torch.maximum(d_radial, d_z), torch.zeros_like(d_radial))

    return outside + inside


def compute_chair_sdf(points_xyz: torch.Tensor, chair_params: torch.Tensor) -> torch.Tensor:
    """
    Compute Signed Distance Function for a chair (seat + backrest + 4 legs).

    Args:
        points_xyz: [N, 3] points (x, y, z)
        chair_params: [7] tensor [cx, cy, seat_w, seat_d, seat_h, back_h, theta]

    Returns:
        [N] SDF values
    """
    SEAT_THICKNESS = CHAIR_SEAT_THICKNESS
    BACK_THICKNESS = CHAIR_BACK_THICKNESS
    LEG_RADIUS = CHAIR_LEG_RADIUS
    LEG_INSET = CHAIR_LEG_INSET

    cx, cy = chair_params[0], chair_params[1]
    seat_w, seat_d = chair_params[2], chair_params[3]
    seat_h = chair_params[4]
    back_h = chair_params[5]
    theta = chair_params[6]

    # Transform to local frame
    cos_t = torch.cos(-theta)
    sin_t = torch.sin(-theta)

    px = points_xyz[:, 0] - cx
    py = points_xyz[:, 1] - cy

    local_x = px * cos_t - py * sin_t
    local_y = px * sin_t + py * cos_t
    local_z = points_xyz[:, 2]

    # =========================================================
    # SEAT (box)
    # =========================================================
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

    # =========================================================
    # BACKREST (box)
    # Backrest is at the BACK of the seat (POSITIVE local Y)
    # When angle=0, backrest faces +Y direction
    # =========================================================
    back_center_y = seat_d / 2 - BACK_THICKNESS / 2  # +Y side (back of chair)
    back_center_z = seat_h + back_h / 2

    dx_back = torch.abs(local_x) - seat_w / 2
    dy_back = torch.abs(local_y - back_center_y) - BACK_THICKNESS / 2
    dz_back = torch.abs(local_z - back_center_z) - back_h / 2

    qmax_back = torch.maximum(dx_back, torch.maximum(dy_back, dz_back))
    outside_back = torch.linalg.norm(torch.stack([
        torch.clamp(dx_back, min=0),
        torch.clamp(dy_back, min=0),
        torch.clamp(dz_back, min=0)
    ], dim=-1), dim=-1)
    sdf_back = outside_back + torch.minimum(qmax_back, torch.zeros_like(qmax_back))

    # =========================================================
    # LEGS (4 vertical cylinders at corners)
    # =========================================================
    leg_z_min = 0.0
    leg_z_max = seat_h - SEAT_THICKNESS  # Legs go from floor to bottom of seat

    # Leg positions (inset from seat corners)
    half_w = seat_w / 2 - LEG_INSET
    half_d = seat_d / 2 - LEG_INSET

    # Compute SDF for each leg and take minimum
    sdf_leg1 = _sdf_cylinder_z(local_x, local_y, local_z,  half_w,  half_d, leg_z_min, leg_z_max, LEG_RADIUS)
    sdf_leg2 = _sdf_cylinder_z(local_x, local_y, local_z, -half_w,  half_d, leg_z_min, leg_z_max, LEG_RADIUS)
    sdf_leg3 = _sdf_cylinder_z(local_x, local_y, local_z, -half_w, -half_d, leg_z_min, leg_z_max, LEG_RADIUS)
    sdf_leg4 = _sdf_cylinder_z(local_x, local_y, local_z,  half_w, -half_d, leg_z_min, leg_z_max, LEG_RADIUS)

    sdf_legs = torch.minimum(torch.minimum(sdf_leg1, sdf_leg2), torch.minimum(sdf_leg3, sdf_leg4))

    # =========================================================
    # COMBINE ALL PARTS (union = minimum of all SDFs)
    # =========================================================
    return torch.minimum(torch.minimum(sdf_seat, sdf_back), sdf_legs)


def _normalize_angle(theta: torch.Tensor) -> torch.Tensor:
    """Normalize angle to [-pi, pi] range (tensor-safe)."""
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def compute_chair_priors(params: torch.Tensor, config, robot_pose: np.ndarray = None,
                         previous_angle: float = None) -> torch.Tensor:
    """
    Compute prior energy for chair parameters.

    Chairs typically align with room axes. Also penalizes sudden angle changes
    to prevent flipping when the robot moves around.

    Args:
        params: Chair parameters [cx, cy, seat_w, seat_d, seat_h, back_h, theta]
        config: ChairBeliefConfig
        robot_pose: Robot pose [x, y, theta] for frame transformation
        previous_angle: Previous angle estimate IN ROBOT FRAME (for continuity prior)

    Returns:
        Total prior energy (alignment + continuity)
    """
    theta = params[6]  # Angle in robot frame

    # Transform to room frame for alignment prior
    if robot_pose is not None:
        theta_room = theta + robot_pose[2]
    else:
        theta_room = theta

    # Normalize to [-pi, pi]
    theta_room = _normalize_angle(theta_room)

    # =================================================================
    # 1. AXIS ALIGNMENT PRIOR
    # Chairs typically align with room axes (0, 90, 180, -90 degrees)
    # =================================================================
    sigma = getattr(config, 'angle_alignment_sigma', 0.15)
    weight = getattr(config, 'angle_alignment_weight', 0.3)
    precision = 1.0 / (sigma ** 2)

    # Distance to nearest axis-aligned angle (0, ±90°, 180°)
    # Use cos(2*theta) which is 1 at 0°, 90°, 180°, 270°
    alignment_cost = (1.0 - torch.cos(2 * theta_room)) / 2.0
    alignment_prior = weight * precision * alignment_cost

    # =================================================================
    # 2. TEMPORAL CONTINUITY PRIOR (prevent sudden flips)
    # Compare in ROBOT FRAME (both theta and previous_angle are in robot frame)
    # =================================================================
    continuity_prior = torch.tensor(0.0, dtype=params.dtype, device=params.device)

    if previous_angle is not None:
        continuity_weight = getattr(config, 'angle_continuity_weight', 10.0)
        continuity_sigma = getattr(config, 'angle_continuity_sigma', 0.1)

        # Both angles are in robot frame - compare directly
        prev_theta = torch.tensor(previous_angle, dtype=params.dtype, device=params.device)
        angle_diff = _normalize_angle(theta - prev_theta)  # Compare in robot frame!

        continuity_precision = 1.0 / (continuity_sigma ** 2)
        continuity_prior = continuity_weight * 0.5 * continuity_precision * (angle_diff ** 2)

    return alignment_prior + continuity_prior
