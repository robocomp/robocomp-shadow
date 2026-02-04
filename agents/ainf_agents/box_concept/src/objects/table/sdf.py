#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Table SDF and Prior Functions

State: [cx, cy, w, h, table_height, leg_length, theta] (7 parameters)
- cx, cy: center position
- w, h: width and depth of table top
- table_height: height of table surface from floor
- leg_length: length of legs
- theta: rotation angle

Fixed constants:
- TOP_THICKNESS = 0.03m (3cm)
- LEG_RADIUS = 0.025m (2.5cm)
"""

import torch
import numpy as np

from src.objects.sdf_constants import SDF_SMOOTH_K, SDF_INSIDE_SCALE

# Table parameters
TABLE_PARAM_COUNT = 7
TABLE_PARAM_NAMES = ['cx', 'cy', 'w', 'h', 'table_height', 'leg_length', 'theta']

# Fixed dimensions
TABLE_TOP_THICKNESS = 0.03  # 3cm
TABLE_LEG_RADIUS = 0.025    # 2.5cm


def compute_table_sdf(points_xyz: torch.Tensor, table_params: torch.Tensor) -> torch.Tensor:
    """
    Compute Signed Distance Function for a table (box top + 4 cylindrical legs).

    Args:
        points_xyz: [N, 3] points (x, y, z)
        table_params: [7] tensor [cx, cy, w, h, table_height, leg_length, theta]

    Returns:
        [N] SDF values
    """
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
