#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Box SDF and Prior Functions

State: [cx, cy, w, h, d, theta] (6 parameters)
- cx, cy: center position in XY plane
- w, h: width and height in XY plane
- d: depth (vertical extent, Z axis)
- theta: rotation angle around Z axis
"""

import torch
import numpy as np

from src.objects.sdf_constants import SDF_SMOOTH_K, SDF_INSIDE_SCALE

# Box parameters
BOX_PARAM_COUNT = 6
BOX_PARAM_NAMES = ['cx', 'cy', 'w', 'h', 'd', 'theta']


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
    k = SDF_SMOOTH_K
    inside_scale = SDF_INSIDE_SCALE

    # Only compute smooth min for inside points (where all d < 0)
    is_inside = (dx < 0) & (dy < 0) & (dz < 0)

    # For inside points: smooth minimum of (dx, dy, dz)
    inside_term = torch.zeros_like(dx)
    if is_inside.any():
        dx_in = dx[is_inside]
        dy_in = dy[is_inside]
        dz_in = dz[is_inside]

        # Smooth min using log-sum-exp
        stacked = torch.stack([dx_in / k, dy_in / k, dz_in / k], dim=-1)
        smooth_min = k * torch.logsumexp(-stacked, dim=-1) * (-1)
        inside_term[is_inside] = inside_scale * smooth_min

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
