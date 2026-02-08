#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
TV SDF and Prior Functions

A TV is modeled as a thin rectangular panel (screen) mounted on a wall.
The screen is much wider than it is deep, and mounted at a certain height from the floor.

State: [cx, cy, width, height, z_base, theta] (6 parameters)
- cx, cy: center position in XY plane (floor projection)
- width: horizontal extent of the screen (typically 0.8-1.5m)
- height: vertical extent of the screen (typically 0.5-0.9m)
- z_base: height from floor to bottom of TV (wall mount height)
- theta: rotation angle around Z axis

Note: depth is fixed at 5cm (TV_FIXED_DEPTH) - not a free parameter.

Geometric constraints (priors):
- TVs are thin: width >> depth (aspect ratio typically 10:1 to 20:1)
- TVs are wider than tall: width > height (2:1 aspect ratio)
- TVs are mounted at a certain height from the floor (typically 0.8-1.2m)
- TVs are usually aligned with walls (0° or 90°)
"""

import torch
import numpy as np

from src.objects.sdf_constants import SDF_SMOOTH_K, SDF_INSIDE_SCALE

# TV parameters
TV_PARAM_COUNT = 6
TV_PARAM_NAMES = ['cx', 'cy', 'width', 'height', 'z_base', 'theta']

# TV-specific constants
TV_MIN_WIDTH = 0.4          # Minimum screen width (40cm)
TV_MAX_WIDTH = 1.2          # Maximum screen width (1.2m)
TV_MIN_HEIGHT = 0.2         # Minimum screen height
TV_MAX_HEIGHT = 0.7         # Maximum screen height
TV_FIXED_DEPTH = 0.05       # Fixed TV thickness (5cm)
TV_MIN_Z_BASE = 0.5         # Minimum mount height from floor
TV_MAX_Z_BASE = 1.5         # Maximum mount height from floor
TV_TYPICAL_ASPECT = 2.0     # Target TV aspect ratio (width/height)


def compute_tv_sdf(points_xyz: torch.Tensor, tv_params: torch.Tensor) -> torch.Tensor:
    """
    Compute Signed Distance Function for a TV (thin rectangular panel).

    The TV is mounted on a wall at height z_base from the floor.
    The screen extends from z_base to z_base + height.
    Depth is fixed at TV_FIXED_DEPTH (5cm).

    Args:
        points_xyz: [N, 3] points (x, y, z)
        tv_params: [6] tensor [cx, cy, width, height, z_base, theta]

    Returns:
        [N] SDF values (positive=outside, negative=inside, zero=surface)
    """
    cx, cy = tv_params[0], tv_params[1]
    width, height = tv_params[2], tv_params[3]
    z_base = tv_params[4]  # Height from floor to bottom of TV
    theta = tv_params[5]

    # Fixed depth (not optimized)
    depth = TV_FIXED_DEPTH

    # Transform to local TV frame
    cos_t = torch.cos(-theta)
    sin_t = torch.sin(-theta)

    px = points_xyz[:, 0] - cx
    py = points_xyz[:, 1] - cy

    local_x = px * cos_t - py * sin_t  # Width direction
    local_y = px * sin_t + py * cos_t  # Depth direction (thin)

    # Half dimensions
    half_w = width / 2      # X direction (wide)
    half_d = depth / 2      # Y direction (thin)
    half_h = height / 2     # Z direction

    # Distance to faces (negative = inside, positive = outside)
    dx = torch.abs(local_x) - half_w
    dy = torch.abs(local_y) - half_d

    # Z: TV mounted at z_base, center at z_base + half_h
    z_center = z_base + half_h
    local_z = points_xyz[:, 2] - z_center
    dz = torch.abs(local_z) - half_h

    # Outside: standard Euclidean distance
    outside = torch.linalg.norm(torch.stack([
        torch.clamp(dx, min=0),
        torch.clamp(dy, min=0),
        torch.clamp(dz, min=0)
    ], dim=-1), dim=-1)

    # Inside: use smooth approximation
    k = SDF_SMOOTH_K
    inside_scale = SDF_INSIDE_SCALE

    is_inside = (dx < 0) & (dy < 0) & (dz < 0)

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


def compute_tv_priors(params: torch.Tensor, config, robot_pose: np.ndarray = None) -> torch.Tensor:
    """
    Compute prior energy terms for TV parameters.

    TV-specific priors:
    1. Angle alignment: TVs align with walls (0°, 90°, 180°, -90°)
    2. Screen aspect: width/height close to 2:1

    Args:
        params: [6] tensor [cx, cy, width, height, z_base, theta]
        config: TVBeliefConfig with prior parameters
        robot_pose: [x, y, theta] robot pose

    Returns:
        Total prior energy (scalar tensor)
    """
    width, height = params[2], params[3]
    z_base = params[4]
    theta = params[5]

    total_prior = torch.tensor(0.0, dtype=params.dtype, device=params.device)

    # Transform angle to room frame if robot_pose provided
    if robot_pose is not None:
        theta_room = theta + robot_pose[2]
    else:
        theta_room = theta

    # Normalize to [-π, π]
    theta_room = torch.atan2(torch.sin(theta_room), torch.cos(theta_room))

    # =================================================================
    # 1. ANGLE ALIGNMENT PRIOR: TVs align with walls
    # =================================================================
    sigma_angle = getattr(config, 'angle_alignment_sigma', 0.1)
    weight_angle = getattr(config, 'angle_alignment_weight', 1.0)
    precision_angle = 1.0 / (sigma_angle ** 2)

    # Distance to nearest aligned angle (0, ±π/2, π)
    dist_to_0 = theta_room ** 2
    dist_to_pos90 = (theta_room - np.pi/2) ** 2
    dist_to_neg90 = (theta_room + np.pi/2) ** 2
    dist_to_180 = (torch.abs(theta_room) - np.pi) ** 2

    min_angle_dist = torch.minimum(
        torch.minimum(dist_to_0, dist_to_180),
        torch.minimum(dist_to_pos90, dist_to_neg90)
    )
    total_prior = total_prior + weight_angle * 0.5 * precision_angle * min_angle_dist

    # =================================================================
    # 2. SCREEN ASPECT RATIO PRIOR: width/height close to 2:1
    # =================================================================
    weight_aspect = getattr(config, 'screen_aspect_weight', 0.5)
    target_aspect = getattr(config, 'target_screen_aspect', 2.0)
    aspect_tolerance = getattr(config, 'screen_aspect_tolerance', 0.3)

    current_aspect = width / (height + 0.01)
    aspect_error = (current_aspect - target_aspect) / target_aspect
    if torch.abs(aspect_error) > aspect_tolerance:
        total_prior = total_prior + weight_aspect * (aspect_error ** 2)


    return total_prior

