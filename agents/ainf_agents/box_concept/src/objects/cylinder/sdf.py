#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Cylinder SDF and Prior Functions

State: [cx, cy, r, h] (4 parameters)
- cx, cy: center position
- r: radius
- h: height (vertical extent)
"""

import torch
import numpy as np

# Cylinder parameters
CYLINDER_PARAM_COUNT = 4
CYLINDER_PARAM_NAMES = ['cx', 'cy', 'r', 'h']


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
