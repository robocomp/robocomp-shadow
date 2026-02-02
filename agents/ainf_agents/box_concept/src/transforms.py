#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Transforms - Coordinate frame transformations for robotics.

Provides transformations between:
- Robot frame: origin at robot, X+ right, Y+ forward
- Room/World frame: fixed origin, typically at room center

Also handles covariance propagation through transformations using Jacobians.
"""

import numpy as np
import torch
from typing import Tuple

from src.belief_core import DTYPE


def transform_points_robot_to_room(points: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
    """
    Transform points from robot frame to room frame.

    T(pose) p = R(θ) p + [x, y]ᵀ

    Args:
        points: [N, 2] or [N, 3] points in robot frame
        robot_pose: [x, y, theta] robot pose in room frame

    Returns:
        Points in room frame (same shape as input)
    """
    x, y, theta = robot_pose
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Rotate and translate XY
    world_x = points[:, 0] * cos_t - points[:, 1] * sin_t + x
    world_y = points[:, 0] * sin_t + points[:, 1] * cos_t + y

    # Preserve Z if 3D points
    if points.shape[1] == 3:
        return np.column_stack([world_x, world_y, points[:, 2]])
    else:
        return np.column_stack([world_x, world_y])


def transform_points_room_to_robot(points: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
    """
    Transform points from room frame to robot frame.

    Inverse of transform_points_robot_to_room:
    p_robot = R(-θ) (p_room - [x, y]ᵀ)

    Args:
        points: [N, 2] or [N, 3] points in room frame
        robot_pose: [x, y, theta] robot pose in room frame

    Returns:
        Points in robot frame (same shape as input)
    """
    x, y, theta = robot_pose
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)

    # Translate then rotate
    px = points[:, 0] - x
    py = points[:, 1] - y

    robot_x = px * cos_t - py * sin_t
    robot_y = px * sin_t + py * cos_t

    if points.shape[1] == 3:
        return np.column_stack([robot_x, robot_y, points[:, 2]])
    else:
        return np.column_stack([robot_x, robot_y])


def transform_points_with_covariance(points: np.ndarray,
                                      robot_pose: np.ndarray,
                                      robot_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform points from robot to room frame with covariance propagation.

    Uses Jacobian-based linearization for uncertainty propagation:
    Σ_world = J @ Σ_robot @ Jᵀ

    where J = [∂p_world/∂(x, y, θ)]

    Args:
        points: [N, 3] points in robot frame
        robot_pose: [x, y, theta] robot pose in room frame
        robot_cov: [3, 3] robot pose covariance

    Returns:
        (world_points [N, 3], point_covariances [N, 2, 2])
    """
    x, y, theta = robot_pose
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    n_points = len(points)

    # Transform points (mean)
    world_x = points[:, 0] * cos_t - points[:, 1] * sin_t + x
    world_y = points[:, 0] * sin_t + points[:, 1] * cos_t + y
    world_z = points[:, 2]
    world_points = np.column_stack([world_x, world_y, world_z])

    # Compute covariance for each point
    point_covs = np.zeros((n_points, 2, 2))

    for i in range(n_points):
        px, py = points[i, 0], points[i, 1]

        # Jacobian: ∂p_world/∂(x, y, θ)
        J = np.array([
            [1.0, 0.0, -sin_t * px - cos_t * py],
            [0.0, 1.0,  cos_t * px - sin_t * py]
        ])

        point_covs[i] = J @ robot_cov @ J.T

    return world_points, point_covs


def transform_box_to_robot_frame(box_mu: torch.Tensor,
                                  robot_pose: np.ndarray) -> torch.Tensor:
    """
    Transform box parameters from room frame to robot frame.

    box_mu = [cx, cy, w, h, d, theta] in room frame

    Args:
        box_mu: Box mean in room frame [6]
        robot_pose: [x, y, theta] robot pose in room frame

    Returns:
        Box mean in robot frame [6]
    """
    rx, ry, rtheta = robot_pose
    cos_t = np.cos(-rtheta)
    sin_t = np.sin(-rtheta)

    # Box center in room frame
    cx_room = box_mu[0]
    cy_room = box_mu[1]

    # Translate then rotate (inverse transform)
    dx = cx_room - rx
    dy = cy_room - ry

    cx_robot = dx * cos_t - dy * sin_t
    cy_robot = dx * sin_t + dy * cos_t

    # Dimensions stay the same
    w, h, d = box_mu[2], box_mu[3], box_mu[4]

    # Angle in robot frame
    theta_robot = box_mu[5] - rtheta

    return torch.tensor([cx_robot, cy_robot, w, h, d, theta_robot],
                       dtype=DTYPE, device=box_mu.device)


def transform_box_to_room_frame(box_mu_robot: torch.Tensor,
                                 robot_pose: np.ndarray) -> torch.Tensor:
    """
    Transform box parameters from robot frame to room frame.

    Args:
        box_mu_robot: Box mean in robot frame [6]
        robot_pose: [x, y, theta] robot pose in room frame

    Returns:
        Box mean in room frame [6]
    """
    rx, ry, rtheta = robot_pose
    cos_t = np.cos(rtheta)
    sin_t = np.sin(rtheta)

    cx_robot = box_mu_robot[0]
    cy_robot = box_mu_robot[1]

    # Rotate and translate to room frame
    cx_room = cos_t * cx_robot - sin_t * cy_robot + rx
    cy_room = sin_t * cx_robot + cos_t * cy_robot + ry

    # Dimensions stay the same
    w, h, d = box_mu_robot[2], box_mu_robot[3], box_mu_robot[4]

    # Angle in room frame
    theta_room = box_mu_robot[5] + rtheta

    return torch.tensor([cx_room, cy_room, w, h, d, theta_room],
                       dtype=DTYPE, device=box_mu_robot.device)


def transform_box_with_covariance(box_mu: torch.Tensor,
                                   box_Sigma: torch.Tensor,
                                   robot_pose: np.ndarray,
                                   robot_cov: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform box from room frame to robot frame with covariance composition.

    Composes box uncertainty with robot pose uncertainty via Jacobians:
    Σ_robot = J_box @ Σ_box @ J_box.T + J_robot @ Σ_robot_pose @ J_robot.T

    Args:
        box_mu: Box mean [6] in room frame
        box_Sigma: Box covariance [6, 6] in room frame
        robot_pose: [x, y, theta] robot pose in room frame
        robot_cov: [3, 3] robot pose covariance

    Returns:
        (box_mu_robot, box_Sigma_robot) in robot frame
    """
    rx, ry, rtheta = robot_pose
    cos_t = np.cos(-rtheta)
    sin_t = np.sin(-rtheta)

    # Transform box mean
    box_mu_robot = transform_box_to_robot_frame(box_mu, robot_pose)

    # Jacobian of box_robot w.r.t. box_room (6x6)
    J_box = np.zeros((6, 6))
    J_box[0, 0] = cos_t   # ∂cx_r/∂cx
    J_box[0, 1] = -sin_t  # ∂cx_r/∂cy
    J_box[1, 0] = sin_t   # ∂cy_r/∂cx
    J_box[1, 1] = cos_t   # ∂cy_r/∂cy
    J_box[2, 2] = 1.0     # ∂w_r/∂w
    J_box[3, 3] = 1.0     # ∂h_r/∂h
    J_box[4, 4] = 1.0     # ∂d_r/∂d
    J_box[5, 5] = 1.0     # ∂θ_r/∂θ

    # Jacobian of box_robot w.r.t. robot_pose (6x3)
    cx_room = box_mu[0].item()
    cy_room = box_mu[1].item()
    dx = cx_room - rx
    dy = cy_room - ry

    cx_r = cos_t * dx - sin_t * dy
    cy_r = sin_t * dx + cos_t * dy

    J_robot = np.zeros((6, 3))
    J_robot[0, 0] = -cos_t      # ∂cx_r/∂rx
    J_robot[0, 1] = sin_t       # ∂cx_r/∂ry
    J_robot[0, 2] = cy_r        # ∂cx_r/∂rθ
    J_robot[1, 0] = -sin_t      # ∂cy_r/∂rx
    J_robot[1, 1] = -cos_t      # ∂cy_r/∂ry
    J_robot[1, 2] = -cx_r       # ∂cy_r/∂rθ
    J_robot[5, 2] = -1.0        # ∂θ_r/∂rθ

    # Convert to tensors
    device = box_mu.device
    J_box_t = torch.tensor(J_box, dtype=DTYPE, device=device)
    J_robot_t = torch.tensor(J_robot, dtype=DTYPE, device=device)
    robot_cov_t = torch.tensor(robot_cov, dtype=DTYPE, device=device)

    # Compose covariances
    cov_from_box = J_box_t @ box_Sigma @ J_box_t.T
    cov_from_robot = J_robot_t @ robot_cov_t @ J_robot_t.T

    box_Sigma_robot = cov_from_box + cov_from_robot

    return box_mu_robot, box_Sigma_robot


def compute_jacobian_room_to_robot(point_room: np.ndarray,
                                    robot_pose: np.ndarray) -> np.ndarray:
    """
    Compute Jacobian of room-to-robot point transformation.

    p_robot = R(-θ) (p_room - [rx, ry])

    Args:
        point_room: [x, y] or [x, y, z] point in room frame
        robot_pose: [rx, ry, rtheta] robot pose

    Returns:
        [2, 3] Jacobian matrix
    """
    rx, ry, rtheta = robot_pose
    cos_t = np.cos(-rtheta)
    sin_t = np.sin(-rtheta)

    p_x, p_y = point_room[0], point_room[1]

    J = np.array([
        [-cos_t, -sin_t, -(p_y - ry) * cos_t + (p_x - rx) * sin_t],
        [sin_t, -cos_t, -(p_y - ry) * sin_t - (p_x - rx) * cos_t]
    ])

    return J
