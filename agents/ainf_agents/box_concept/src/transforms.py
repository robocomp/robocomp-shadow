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
    Transform object from room frame to robot frame with covariance composition.

    Composes object uncertainty with robot pose uncertainty via Jacobians:
    Σ_robot = J_obj @ Σ_obj @ J_obj.T + J_robot @ Σ_robot_pose @ J_robot.T

    Works for any state dimension where:
    - First 2 elements are position (cx, cy)
    - Last element is angle (theta)
    - Middle elements are size/shape parameters (unchanged by transform)

    Args:
        box_mu: Object mean [N] in room frame (N = 6 for box, 7 for table, etc.)
        box_Sigma: Object covariance [N, N] in room frame
        robot_pose: [x, y, theta] robot pose in room frame
        robot_cov: [3, 3] robot pose covariance

    Returns:
        (obj_mu_robot, obj_Sigma_robot) in robot frame
    """
    state_dim = len(box_mu)
    rx, ry, rtheta = robot_pose
    cos_t = np.cos(-rtheta)
    sin_t = np.sin(-rtheta)

    # Transform object mean
    box_mu_robot = transform_object_to_robot_frame(box_mu, robot_pose)

    # Jacobian of obj_robot w.r.t. obj_room (NxN)
    # Position transforms, sizes stay same, angle transforms
    J_obj = np.eye(state_dim)
    J_obj[0, 0] = cos_t   # ∂cx_r/∂cx
    J_obj[0, 1] = -sin_t  # ∂cx_r/∂cy
    J_obj[1, 0] = sin_t   # ∂cy_r/∂cx
    J_obj[1, 1] = cos_t   # ∂cy_r/∂cy
    # Size parameters (indices 2 to state_dim-2) already have 1.0 from eye()
    # Angle (last element) already has 1.0 from eye()

    # Jacobian of obj_robot w.r.t. robot_pose (Nx3)
    cx_room = box_mu[0].item()
    cy_room = box_mu[1].item()
    dx = cx_room - rx
    dy = cy_room - ry

    cx_r = cos_t * dx - sin_t * dy
    cy_r = sin_t * dx + cos_t * dy

    J_robot = np.zeros((state_dim, 3))
    J_robot[0, 0] = -cos_t      # ∂cx_r/∂rx
    J_robot[0, 1] = sin_t       # ∂cx_r/∂ry
    J_robot[0, 2] = cy_r        # ∂cx_r/∂rθ
    J_robot[1, 0] = -sin_t      # ∂cy_r/∂rx
    J_robot[1, 1] = -cos_t      # ∂cy_r/∂ry
    J_robot[1, 2] = -cx_r       # ∂cy_r/∂rθ
    J_robot[state_dim-1, 2] = -1.0  # ∂θ_r/∂rθ (angle is last element)

    # Convert to tensors
    device = box_mu.device
    J_obj_t = torch.tensor(J_obj, dtype=DTYPE, device=device)
    J_robot_t = torch.tensor(J_robot, dtype=DTYPE, device=device)
    robot_cov_t = torch.tensor(robot_cov, dtype=DTYPE, device=device)

    # Compose covariances
    cov_from_obj = J_obj_t @ box_Sigma @ J_obj_t.T
    cov_from_robot = J_robot_t @ robot_cov_t @ J_robot_t.T

    box_Sigma_robot = cov_from_obj + cov_from_robot

    return box_mu_robot, box_Sigma_robot


def transform_object_to_robot_frame(obj_mu: torch.Tensor,
                                     robot_pose: np.ndarray) -> torch.Tensor:
    """
    Transform object parameters from room frame to robot frame.

    Works for any state where:
    - First 2 elements are position (cx, cy)
    - Last element is angle (theta)
    - Middle elements are unchanged (sizes)

    Args:
        obj_mu: Object mean in room frame [N]
        robot_pose: [x, y, theta] robot pose in room frame

    Returns:
        Object mean in robot frame [N]
    """
    rx, ry, rtheta = robot_pose
    cos_t = np.cos(-rtheta)
    sin_t = np.sin(-rtheta)

    # Position transform
    cx_room = obj_mu[0]
    cy_room = obj_mu[1]
    dx = cx_room - rx
    dy = cy_room - ry
    cx_robot = dx * cos_t - dy * sin_t
    cy_robot = dx * sin_t + dy * cos_t

    # Build output: position + middle params + angle
    result = obj_mu.clone()
    result[0] = cx_robot
    result[1] = cy_robot
    result[-1] = obj_mu[-1] - rtheta  # Angle transform

    return result


def transform_object_to_room_frame(obj_mu_robot: torch.Tensor,
                                    robot_pose: np.ndarray) -> torch.Tensor:
    """
    Transform object parameters from robot frame to room frame.

    Works for any state where:
    - First 2 elements are position (cx, cy)
    - Last element is angle (theta)
    - Middle elements are unchanged (sizes)

    Args:
        obj_mu_robot: Object mean in robot frame [N]
        robot_pose: [x, y, theta] robot pose in room frame

    Returns:
        Object mean in room frame [N]
    """
    rx, ry, rtheta = robot_pose
    cos_t = np.cos(rtheta)
    sin_t = np.sin(rtheta)

    cx_robot = obj_mu_robot[0]
    cy_robot = obj_mu_robot[1]

    # Rotate and translate to room frame
    cx_room = cos_t * cx_robot - sin_t * cy_robot + rx
    cy_room = sin_t * cx_robot + cos_t * cy_robot + ry

    # Build output
    result = obj_mu_robot.clone()
    result[0] = cx_room
    result[1] = cy_room
    result[-1] = obj_mu_robot[-1] + rtheta  # Angle transform

    return result


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
