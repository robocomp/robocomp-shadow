#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Shared Priors - Inter-object and temporal smoothness constraints.

These priors apply to all object types and enforce:
1. Non-overlap: Objects cannot occupy the same space
2. Temporal smoothness: Object parameters should change smoothly over time
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.belief_core import Belief


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SharedPriorConfig:
    """Configuration for shared priors (inter-object and temporal)."""

    # Non-overlap prior: prevent objects from occupying the same space
    lambda_overlap: float = 5.0          # Weight for non-overlap prior
    min_object_distance: float = 0.3     # Minimum distance between object centers (meters)

    # Temporal smoothness prior: prevent sudden parameter changes
    lambda_smoothness: float = 2.0       # Weight for temporal smoothness prior
    max_position_change: float = 0.1     # Max position change per frame (meters)
    max_size_change: float = 0.05        # Max size change per frame (meters)
    max_angle_change: float = 0.1        # Max angle change per frame (radians, ~5.7°)


# Default configuration instance
DEFAULT_SHARED_PRIOR_CONFIG = SharedPriorConfig()


# =============================================================================
# PRIOR FUNCTIONS
# =============================================================================

def compute_overlap_prior(
    mu: torch.Tensor,
    current_belief_id: int,
    beliefs: Dict[int, Any],
    robot_pose: np.ndarray,
    robot_cov: np.ndarray,
    config: SharedPriorConfig = DEFAULT_SHARED_PRIOR_CONFIG
) -> torch.Tensor:
    """
    Compute non-overlap prior: penalize when objects are too close to each other.

    Uses a soft repulsion force that increases as objects get closer than min_object_distance.

    Args:
        mu: Current state estimate [cx, cy, ...] in robot frame
        current_belief_id: ID of the belief being optimized (to exclude from distance calc)
        beliefs: Dict of all active beliefs {id: MultiModelBelief}
        robot_pose: Robot pose [x, y, theta]
        robot_cov: Robot pose covariance [3, 3]
        config: Prior configuration

    Returns:
        Prior energy term (scalar tensor)
    """
    from src.transforms import transform_box_with_covariance

    overlap_energy = torch.tensor(0.0, dtype=mu.dtype, device=mu.device)

    if len(beliefs) <= 1:
        return overlap_energy

    # Get position of current object
    cx, cy = mu[0], mu[1]

    # Compare with all other beliefs
    for bid, other_mb in beliefs.items():
        if bid == current_belief_id:
            continue

        # Get other object's position (in room frame, need to transform to robot frame)
        other_belief = other_mb.get_active_belief()
        if other_belief is None:
            continue

        # Transform other belief to robot frame
        other_mu_robot, _ = transform_box_with_covariance(
            other_belief.mu, other_belief.Sigma, robot_pose, robot_cov)

        other_cx, other_cy = other_mu_robot[0], other_mu_robot[1]

        # Compute distance between centers
        dist = torch.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)

        # Soft repulsion: quadratic penalty when closer than min_distance
        # Energy = lambda * max(0, min_dist - dist)^2
        if dist < config.min_object_distance:
            violation = config.min_object_distance - dist
            overlap_energy = overlap_energy + config.lambda_overlap * violation**2

    return overlap_energy


def compute_smoothness_prior(
    mu: torch.Tensor,
    belief: 'Belief',
    robot_pose: np.ndarray,
    robot_cov: np.ndarray,
    config: SharedPriorConfig = DEFAULT_SHARED_PRIOR_CONFIG
) -> torch.Tensor:
    """
    Compute temporal smoothness prior: penalize large changes from previous state.

    This prevents parameters from changing too rapidly between frames,
    e.g., width doubling in one iteration.

    Args:
        mu: Current state estimate in robot frame
        belief: The belief object (contains previous mu in room frame)
        robot_pose: Robot pose [x, y, theta]
        robot_cov: Robot pose covariance [3, 3]
        config: Prior configuration

    Returns:
        Prior energy term (scalar tensor)
    """
    from src.transforms import transform_box_with_covariance

    smoothness_energy = torch.tensor(0.0, dtype=mu.dtype, device=mu.device)

    if belief.mu is None:
        return smoothness_energy

    # Transform previous state to robot frame for comparison
    prev_mu_robot, _ = transform_box_with_covariance(
        belief.mu, belief.Sigma, robot_pose, robot_cov)

    # Position smoothness: penalize large position changes
    pos_diff = torch.sqrt((mu[0] - prev_mu_robot[0])**2 + (mu[1] - prev_mu_robot[1])**2)
    if pos_diff > config.max_position_change:
        pos_violation = pos_diff - config.max_position_change
        smoothness_energy = smoothness_energy + config.lambda_smoothness * pos_violation**2

    # Size smoothness: penalize large size changes (for each size parameter)
    state_dim = len(mu)
    for i in range(2, state_dim - 1):  # Skip position (0,1) and angle (last)
        size_diff = torch.abs(mu[i] - prev_mu_robot[i])
        if size_diff > config.max_size_change:
            size_violation = size_diff - config.max_size_change
            smoothness_energy = smoothness_energy + config.lambda_smoothness * size_violation**2

    # Angle smoothness: penalize large angle changes
    # Handle angle wrapping
    angle_diff = mu[-1] - prev_mu_robot[-1]
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))  # Normalize to [-pi, pi]
    angle_diff_abs = torch.abs(angle_diff)

    if angle_diff_abs > config.max_angle_change:
        angle_violation = angle_diff_abs - config.max_angle_change
        smoothness_energy = smoothness_energy + config.lambda_smoothness * angle_violation**2

    return smoothness_energy


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SharedPriorConfig',
    'DEFAULT_SHARED_PRIOR_CONFIG',
    'compute_overlap_prior',
    'compute_smoothness_prior',
]

