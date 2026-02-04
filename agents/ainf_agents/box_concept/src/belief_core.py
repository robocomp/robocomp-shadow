#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Belief Core - Base classes for Active Inference belief management.

This module provides abstract base classes for beliefs and configurations
that can be extended for different object types (boxes, tables, chairs, etc.).

Mathematical framework (Active Inference):
F(s) = (1/2σ²_o) Σᵢ dᵢ(s)² + (1/2)(s-μₚ)ᵀΣₚ⁻¹(s-μₚ)

where:
- dᵢ(s) = SDF(pᵢ, s) is the signed distance function for the object
- First term: prediction error (negative accuracy / likelihood)
- Second term: complexity (prior regularizer / KL divergence)
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from dataclasses import dataclass, field

# Device and dtype configuration (shared across all beliefs)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32


@dataclass
class BeliefConfig:
    """
    Base configuration for object beliefs.

    Subclasses should extend this with object-specific parameters
    (e.g., prior dimensions for boxes vs tables).
    """
    # Observation model
    sigma_obs: float = 0.05  # σ_o: observation noise (meters)

    # Process model (how much belief can change per frame)
    sigma_process_xy: float = 0.02  # Process noise for position
    sigma_process_size: float = 0.01  # Process noise for dimensions
    sigma_process_angle: float = 0.05  # Process noise for angle

    # Initialization uncertainty
    initial_position_std: float = 0.1
    initial_size_std: float = 0.1
    initial_angle_std: float = 0.2

    # Size constraints
    min_size: float = 0.10  # Minimum dimension (meters)
    max_size: float = 2.0   # Maximum dimension (meters)

    # Lifecycle parameters
    confidence_decay: float = 0.85  # γ: decay factor when not observed
    confidence_boost: float = 0.15  # Δκ: increase on observation
    confidence_threshold: float = 0.25  # Below this, remove belief
    confirmed_threshold: float = 0.70  # Above this, belief is confirmed
    initial_confidence: float = 0.30


class Belief(ABC):
    """
    Abstract base class for object beliefs.

    Each belief represents a probabilistic estimate of an object's state:
    - μ (mu): mean of the Gaussian belief
    - Σ (Sigma): covariance matrix
    - κ (confidence): confidence score ∈ [0, 1]

    Subclasses must implement:
    - sdf(): Signed Distance Function for the object geometry
    - state_dim: dimension of the state vector
    - from_cluster(): factory method to create belief from point cluster
    """

    def __init__(self,
                 belief_id: int,
                 mu: torch.Tensor,
                 Sigma: torch.Tensor,
                 config: BeliefConfig,
                 confidence: float = 0.5):
        """
        Initialize a belief.

        Args:
            belief_id: Unique identifier
            mu: Mean of the belief (state vector)
            Sigma: Covariance matrix
            config: Configuration parameters
            confidence: Initial confidence score
        """
        self.id = belief_id
        self.mu = mu
        self.Sigma = Sigma
        self.config = config
        self.confidence = confidence

        # Lifecycle tracking
        self.age = 0
        self.last_seen = 0
        self.observation_count = 0
        self.is_confirmed = False
        self.last_sdf_mean = 0.0

        # Historical points storage (for evidence accumulation)
        self._init_historical_storage()

    def _init_historical_storage(self):
        """Initialize storage for historical points."""
        device = self.mu.device
        self.historical_points = torch.empty((0, 3), dtype=DTYPE, device=device)
        self.historical_capture_covs = torch.empty((0, 2, 2), dtype=DTYPE, device=device)
        self.historical_rfe = torch.empty(0, dtype=DTYPE, device=device)

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state vector."""
        pass

    @property
    @abstractmethod
    def position(self) -> Tuple[float, float]:
        """Return (x, y) position of the object center."""
        pass

    @property
    @abstractmethod
    def angle(self) -> float:
        """Return orientation angle in radians."""
        pass

    @abstractmethod
    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute Signed Distance Function for this object.

        Args:
            points: [N, 3] tensor of 3D points

        Returns:
            [N] tensor of SDF values (positive=outside, negative=inside)
        """
        pass

    @classmethod
    @abstractmethod
    def from_cluster(cls,
                     belief_id: int,
                     cluster: np.ndarray,
                     config: BeliefConfig,
                     device: torch.device) -> Optional['Belief']:
        """
        Factory method to create a belief from a point cluster.

        Args:
            belief_id: Unique identifier for the new belief
            cluster: [N, 3] numpy array of points
            config: Configuration parameters
            device: Torch device

        Returns:
            New Belief instance, or None if cluster is invalid
        """
        pass

    def propagate_prior(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate belief through time (prediction step).

        Returns:
            (prior_mu, prior_Sigma) with added process noise
        """
        prior_mu = self.mu.clone()

        # Build process noise diagonal based on state structure
        # Subclasses may override for different state vectors
        process_vars = self._get_process_noise_variances()
        process_noise = torch.diag(torch.tensor(process_vars, dtype=DTYPE, device=self.mu.device))

        prior_Sigma = self.Sigma + process_noise

        return prior_mu, prior_Sigma

    def compute_prior_term(self, mu: torch.Tensor, robot_pose: np.ndarray = None) -> torch.Tensor:
        """
        Compute the prior energy term for this belief.

        This method should be overridden by subclasses to add model-specific priors
        (e.g., angle alignment for indoor objects).

        Default implementation returns 0 (no additional prior).

        Args:
            mu: Current state estimate (in robot frame)
            robot_pose: Robot pose [x, y, theta] for frame transformation

        Returns:
            Prior energy term (scalar tensor)
        """
        return torch.tensor(0.0, dtype=DTYPE, device=mu.device)

    @abstractmethod
    def _get_process_noise_variances(self) -> list:
        """Return list of process noise variances for each state dimension."""
        pass

    def update_lifecycle(self, frame_count: int, was_observed: bool):
        """
        Update belief lifecycle state.

        Args:
            frame_count: Current frame number
            was_observed: Whether belief was matched to observations this frame
        """
        self.age += 1

        if was_observed:
            self.last_seen = frame_count
            self.observation_count += 1
            self.confidence = min(1.0, self.confidence + self.config.confidence_boost)

            if self.confidence >= self.config.confirmed_threshold:
                self.is_confirmed = True
        else:
            # Apply decay (confirmed beliefs decay slower)
            if not self.is_confirmed or (frame_count - self.last_seen) > 30:
                self.confidence *= self.config.confidence_decay

    def should_remove(self) -> bool:
        """Check if belief should be removed due to low confidence."""
        return self.confidence < self.config.confidence_threshold

    @property
    def num_historical_points(self) -> int:
        """Number of stored historical points."""
        return len(self.historical_points)

    def to_dict(self) -> dict:
        """Convert belief to dictionary for visualization/serialization."""
        return {
            'id': self.id,
            'confidence': self.confidence,
            'is_confirmed': self.is_confirmed,
            'sdf_mean': self.last_sdf_mean,
            'num_hist_points': self.num_historical_points,
            'age': self.age,
            'observation_count': self.observation_count
        }
