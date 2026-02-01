#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Box Manager - Active Inference-based Obstacle Belief Management

This module implements rectangular obstacle beliefs following the Active Inference
framework as described in the paper "Using Active Inference for Perception, Planning
and Control".

Key concepts from the paper:
- Variational Free Energy: F = prediction_error + complexity
- SDF-based likelihood model
- Gaussian beliefs with mean and covariance
- Belief lifecycle: initialization, update, decay, removal
- Data association via cost matrix minimization

Mathematical framework:
F(s) = (1/2σ²_o) Σᵢ dᵢ(s)² + (1/2)(s-μₚ)ᵀΣₚ⁻¹(s-μₚ)

where:
- dᵢ(s) = SDF_rect(pᵢ, s) is the signed distance function
- First term: prediction error (negative accuracy)
- Second term: complexity (prior regulariser / KL divergence)
"""

import sys
import numpy as np
import torch
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from rich.console import Console
from scipy.optimize import linear_sum_assignment

sys.path.append('/opt/robocomp/lib')
from pydsr import Node, Attribute

console = Console(highlight=False)

# Device and dtype configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32


@dataclass
class RectangleBeliefConfig:
    """Configuration for 3D box obstacle beliefs.

    Parameters follow the paper's notation:
    - σ_o: observation noise standard deviation
    - σ_process: process noise for belief propagation

    State vector s = (cx, cy, w, h, d, θ) where:
    - (cx, cy): center position in room frame (meters)
    - (w, h, d): width, height, depth (meters)
    - θ: orientation angle (radians)
    """
    # Observation model
    sigma_obs: float = 0.05  # σ_o: observation noise (meters)

    # Prior/Process model - controls how much the belief can change per frame
    sigma_process_xy: float = 0.02  # Process noise for position (meters) - small because boxes are static
    sigma_process_size: float = 0.01  # Process noise for dimensions (w, h, d)
    sigma_process_angle: float = 0.05  # Process noise for angle

    # Prior for box dimensions (mean = 0.5m for typical box)
    prior_size_mean: float = 0.5  # μ_size: prior mean for w, h, d (meters)
    prior_size_std: float = 0.2  # σ_size: prior std for dimensions - tighter prior

    # Initialization - start with moderate uncertainty
    initial_position_std: float = 0.1  # Initial uncertainty in position - tighter
    initial_size_std: float = 0.1  # Initial uncertainty in dimensions
    initial_angle_std: float = 0.2  # Initial uncertainty in angle

    # Size constraints - boxes must be reasonably sized
    min_size: float = 0.20  # Minimum box dimension (meters)
    max_size: float = 1.2  # Maximum box dimension (meters)
    min_aspect_ratio: float = 0.35  # Minimum w/h ratio to avoid thin lines

    # Lifecycle
    confidence_decay: float = 0.85  # γ: decay factor - faster decay
    confidence_boost: float = 0.15  # Δκ: confidence increase on match
    confidence_threshold: float = 0.25  # Removal threshold - higher
    confirmed_threshold: float = 0.70  # Threshold to become confirmed
    initial_confidence: float = 0.30  # Lower initial confidence for new beliefs

    # Clustering - larger clusters to avoid splitting single objects
    cluster_eps: float = 0.25  # DBSCAN-like epsilon (meters) - increased for bigger clusters
    min_cluster_points: int = 12  # Minimum points per cluster - increased

    # Association - more permissive to allow tracking
    max_association_cost: float = 5.0  # τ_reject: maximum cost for matching - increased significantly
    max_association_distance: float = 0.8  # Maximum center distance for association (meters) - increased

    # Wall filtering
    wall_margin: float = 0.30  # Distance threshold from walls (meters)


@dataclass
class RectangleBelief:
    """
    Gaussian belief over 3D box obstacle parameters.

    State vector s = (cx, cy, w, h, d, θ) where:
    - (cx, cy): center position in room frame (meters)
    - (w, h, d): width, height, depth (meters) - 3D dimensions
    - θ: orientation angle (radians)

    Maintains:
    - μ: mean of the belief (6D tensor)
    - Σ: covariance matrix (6x6 tensor)
    - κ: confidence score ∈ [0,1]

    Prior for dimensions centered at 0.5m (typical box size)
    """
    id: int
    mu: torch.Tensor  # Mean [cx, cy, w, h, d, theta] - 6D
    Sigma: torch.Tensor  # Covariance 6x6
    confidence: float = 0.5
    age: int = 0
    last_seen: int = 0
    observation_count: int = 0
    is_confirmed: bool = False
    last_sdf_mean: float = 0.0  # Last computed mean SDF value
    config: RectangleBeliefConfig = field(default_factory=RectangleBeliefConfig)

    @property
    def cx(self) -> float:
        return self.mu[0].item()

    @property
    def cy(self) -> float:
        return self.mu[1].item()

    @property
    def width(self) -> float:
        return self.mu[2].item()

    @property
    def height(self) -> float:
        return self.mu[3].item()

    @property
    def depth(self) -> float:
        return self.mu[4].item()

    @property
    def angle(self) -> float:
        return self.mu[5].item()

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute Signed Distance Function for oriented 3D box.

        SDF(p, s) = distance from point p to box boundary
        - Positive: point outside box
        - Negative: point inside box
        - Zero: point on boundary

        Supports both 2D points [N, 2] and 3D points [N, 3].
        For 2D points, assumes z=0 (ground plane) and uses depth for vertical extent.

        From paper Eq. cluster_likelihood:
        d_i(s) = SDF(p_i, s)
        """
        cx, cy, w, h, d = self.mu[0], self.mu[1], self.mu[2], self.mu[3], self.mu[4]
        theta = self.mu[5]  # θ is at index 5

        # Check if points are 2D or 3D
        is_3d = points.shape[1] == 3

        # Transform points to local box frame (rotation around z-axis)
        cos_t = torch.cos(-theta)
        sin_t = torch.sin(-theta)

        # Translate to box center
        px = points[:, 0] - cx
        py = points[:, 1] - cy

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

        if is_3d:
            # 3D SDF: consider z coordinate and depth
            pz = points[:, 2]
            # Box extends from z=0 to z=d (sitting on ground)
            local_z = pz - half_d  # Center the z-axis
            dz = torch.abs(local_z) - half_d

            # 3D outside distance (Euclidean to nearest point on box surface)
            outside = torch.sqrt(
                torch.clamp(dx, min=0)**2 +
                torch.clamp(dy, min=0)**2 +
                torch.clamp(dz, min=0)**2 + 1e-8
            )

            # 3D inside distance (negative, distance to nearest face)
            inside = torch.max(torch.max(dx, dy), dz)

            # Combine: positive outside, negative inside
            is_outside = (dx > 0) | (dy > 0) | (dz > 0)
        else:
            # 2D SDF: only XY plane (top-down view)
            outside = torch.sqrt(
                torch.clamp(dx, min=0)**2 +
                torch.clamp(dy, min=0)**2 + 1e-8
            )

            # 2D inside distance (negative, distance to nearest edge)
            inside = torch.max(dx, dy)

            # Combine: positive outside, negative inside
            is_outside = (dx > 0) | (dy > 0)

        return torch.where(is_outside, outside, inside)

    def compute_vfe(self, points: torch.Tensor, prior_mu: torch.Tensor = None,
                    prior_Sigma: torch.Tensor = None) -> torch.Tensor:
        """
        Compute Variational Free Energy for this belief given observations.

        From paper Eq. obstacle_objective:
        F(s) = (1/2σ²_o) Σᵢ dᵢ(s)² + (1/2)(s-μₚ)ᵀΣₚ⁻¹(s-μₚ) + F_size_prior

        The size prior term regularizes dimensions toward typical box size (0.5m).

        Args:
            points: LIDAR points [N, 2] tensor
            prior_mu: Prior mean (defaults to current mean)
            prior_Sigma: Prior covariance (defaults to current covariance)

        Returns:
            Scalar VFE value
        """
        if prior_mu is None:
            prior_mu = self.mu
        if prior_Sigma is None:
            prior_Sigma = self.Sigma

        sigma_sq = self.config.sigma_obs ** 2

        # =====================================================================
        # PREDICTION ERROR TERM (negative accuracy)
        # F_pred = (1/2σ²_o) Σᵢ dᵢ(s)²
        # =====================================================================
        sdf_values = self.sdf(points)
        prediction_error = torch.sum(sdf_values ** 2) / (2.0 * sigma_sq)

        # =====================================================================
        # COMPLEXITY TERM (prior regulariser / KL divergence)
        # F_complexity = (1/2)(s-μₚ)ᵀΣₚ⁻¹(s-μₚ)
        # =====================================================================
        delta = self.mu - prior_mu
        # Use pseudo-inverse for numerical stability (6x6 matrix now)
        Sigma_inv = torch.linalg.pinv(prior_Sigma + 1e-6 * torch.eye(6, device=self.mu.device))
        complexity = 0.5 * delta @ Sigma_inv @ delta

        # =====================================================================
        # SIZE PRIOR TERM (regularize dimensions toward typical box size 0.5m)
        # F_size = (1/2σ²_size) Σⱼ (sⱼ - μ_size)² for j in {w, h, d}
        # =====================================================================
        size_prior_mean = self.config.prior_size_mean
        size_prior_var = self.config.prior_size_std ** 2
        w, h, d = self.mu[2], self.mu[3], self.mu[4]
        size_prior = ((w - size_prior_mean)**2 + (h - size_prior_mean)**2 +
                      (d - size_prior_mean)**2) / (2.0 * size_prior_var)

        return prediction_error + complexity + size_prior

    def optimize(self, points: torch.Tensor, prior_mu: torch.Tensor,
                 prior_Sigma: torch.Tensor, point_covs: torch.Tensor = None,
                 num_iters: int = 10, lr: float = 0.05):
        """
        Update belief by minimizing Variational Free Energy via gradient descent.

        From paper Eq. obstacle_objective:
        F(s) = (1/2σ²_total) Σᵢ dᵢ(s)² + (1/2)(s-μₚ)ᵀΣₚ⁻¹(s-μₚ)

        The total observation variance includes:
        - σ²_o: intrinsic LIDAR noise
        - σ²_robot: propagated uncertainty from robot pose (from point_covs)

        Args:
            points: LIDAR cluster points [N, 2]
            prior_mu: Prior mean from previous posterior
            prior_Sigma: Prior covariance from previous posterior + process noise
            point_covs: Point covariances [N, 2, 2] from robot pose uncertainty propagation
            num_iters: Optimization iterations
            lr: Learning rate
        """
        if len(points) < 3:
            return

        # Base observation variance
        sigma_sq_base = self.config.sigma_obs ** 2

        # Compute effective observation variance per point
        # σ²_total = σ²_obs + trace(Σ_point)/2 (average variance from robot pose)
        if point_covs is not None and len(point_covs) > 0:
            # Use trace of point covariance as isotropic approximation
            point_var = torch.mean(torch.diagonal(point_covs, dim1=1, dim2=2).sum(dim=1)) / 2.0
            sigma_sq_effective = sigma_sq_base + point_var.item()
        else:
            sigma_sq_effective = sigma_sq_base

        # Make parameters require gradients
        mu = self.mu.clone().detach().requires_grad_(True)

        for iteration in range(num_iters):
            # Extract parameters: cx, cy, w, h, d, theta
            cx, cy, w, h, d, theta = mu[0], mu[1], mu[2], mu[3], mu[4], mu[5]

            # Transform points to local frame
            cos_t = torch.cos(-theta)
            sin_t = torch.sin(-theta)
            px = points[:, 0] - cx
            py = points[:, 1] - cy
            local_x = px * cos_t - py * sin_t
            local_y = px * sin_t + py * cos_t

            # Half dimensions
            half_w = w / 2
            half_h = h / 2

            # Distance to box faces
            dx = torch.abs(local_x) - half_w
            dy = torch.abs(local_y) - half_h

            # 2D SDF
            outside_mask = (dx > 0) | (dy > 0)
            sdf = torch.where(
                outside_mask,
                torch.sqrt(torch.clamp(dx, min=0)**2 + torch.clamp(dy, min=0)**2 + 1e-8),
                torch.max(dx, dy)
            )

            # Likelihood term with effective variance including robot uncertainty
            likelihood = torch.mean(sdf ** 2) / (2.0 * sigma_sq_effective)

            # Total loss = likelihood (prediction error)
            loss = likelihood

            # Compute gradients
            if mu.grad is not None:
                mu.grad.zero_()
            loss.backward()

            # Gradient descent update
            with torch.no_grad():
                if mu.grad is not None and not torch.isnan(mu.grad).any():
                    grad = torch.clamp(mu.grad, -2.0, 2.0)
                    mu -= lr * grad

                    # Clamp dimensions (w, h, d)
                    mu[2] = torch.clamp(mu[2], self.config.min_size, self.config.max_size)
                    mu[3] = torch.clamp(mu[3], self.config.min_size, self.config.max_size)
                    mu[4] = torch.clamp(mu[4], self.config.min_size, self.config.max_size)

                    # Normalize angle to [-π/2, π/2]
                    while mu[5] > np.pi/2:
                        mu[5] -= np.pi
                    while mu[5] < -np.pi/2:
                        mu[5] += np.pi

            mu.requires_grad_(True)

        # Update belief mean
        self.mu = mu.detach()

        # Enforce width >= height convention (swap if needed)
        if self.mu[3] > self.mu[2]:
            self.mu[2], self.mu[3] = self.mu[3].clone(), self.mu[2].clone()
            self.mu[5] = self.mu[5] + np.pi/2
            while self.mu[5] > np.pi/2:
                self.mu[5] -= np.pi

        # Increase observation count
        self.observation_count += 1

    def propagate_prior(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate belief through time (prediction step).

        From paper Eq. obstacle_prior:
        μₚ = μ^(t-1)
        Σₚ = Σ^(t-1) + Σ_process

        Returns:
            (prior_mu, prior_Sigma) for the next update (6D)
        """
        prior_mu = self.mu.clone()

        # Add process noise (obstacles are static, so small noise)
        # Order: [cx, cy, w, h, d, theta]
        process_noise = torch.diag(torch.tensor([
            self.config.sigma_process_xy ** 2,      # cx
            self.config.sigma_process_xy ** 2,      # cy
            self.config.sigma_process_size ** 2,    # w
            self.config.sigma_process_size ** 2,    # h
            self.config.sigma_process_size ** 2,    # d
            self.config.sigma_process_angle ** 2    # theta
        ], dtype=DTYPE, device=self.mu.device))

        prior_Sigma = self.Sigma + process_noise

        return prior_mu, prior_Sigma

    def get_corners(self) -> np.ndarray:
        """Get the 4 corners of the rectangle in world frame."""
        cx, cy = self.cx, self.cy
        half_w, half_h = self.width / 2, self.height / 2
        theta = self.angle

        # Local corners
        local = np.array([
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h]
        ])

        # Rotation matrix
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        # Transform to world
        world = local @ R.T + np.array([cx, cy])
        return world


class BoxManager:
    """
    Manages the lifecycle of rectangular obstacle beliefs using Active Inference.

    This class implements the perception cycle from the paper:
    1. Transform LIDAR points to room frame
    2. Cluster points (DBSCAN-like)
    3. Filter wall points
    4. Associate clusters to existing beliefs (Hungarian algorithm)
    5. Update matched beliefs via VFE minimization
    6. Initialize new beliefs for unmatched clusters
    7. Decay/remove unmatched beliefs
    8. Update DSR graph
    """

    def __init__(self, g, agent_id: int, config: RectangleBeliefConfig = None):
        """
        Initialize the BoxManager.

        Args:
            g: DSR graph instance
            agent_id: Agent ID for creating nodes
            config: Configuration parameters
        """
        self.g = g
        self.agent_id = agent_id
        self.config = config or RectangleBeliefConfig()

        self.beliefs: Dict[int, RectangleBelief] = {}
        self.next_belief_id = 0
        self.frame_count = 0

        self.device = DEVICE
        console.print(f"[cyan]BoxManager initialized on device: {self.device}")

        # Robot pose uncertainty (updated each frame)
        self.robot_pose = np.array([0.0, 0.0, 0.0])
        self.robot_cov = np.eye(3) * 0.01
        self.mean_point_cov = None  # Mean point covariance tensor for SDF computation

        # Data for visualization (updated each frame)
        self.viz_data = {
            'lidar_points_raw': np.array([]),
            'lidar_points_filtered': np.array([]),
            'clusters': [],
            'room_dims': (6.0, 6.0),
            'robot_pose': np.array([0.0, 0.0, 0.0])
        }

    def get_beliefs_as_dicts(self) -> List[dict]:
        """Get beliefs as a list of dictionaries for visualization."""
        result = []
        for belief_id, belief in self.beliefs.items():
            result.append({
                'id': belief_id,
                'cx': belief.cx,
                'cy': belief.cy,
                'width': belief.width,
                'height': belief.height,
                'depth': belief.depth,
                'angle': belief.angle,
                'confidence': belief.confidence,
                'is_confirmed': belief.is_confirmed,
                'sdf_mean': belief.last_sdf_mean
            })
        return result

    def update(self, lidar_points: np.ndarray, robot_pose: np.ndarray,
               robot_cov: np.ndarray, room_dims: Tuple[float, float]) -> List[RectangleBelief]:
        """
        Main perception cycle following Active Inference framework.

        NEW APPROACH: Points stay in robot frame, boxes are transformed to robot frame
        for SDF computation. This efficiently propagates robot pose uncertainty to
        only 1 pose (the box) instead of N points.

        Args:
            lidar_points: LIDAR points in robot frame [N, 2] (meters)
            robot_pose: Robot pose [x, y, theta] in room frame
            robot_cov: Robot pose covariance 3x3
            room_dims: Room dimensions (width, depth) in meters

        Returns:
            List of current beliefs
        """
        import time
        t0 = time.time()

        self.frame_count += 1

        # Store robot pose and covariance for box-to-robot transformation
        self.viz_data['room_dims'] = room_dims
        self.viz_data['robot_pose'] = robot_pose.copy()
        self.robot_pose = robot_pose.copy()
        self.robot_cov = robot_cov.copy()

        if len(lidar_points) == 0:
            self.viz_data['lidar_points_raw'] = np.array([])
            self.viz_data['lidar_points_filtered'] = np.array([])
            self.viz_data['clusters'] = []
            self._decay_unmatched_beliefs(set())
            return list(self.beliefs.values())

        # Store original LIDAR points in robot frame for SDF computation
        self.lidar_points_robot = lidar_points.copy()

        # Transform points to room frame for visualization and wall filtering
        world_points = self._transform_to_room_frame(lidar_points, robot_pose)
        self.viz_data['lidar_points_raw'] = world_points.copy()
        t1 = time.time()

        # 2. Filter wall points (in room frame for correct wall distance)
        non_wall_points_room = self._filter_wall_points(world_points, room_dims)
        self.viz_data['lidar_points_filtered'] = non_wall_points_room.copy()

        # Also get the corresponding points in robot frame (same indices)
        wall_mask = self._get_wall_filter_mask(world_points, room_dims)
        non_wall_points_robot = lidar_points[wall_mask]
        t2 = time.time()

        if len(non_wall_points_robot) < self.config.min_cluster_points:
            self.viz_data['clusters'] = []
            self._decay_unmatched_beliefs(set())
            return list(self.beliefs.values())

        # 3. Cluster points in ROBOT frame (single clustering)
        #    Then transform clusters to room frame for visualization and association
        clusters_robot = self._cluster_points(non_wall_points_robot)

        # Transform clusters to room frame for visualization and association with beliefs
        clusters_room = [self._transform_to_room_frame(c, self.robot_pose) for c in clusters_robot]
        self.viz_data['clusters'] = [c.copy() for c in clusters_room]
        t3 = time.time()

        if len(clusters_robot) == 0:
            self._decay_unmatched_beliefs(set())
            return list(self.beliefs.values())

        # 4. Data association via cost matrix (Hungarian algorithm)
        #    Use room frame clusters for centroid distance to beliefs (which are in room frame)
        associations, unmatched_clusters, unmatched_beliefs = self._associate_clusters(clusters_room)
        t4 = time.time()

        # Debug: print detailed info every frame (commented for now)
        # if len(clusters_room) > 0:
        #     cluster_centroid = np.mean(clusters_room[0], axis=0)
        #     console.print(f"[blue]Frame {self.frame_count}: Cluster at ({cluster_centroid[0]:.2f}, {cluster_centroid[1]:.2f}), "
        #                  f"Beliefs: {len(self.beliefs)}, Associations: {len(associations)}, "
        #                  f"Unmatched clusters: {len(unmatched_clusters)}")
        #
        #     # Show distances to all beliefs
        #     if len(self.beliefs) > 0 and len(unmatched_clusters) > 0:
        #         for belief_id, belief in self.beliefs.items():
        #             dist = np.sqrt((cluster_centroid[0] - belief.cx)**2 + (cluster_centroid[1] - belief.cy)**2)
        #             console.print(f"[yellow]  Belief {belief_id}: ({belief.cx:.2f}, {belief.cy:.2f}), dist={dist:.2f}m, conf={belief.confidence:.2f}")

        # 5. Update matched beliefs via VFE minimization
        #    Pass both room and robot frame clusters (same indices)
        self._update_matched_beliefs(associations, clusters_room, clusters_robot)
        t5 = time.time()

        # 6. Initialize new beliefs for unmatched clusters (use room frame for initialization)
        self._create_new_beliefs(unmatched_clusters, clusters_room)
        t6 = time.time()

        # 7. Decay unmatched beliefs
        self._decay_unmatched_beliefs(unmatched_beliefs)

        # 8. Merge overlapping beliefs (NMS for beliefs)
        self._merge_overlapping_beliefs()

        # 9. Update DSR graph
        #self._update_dsr()

        # Print timing every 30 frames (commented for now)
        # if self.frame_count % 30 == 0:
        #     console.print(f"[yellow]Timing: transform={1000*(t1-t0):.1f}ms, filter={1000*(t2-t1):.1f}ms, "
        #                  f"cluster={1000*(t3-t2):.1f}ms, assoc={1000*(t4-t3):.1f}ms, "
        #                  f"update={1000*(t5-t4):.1f}ms, create={1000*(t6-t5):.1f}ms")

        return list(self.beliefs.values())

    def _transform_to_room_frame(self, points: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
        """
        Transform points from robot frame to room frame.

        From paper Eq. lidar_room_transform:
        T(μ_loc) p = R(θ) p + [x, y]ᵀ
        """
        x, y, theta = robot_pose
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Rotate and translate
        world_x = points[:, 0] * cos_t - points[:, 1] * sin_t + x
        world_y = points[:, 0] * sin_t + points[:, 1] * cos_t + y

        return np.column_stack([world_x, world_y])

    def _transform_box_to_robot_frame(self, box_mu: torch.Tensor, robot_pose: np.ndarray) -> torch.Tensor:
        """
        Transform box parameters from room frame to robot frame.

        box_mu = [cx, cy, w, h, d, theta] in room frame
        robot_pose = [rx, ry, rtheta] robot pose in room frame

        The inverse transformation is:
        T_room_to_robot = T_robot_to_room^(-1)

        For SE(2): if T = [R, t], then T^(-1) = [Rᵀ, -Rᵀt]

        Box center in robot frame:
        [cx_robot, cy_robot] = Rᵀ([cx_room, cy_room] - [rx, ry])

        Box angle in robot frame:
        theta_robot = theta_room - rtheta

        Returns:
            box_mu_robot [cx_robot, cy_robot, w, h, d, theta_robot]
        """
        rx, ry, rtheta = robot_pose
        cos_t = np.cos(-rtheta)  # Rᵀ = R(-θ)
        sin_t = np.sin(-rtheta)

        # Box center in room frame
        cx_room = box_mu[0]
        cy_room = box_mu[1]

        # Translate then rotate (inverse of rotate then translate)
        dx = cx_room - rx
        dy = cy_room - ry

        # Rotate to robot frame
        cx_robot = dx * cos_t - dy * sin_t
        cy_robot = dx * sin_t + dy * cos_t

        # Dimensions stay the same
        w, h, d = box_mu[2], box_mu[3], box_mu[4]

        # Angle in robot frame
        theta_robot = box_mu[5] - rtheta

        return torch.tensor([cx_robot, cy_robot, w, h, d, theta_robot],
                           dtype=DTYPE, device=box_mu.device)

    def _transform_box_to_robot_frame_with_cov(self, box_mu: torch.Tensor, box_Sigma: torch.Tensor,
                                                 robot_pose: np.ndarray, robot_cov: np.ndarray
                                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform box parameters from room frame to robot frame with covariance composition.

        This is the key function for propagating uncertainty efficiently.
        Instead of transforming N points, we transform 1 box pose and compose
        the covariances.

        The transformation is:
        s_robot = f(s_room, pose_robot)

        where f is the inverse SE(2) transformation applied to the box center
        and angle adjustment.

        Jacobian J_box of s_robot w.r.t. s_room:
        - Position: J_pos = R(-θ_robot) (2x2 rotation matrix)
        - Dimensions: Identity (don't depend on transform)
        - Angle: 1 (direct subtraction)

        Jacobian J_robot of s_robot w.r.t. robot pose (rx, ry, rθ):
        - Depends on the specific derivative of the transformation

        Total covariance in robot frame:
        Σ_robot = J_box @ Σ_box @ J_box.T + J_robot @ Σ_robot @ J_robot.T

        Args:
            box_mu: Box mean [cx, cy, w, h, d, theta] in room frame (6D)
            box_Sigma: Box covariance 6x6 in room frame
            robot_pose: Robot pose [x, y, theta] in room frame
            robot_cov: Robot pose covariance 3x3

        Returns:
            (box_mu_robot, box_Sigma_robot) in robot frame
        """
        rx, ry, rtheta = robot_pose
        cos_t = np.cos(-rtheta)
        sin_t = np.sin(-rtheta)

        # Transform box mean
        box_mu_robot = self._transform_box_to_robot_frame(box_mu, robot_pose)

        # =====================================================================
        # Jacobian of box_robot w.r.t. box_room (6x6)
        # s_robot = [cx_r, cy_r, w, h, d, θ_r]
        # s_room  = [cx,   cy,   w, h, d, θ  ]
        #
        # cx_r = cos(-rθ)*(cx - rx) - sin(-rθ)*(cy - ry)
        # cy_r = sin(-rθ)*(cx - rx) + cos(-rθ)*(cy - ry)
        # w_r = w, h_r = h, d_r = d
        # θ_r = θ - rθ
        # =====================================================================
        J_box = np.zeros((6, 6))
        J_box[0, 0] = cos_t   # ∂cx_r/∂cx
        J_box[0, 1] = -sin_t  # ∂cx_r/∂cy
        J_box[1, 0] = sin_t   # ∂cy_r/∂cx
        J_box[1, 1] = cos_t   # ∂cy_r/∂cy
        J_box[2, 2] = 1.0     # ∂w_r/∂w
        J_box[3, 3] = 1.0     # ∂h_r/∂h
        J_box[4, 4] = 1.0     # ∂d_r/∂d
        J_box[5, 5] = 1.0     # ∂θ_r/∂θ

        # =====================================================================
        # Jacobian of box_robot w.r.t. robot_pose (6x3)
        # robot_pose = [rx, ry, rθ]
        #
        # cx_r = cos(-rθ)*(cx - rx) - sin(-rθ)*(cy - ry)
        # cy_r = sin(-rθ)*(cx - rx) + cos(-rθ)*(cy - ry)
        #
        # ∂cx_r/∂rx = -cos(-rθ) = -cos_t
        # ∂cx_r/∂ry = sin(-rθ) = sin_t
        # ∂cx_r/∂rθ = sin(-rθ)*(cx - rx) + cos(-rθ)*(cy - ry) = cy_r
        #
        # ∂cy_r/∂rx = -sin(-rθ) = -sin_t
        # ∂cy_r/∂ry = -cos(-rθ) = -cos_t
        # ∂cy_r/∂rθ = cos(-rθ)*(cx - rx) - sin(-rθ)*(cy - ry) = -cx_r
        #
        # ∂θ_r/∂rθ = -1
        # =====================================================================
        cx_room = box_mu[0].item()
        cy_room = box_mu[1].item()
        dx = cx_room - rx
        dy = cy_room - ry

        # cx_r and cy_r computed for the Jacobian
        cx_r = cos_t * dx - sin_t * dy
        cy_r = sin_t * dx + cos_t * dy

        J_robot = np.zeros((6, 3))
        J_robot[0, 0] = -cos_t      # ∂cx_r/∂rx
        J_robot[0, 1] = sin_t       # ∂cx_r/∂ry
        J_robot[0, 2] = cy_r        # ∂cx_r/∂rθ (= sin(-rθ)*dx + cos(-rθ)*dy)
        J_robot[1, 0] = -sin_t      # ∂cy_r/∂rx
        J_robot[1, 1] = -cos_t      # ∂cy_r/∂ry
        J_robot[1, 2] = -cx_r       # ∂cy_r/∂rθ (= cos(-rθ)*dx - sin(-rθ)*dy, but with sign)
        J_robot[5, 2] = -1.0        # ∂θ_r/∂rθ

        # Convert to tensors
        J_box_t = torch.tensor(J_box, dtype=DTYPE, device=box_mu.device)
        J_robot_t = torch.tensor(J_robot, dtype=DTYPE, device=box_mu.device)
        robot_cov_t = torch.tensor(robot_cov, dtype=DTYPE, device=box_mu.device)

        # Compose covariances:
        # Σ_robot = J_box @ Σ_box @ J_box.T + J_robot @ Σ_robot_pose @ J_robot.T
        cov_from_box = J_box_t @ box_Sigma @ J_box_t.T
        cov_from_robot = J_robot_t @ robot_cov_t @ J_robot_t.T

        box_Sigma_robot = cov_from_box + cov_from_robot

        return box_mu_robot, box_Sigma_robot

    def _transform_to_room_frame_with_cov(self, points: np.ndarray, robot_pose: np.ndarray,
                                           robot_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform points from robot frame to room frame with uncertainty propagation.

        From paper Eq. lidar_room_transform:
        T(μ_loc) p = R(θ) p + [x, y]ᵀ

        The transformation is nonlinear due to θ. We propagate uncertainty using
        first-order linearization (Jacobian-based):

        For each point p_rob = (px, py) in robot frame:
        p_world = [cos(θ)*px - sin(θ)*py + x, sin(θ)*px + cos(θ)*py + y]ᵀ

        Jacobian J of p_world w.r.t. robot pose (x, y, θ):
        J = [∂p_world/∂x, ∂p_world/∂y, ∂p_world/∂θ]
          = [[1, 0, -sin(θ)*px - cos(θ)*py],
             [0, 1,  cos(θ)*px - sin(θ)*py]]

        Covariance propagation: Σ_world = J * Σ_robot * Jᵀ

        Args:
            points: LIDAR points in robot frame [N, 2]
            robot_pose: Robot pose [x, y, theta] in room frame
            robot_cov: Robot pose covariance 3x3

        Returns:
            Tuple (world_points [N, 2], point_covariances [N, 2, 2])
        """
        x, y, theta = robot_pose
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        n_points = len(points)

        # Transform points (mean)
        world_x = points[:, 0] * cos_t - points[:, 1] * sin_t + x
        world_y = points[:, 0] * sin_t + points[:, 1] * cos_t + y
        world_points = np.column_stack([world_x, world_y])

        # Compute covariance for each point via Jacobian propagation
        # J is 2x3 for each point: [∂p_world/∂(x,y,θ)]
        # Σ_world (2x2) = J (2x3) @ Σ_robot (3x3) @ Jᵀ (3x2)
        point_covs = np.zeros((n_points, 2, 2))

        for i in range(n_points):
            px, py = points[i]

            # Jacobian of transformation w.r.t. robot pose (x, y, θ)
            # ∂(world_x)/∂x = 1, ∂(world_x)/∂y = 0, ∂(world_x)/∂θ = -sin(θ)*px - cos(θ)*py
            # ∂(world_y)/∂x = 0, ∂(world_y)/∂y = 1, ∂(world_y)/∂θ = cos(θ)*px - sin(θ)*py
            J = np.array([
                [1.0, 0.0, -sin_t * px - cos_t * py],
                [0.0, 1.0,  cos_t * px - sin_t * py]
            ])

            # Propagate covariance: Σ_world = J @ Σ_robot @ Jᵀ
            point_covs[i] = J @ robot_cov @ J.T

        return world_points, point_covs

    def _filter_wall_points(self, points: np.ndarray, room_dims: Tuple[float, float]) -> np.ndarray:
        """
        Filter out points belonging to walls.

        Uses distance to room boundaries. Points close to walls are likely
        wall reflections, not obstacles.
        """
        width, depth = room_dims
        half_w, half_d = width / 2, depth / 2
        margin = self.config.wall_margin

        # Distance to each wall
        dist_left = np.abs(points[:, 0] + half_w)
        dist_right = np.abs(points[:, 0] - half_w)
        dist_back = np.abs(points[:, 1] + half_d)
        dist_front = np.abs(points[:, 1] - half_d)

        min_wall_dist = np.minimum(
            np.minimum(dist_left, dist_right),
            np.minimum(dist_back, dist_front)
        )

        # Keep points far from walls
        mask = min_wall_dist > margin

        # Also filter points that are outside the room (sensor noise)
        inside_room = (np.abs(points[:, 0]) < half_w - 0.05) & (np.abs(points[:, 1]) < half_d - 0.05)
        mask = mask & inside_room

        return points[mask]

    def _get_wall_filter_mask(self, points: np.ndarray, room_dims: Tuple[float, float]) -> np.ndarray:
        """
        Get boolean mask for wall filtering (True = keep, False = wall point).

        Same logic as _filter_wall_points but returns mask instead of filtered points.
        """
        width, depth = room_dims
        half_w, half_d = width / 2, depth / 2
        margin = self.config.wall_margin

        # Distance to each wall
        dist_left = np.abs(points[:, 0] + half_w)
        dist_right = np.abs(points[:, 0] - half_w)
        dist_back = np.abs(points[:, 1] + half_d)
        dist_front = np.abs(points[:, 1] - half_d)

        min_wall_dist = np.minimum(
            np.minimum(dist_left, dist_right),
            np.minimum(dist_back, dist_front)
        )

        # Keep points far from walls
        mask = min_wall_dist > margin

        # Also filter points that are outside the room (sensor noise)
        inside_room = (np.abs(points[:, 0]) < half_w - 0.05) & (np.abs(points[:, 1]) < half_d - 0.05)
        mask = mask & inside_room

        return mask

    def _cluster_points(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Cluster points using distance-based clustering with aggressive expansion.

        Similar to DBSCAN as described in paper Sec. obstacle-dbscan.
        Uses larger expansion radius to merge nearby point groups into single clusters.
        Applies NMS-like merging to avoid duplicate clusters for the same object.
        """
        if len(points) == 0:
            return []

        eps = self.config.cluster_eps
        min_pts = self.config.min_cluster_points

        clusters = []
        remaining = points.copy()
        used = np.zeros(len(remaining), dtype=bool)

        for i in range(len(remaining)):
            if used[i]:
                continue

            seed = remaining[i]

            # Find all neighbors within eps
            distances = np.linalg.norm(remaining - seed, axis=1)
            neighbor_mask = (distances < eps) & (~used)

            if np.sum(neighbor_mask) < min_pts:
                continue

            # Start cluster with initial neighbors
            cluster_indices = set(np.where(neighbor_mask)[0])
            used[list(cluster_indices)] = True

            # Expand cluster iteratively
            to_check = list(cluster_indices)
            while to_check:
                current_idx = to_check.pop(0)
                current_point = remaining[current_idx]

                # Find neighbors of this point
                distances = np.linalg.norm(remaining - current_point, axis=1)
                new_neighbors = np.where((distances < eps) & (~used))[0]

                for neighbor_idx in new_neighbors:
                    if neighbor_idx not in cluster_indices:
                        cluster_indices.add(neighbor_idx)
                        used[neighbor_idx] = True
                        to_check.append(neighbor_idx)

            # Create cluster from collected indices
            cluster = remaining[list(cluster_indices)]
            if len(cluster) >= min_pts:
                clusters.append(cluster)

        # Apply NMS-like merging to combine overlapping clusters
        clusters = self._merge_overlapping_clusters(clusters)

        return clusters

    def _merge_overlapping_clusters(self, clusters: List[np.ndarray]) -> List[np.ndarray]:
        """
        Merge clusters that are too close together (NMS-like).

        Single-pass algorithm for efficiency.
        """
        if len(clusters) <= 1:
            return clusters

        merge_distance = self.config.cluster_eps * 2.5

        # Pre-compute centroids and bboxes
        centroids = [np.mean(c, axis=0) for c in clusters]
        bboxes = [self._get_bbox(c) for c in clusters]

        merged = []
        used = [False] * len(clusters)

        for i in range(len(clusters)):
            if used[i]:
                continue

            # Start with cluster i
            to_merge = [i]
            used[i] = True

            # Single pass: find all clusters to merge with this group
            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue

                # Check against any cluster in the merge group
                should_merge = False
                for k in to_merge:
                    centroid_dist = np.linalg.norm(centroids[k] - centroids[j])
                    bbox_overlap = self._bbox_overlap(bboxes[k], bboxes[j])

                    if centroid_dist < merge_distance or bbox_overlap:
                        should_merge = True
                        break

                if should_merge:
                    to_merge.append(j)
                    used[j] = True

            # Merge all clusters in the group
            merged_cluster = np.vstack([clusters[idx] for idx in to_merge])
            merged.append(merged_cluster)

        return merged

    def _get_bbox(self, points: np.ndarray) -> Tuple[float, float, float, float]:
        """Get bounding box (min_x, min_y, max_x, max_y) for a set of points."""
        min_xy = np.min(points, axis=0)
        max_xy = np.max(points, axis=0)
        return (min_xy[0], min_xy[1], max_xy[0], max_xy[1])

    def _bbox_overlap(self, bbox1: Tuple[float, float, float, float],
                      bbox2: Tuple[float, float, float, float]) -> bool:
        """Check if two bounding boxes overlap or are very close."""
        margin = self.config.cluster_eps  # Small margin for near-overlaps

        # Expand bbox1 by margin
        x1_min, y1_min, x1_max, y1_max = bbox1
        x1_min -= margin
        y1_min -= margin
        x1_max += margin
        y1_max += margin

        x2_min, y2_min, x2_max, y2_max = bbox2

        # Check for no overlap
        if x1_max < x2_min or x2_max < x1_min:
            return False
        if y1_max < y2_min or y2_max < y1_min:
            return False

        return True

    def _associate_clusters(self, clusters: List[np.ndarray]) -> Tuple[List[Tuple[int, int]],
                                                                        set, set]:
        """
        Associate clusters to beliefs via Hungarian algorithm.

        Uses a combination of:
        1. Distance between cluster centroid and belief center
        2. SDF residuals for shape matching

        Returns:
            (associations, unmatched_cluster_indices, unmatched_belief_ids)
        """
        if len(clusters) == 0 or len(self.beliefs) == 0:
            # No associations possible
            unmatched_clusters = set(range(len(clusters)))
            unmatched_beliefs = set(self.beliefs.keys())
            return [], unmatched_clusters, unmatched_beliefs

        n_clusters = len(clusters)
        n_beliefs = len(self.beliefs)
        belief_ids = list(self.beliefs.keys())

        # Build cost matrix using centroid distance + SDF cost
        cost_matrix = np.full((n_clusters, n_beliefs), np.inf)

        for k, cluster in enumerate(clusters):
            cluster_centroid = np.mean(cluster, axis=0)
            cluster_t = torch.tensor(cluster, dtype=DTYPE, device=self.device)

            for j, belief_id in enumerate(belief_ids):
                belief = self.beliefs[belief_id]

                # 1. Distance between centroids
                center_dist = np.sqrt((cluster_centroid[0] - belief.cx)**2 +
                                      (cluster_centroid[1] - belief.cy)**2)

                # Skip if too far
                if center_dist > self.config.max_association_distance:
                    continue

                # 2. SDF residuals for shape matching
                sdf_values = belief.sdf(cluster_t)
                sdf_cost = torch.mean(sdf_values ** 2).item()  # Use mean instead of sum

                # Combined cost: distance + sdf (normalized)
                cost = center_dist + sdf_cost
                cost_matrix[k, j] = cost

        # Check if cost matrix is feasible (has at least one finite value)
        if np.all(np.isinf(cost_matrix)):
            # No valid associations possible - all clusters are new
            unmatched_clusters = set(range(n_clusters))
            unmatched_beliefs = set(belief_ids)
            return [], unmatched_clusters, unmatched_beliefs

        # Replace inf with a large finite value for Hungarian algorithm
        max_finite = np.max(cost_matrix[np.isfinite(cost_matrix)]) if np.any(np.isfinite(cost_matrix)) else 1.0
        large_cost = max_finite * 1000
        cost_matrix_safe = np.where(np.isinf(cost_matrix), large_cost, cost_matrix)

        # Solve assignment problem (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix_safe)

        # Filter valid associations (use original cost_matrix to check validity)
        associations = []
        matched_clusters = set()
        matched_beliefs = set()

        for k, j in zip(row_ind, col_ind):
            # Only accept if original cost was finite
            if np.isfinite(cost_matrix[k, j]):
                associations.append((k, belief_ids[j]))
                matched_clusters.add(k)
                matched_beliefs.add(belief_ids[j])
                # Debug output
                # console.print(f"[blue]Associated cluster {k} with belief {belief_ids[j]}, cost={cost_matrix[k,j]:.3f}")

        unmatched_clusters = set(range(n_clusters)) - matched_clusters
        unmatched_beliefs = set(belief_ids) - matched_beliefs

        return associations, unmatched_clusters, unmatched_beliefs

    def _update_matched_beliefs(self, associations: List[Tuple[int, int]],
                                 clusters_room: List[np.ndarray],
                                 clusters_robot: List[np.ndarray]):
        """
        Update matched beliefs by minimizing Variational Free Energy.

        NEW APPROACH: The box is transformed from room frame to robot frame,
        composing the covariances. The SDF is computed with LIDAR points
        in robot frame. This propagates robot pose uncertainty efficiently.

        From paper Eq. obstacle_objective:
        F(s) = (1/2σ²_o) Σᵢ dᵢ(s)² + (1/2)(s-μₚ)ᵀΣₚ⁻¹(s-μₚ)

        The composed covariance (box + robot) is used in the prior term.
        """
        for cluster_idx, belief_id in associations:
            # Get cluster in ROBOT frame for SDF computation
            cluster_robot = clusters_robot[cluster_idx] if cluster_idx < len(clusters_robot) else clusters_room[cluster_idx]
            belief = self.beliefs[belief_id]

            # Store position before optimization
            old_cx, old_cy = belief.cx, belief.cy

            # Convert points to tensor (in robot frame)
            points_robot_t = torch.tensor(cluster_robot, dtype=DTYPE, device=self.device)

            # Prior = posterior from previous frame (current belief state in room frame)
            prior_mu_room, prior_Sigma_room = belief.propagate_prior()

            # Transform box to robot frame with covariance composition
            # This composes box uncertainty + robot pose uncertainty
            box_mu_robot, box_Sigma_robot = self._transform_box_to_robot_frame_with_cov(
                belief.mu, belief.Sigma, self.robot_pose, self.robot_cov
            )

            # Optimize in robot frame
            # The optimization updates box_mu_robot, then we transform back to room frame
            optimized_mu_robot, sdf_mean = self._optimize_in_robot_frame(
                points_robot_t, box_mu_robot, box_Sigma_robot, prior_mu_room, prior_Sigma_room, belief.config
            )

            # Transform optimized parameters back to room frame
            belief.mu = self._transform_box_to_room_frame(optimized_mu_robot, self.robot_pose)

            # Store the SDF mean for visualization
            belief.last_sdf_mean = sdf_mean

            # Log movement (commented for now)
            # move_dist = np.sqrt((belief.cx - old_cx)**2 + (belief.cy - old_cy)**2)
            # if move_dist > 0.05:  # Only log movements > 5cm
            #     console.print(f"[magenta]Belief {belief_id} moved {move_dist:.2f}m: ({old_cx:.2f},{old_cy:.2f}) -> ({belief.cx:.2f},{belief.cy:.2f})")

            # Update lifecycle
            belief.last_seen = self.frame_count
            belief.age += 1
            belief.confidence = min(1.0, belief.confidence + self.config.confidence_boost)
            belief.observation_count += 1

            # Check for confirmation
            if belief.confidence >= self.config.confirmed_threshold:
                belief.is_confirmed = True

    def _transform_box_to_room_frame(self, box_mu_robot: torch.Tensor,
                                      robot_pose: np.ndarray) -> torch.Tensor:
        """
        Transform box parameters from robot frame back to room frame.

        This is the forward transformation (inverse of _transform_box_to_robot_frame).

        T_robot_to_room: p_room = R(θ) * p_robot + [rx, ry]

        Box center in room frame:
        [cx_room, cy_room] = R(θ) * [cx_robot, cy_robot] + [rx, ry]

        Box angle in room frame:
        theta_room = theta_robot + rtheta
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

    def _optimize_in_robot_frame(self, points: torch.Tensor, box_mu: torch.Tensor,
                                  box_Sigma: torch.Tensor, prior_mu_room: torch.Tensor,
                                  prior_Sigma_room: torch.Tensor, config: RectangleBeliefConfig,
                                  num_iters: int = 10, lr: float = 0.05) -> torch.Tensor:
        """
        Optimize box parameters in robot frame via VFE minimization.

        The SDF is computed with points and box in the same frame (robot).
        The composed covariance (box + robot) is used for regularization.

        Args:
            points: LIDAR points in robot frame [N, 2]
            box_mu: Box mean in robot frame [cx, cy, w, h, d, theta]
            box_Sigma: Composed covariance (box + robot) in robot frame
            prior_mu_room: Prior mean in room frame (for reference)
            prior_Sigma_room: Prior covariance in room frame
            config: Belief configuration

        Returns:
            Tuple (optimized_box_mu, sdf_mean) - optimized parameters and mean SDF value
        """
        if len(points) < 3:
            return box_mu, 0.0

        sigma_sq = config.sigma_obs ** 2

        # Make parameters require gradients
        mu = box_mu.clone().detach().requires_grad_(True)
        final_sdf_mean = 0.0

        for iteration in range(num_iters):
            # Extract parameters: cx, cy, w, h, d, theta
            cx, cy, w, h, d, theta = mu[0], mu[1], mu[2], mu[3], mu[4], mu[5]

            # Transform points to local box frame
            cos_t = torch.cos(-theta)
            sin_t = torch.sin(-theta)
            px = points[:, 0] - cx
            py = points[:, 1] - cy
            local_x = px * cos_t - py * sin_t
            local_y = px * sin_t + py * cos_t

            # Half dimensions
            half_w = w / 2
            half_h = h / 2

            # Distance to box faces
            dx = torch.abs(local_x) - half_w
            dy = torch.abs(local_y) - half_h

            # 2D SDF
            outside_mask = (dx > 0) | (dy > 0)
            sdf = torch.where(
                outside_mask,
                torch.sqrt(torch.clamp(dx, min=0)**2 + torch.clamp(dy, min=0)**2 + 1e-8),
                torch.max(dx, dy)
            )

            # Store final SDF mean (from last iteration)
            final_sdf_mean = torch.mean(torch.abs(sdf)).item()

            # Likelihood term with effective variance
            # The composed covariance already includes robot uncertainty
            likelihood = torch.mean(sdf ** 2) / (2.0 * sigma_sq)

            # Total loss = likelihood (prediction error)
            loss = likelihood

            # Compute gradients
            if mu.grad is not None:
                mu.grad.zero_()
            loss.backward()

            # Gradient descent update
            with torch.no_grad():
                if mu.grad is not None and not torch.isnan(mu.grad).any():
                    grad = torch.clamp(mu.grad, -2.0, 2.0)
                    mu -= lr * grad

                    # Clamp dimensions (w, h, d)
                    mu[2] = torch.clamp(mu[2], config.min_size, config.max_size)
                    mu[3] = torch.clamp(mu[3], config.min_size, config.max_size)
                    mu[4] = torch.clamp(mu[4], config.min_size, config.max_size)

                    # Normalize angle to [-π/2, π/2]
                    while mu[5] > np.pi/2:
                        mu[5] -= np.pi
                    while mu[5] < -np.pi/2:
                        mu[5] += np.pi

            mu.requires_grad_(True)

        # Enforce width >= height convention (swap if needed)
        optimized_mu = mu.detach()
        if optimized_mu[3] > optimized_mu[2]:
            optimized_mu[2], optimized_mu[3] = optimized_mu[3].clone(), optimized_mu[2].clone()
            optimized_mu[5] = optimized_mu[5] + np.pi/2
            while optimized_mu[5] > np.pi/2:
                optimized_mu[5] -= np.pi

        return optimized_mu, final_sdf_mean

    def _create_new_beliefs(self, unmatched_clusters: set, clusters: List[np.ndarray]):
        """
        Initialize new beliefs for unmatched clusters.

        From paper Sec. circle_fit (adapted for 3D boxes):
        Uses PCA-based fitting for initial parameters.
        Depth is initialized from the prior mean (0.5m typical box).

        Filters out:
        - Clusters that are too small or too large
        - Clusters with extreme aspect ratios (likely walls)
        """
        for cluster_idx in unmatched_clusters:
            cluster = clusters[cluster_idx]

            if len(cluster) < self.config.min_cluster_points:
                continue

            # Fit rectangle using PCA (2D projection)
            result = self._fit_rectangle_pca(cluster)
            if result is None:
                continue

            cx, cy, w, h, angle = result

            # Validate dimensions
            if w < self.config.min_size or h < self.config.min_size:
                continue
            if w > self.config.max_size or h > self.config.max_size:
                continue

            # Check aspect ratio - reject thin lines (likely walls)
            aspect_ratio = min(w, h) / max(w, h)
            if aspect_ratio < self.config.min_aspect_ratio:
                # This looks like a wall segment, not a box
                continue

            # Initialize depth from prior (typical box size 0.5m)
            d = self.config.prior_size_mean

            # Create 6D belief: [cx, cy, w, h, d, theta]
            mu = torch.tensor([cx, cy, w, h, d, angle], dtype=DTYPE, device=self.device)

            # 6x6 covariance matrix
            Sigma = torch.diag(torch.tensor([
                self.config.initial_position_std ** 2,   # cx
                self.config.initial_position_std ** 2,   # cy
                self.config.initial_size_std ** 2,       # w
                self.config.initial_size_std ** 2,       # h
                self.config.initial_size_std ** 2,       # d (higher uncertainty, not observed)
                self.config.initial_angle_std ** 2       # theta
            ], dtype=DTYPE, device=self.device))

            belief = RectangleBelief(
                id=self.next_belief_id,
                mu=mu,
                Sigma=Sigma,
                confidence=self.config.initial_confidence,
                age=0,
                last_seen=self.frame_count,
                observation_count=1,
                is_confirmed=False,
                config=self.config
            )

            self.beliefs[self.next_belief_id] = belief
            self.next_belief_id += 1

            # console.print(f"[green]New belief: id={belief.id}, pos=({cx:.2f}, {cy:.2f}), "
            #              f"size=({w:.2f}, {h:.2f}, {d:.2f}), angle={np.degrees(angle):.1f}°")

    def _fit_rectangle_pca(self, points: np.ndarray) -> Optional[Tuple[float, float, float, float, float]]:
        """
        Fit oriented rectangle to points using PCA.

        Also rejects linear clusters (walls) based on eigenvalue ratio.

        Returns:
            (cx, cy, width, height, angle) or None if fitting fails or cluster is linear
        """
        if len(points) < 3:
            return None

        # Centroid
        centroid = np.mean(points, axis=0)

        # PCA for orientation
        centered = points - centroid
        cov = np.cov(centered.T)

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Check linearity using eigenvalue ratio
            # If one eigenvalue dominates, the cluster is a line (wall)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            if eigenvalues[0] > 0:
                linearity = 1.0 - (eigenvalues[1] / eigenvalues[0])
                # If linearity > 0.85, this is likely a wall segment
                if linearity > 0.85:
                    return None

            # Principal axis
            angle = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
            # Normalize to [-π/2, π/2]
            while angle > np.pi/2:
                angle -= np.pi
            while angle < -np.pi/2:
                angle += np.pi
        except:
            angle = 0.0

        # Rotate points to align with axes
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rotated = np.column_stack([
            centered[:, 0] * cos_a - centered[:, 1] * sin_a,
            centered[:, 0] * sin_a + centered[:, 1] * cos_a
        ])

        # Bounding box
        min_xy = np.min(rotated, axis=0)
        max_xy = np.max(rotated, axis=0)

        width = max(max_xy[0] - min_xy[0], 0.1)
        height = max(max_xy[1] - min_xy[1], 0.1)

        # Ensure width >= height
        if height > width:
            width, height = height, width
            angle += np.pi/2
            while angle > np.pi/2:
                angle -= np.pi

        return centroid[0], centroid[1], width, height, angle

    def _decay_unmatched_beliefs(self, unmatched_belief_ids: set):
        """
        Apply confidence decay and uncertainty growth to unmatched beliefs.

        From paper Sec. belief_decay:
        κⱼ^(t) = γ · κⱼ^(t-1)
        Σⱼ^(t) = Σⱼ^(t-1) + Σ_process

        Also decays confirmed beliefs if not seen for too long.
        """
        to_remove = []
        frames_to_decay_confirmed = 30  # Start decaying confirmed beliefs after 30 frames unseen

        for belief_id in list(self.beliefs.keys()):
            belief = self.beliefs[belief_id]

            # Check if this belief was matched this frame
            was_matched = belief_id not in unmatched_belief_ids

            if was_matched:
                continue  # Skip beliefs that were matched

            # Calculate frames since last seen
            frames_unseen = self.frame_count - belief.last_seen

            # Decay provisional beliefs immediately
            # Decay confirmed beliefs only after frames_to_decay_confirmed frames unseen
            should_decay = (not belief.is_confirmed) or (frames_unseen > frames_to_decay_confirmed)

            if should_decay:
                # Confidence decay
                belief.confidence *= self.config.confidence_decay

                # Uncertainty growth (6x6 process noise)
                process_noise = torch.diag(torch.tensor([
                    self.config.sigma_process_xy ** 2,      # cx
                    self.config.sigma_process_xy ** 2,      # cy
                    self.config.sigma_process_size ** 2,    # w
                    self.config.sigma_process_size ** 2,    # h
                    self.config.sigma_process_size ** 2,    # d
                    self.config.sigma_process_angle ** 2    # theta
                ], dtype=DTYPE, device=belief.mu.device))

                belief.Sigma = belief.Sigma + process_noise

            # Check for removal
            if belief.confidence < self.config.confidence_threshold:
                to_remove.append(belief_id)

        # Remove beliefs
        for belief_id in to_remove:
            # console.print(f"[red]Removing belief: id={belief_id}, confidence below threshold")
            self._delete_from_dsr(belief_id)
            del self.beliefs[belief_id]

    def _merge_overlapping_beliefs(self):
        """
        Merge beliefs that are too close together (NMS for beliefs).

        Keeps the belief with higher confidence and removes duplicates.
        """
        if len(self.beliefs) <= 1:
            return

        merge_distance = 0.3  # Beliefs closer than 30cm are considered duplicates

        belief_ids = list(self.beliefs.keys())
        to_remove = set()

        for i, id1 in enumerate(belief_ids):
            if id1 in to_remove:
                continue

            belief1 = self.beliefs[id1]

            for j in range(i + 1, len(belief_ids)):
                id2 = belief_ids[j]
                if id2 in to_remove:
                    continue

                belief2 = self.beliefs[id2]

                # Calculate distance between centers
                dist = np.sqrt((belief1.cx - belief2.cx)**2 + (belief1.cy - belief2.cy)**2)

                if dist < merge_distance:
                    # Keep the one with higher confidence
                    if belief1.confidence >= belief2.confidence:
                        to_remove.add(id2)
                        # Transfer observation count
                        belief1.observation_count += belief2.observation_count
                    else:
                        to_remove.add(id1)
                        belief2.observation_count += belief1.observation_count
                        break  # id1 is removed, stop checking against it

        # Remove merged beliefs
        for belief_id in to_remove:
            # console.print(f"[magenta]Merging belief: id={belief_id} (duplicate)")
            self._delete_from_dsr(belief_id)
            del self.beliefs[belief_id]

    def _update_dsr(self):
        """Update DSR graph with current beliefs."""
        for belief_id, belief in self.beliefs.items():
            box_name = f"box_{belief_id}"
            existing_node = self.g.get_node(box_name)

            if existing_node:
                # Update existing node
                existing_node.attrs["pos_x"] = Attribute(float(belief.cx * 1000), self.agent_id)
                existing_node.attrs["pos_y"] = Attribute(float(belief.cy * 1000), self.agent_id)
                existing_node.attrs["width"] = Attribute(float(belief.width * 1000), self.agent_id)
                existing_node.attrs["height"] = Attribute(float(belief.height * 1000), self.agent_id)
                existing_node.attrs["depth"] = Attribute(float(belief.depth * 1000), self.agent_id)
                existing_node.attrs["rotation_angle"] = Attribute(float(belief.angle), self.agent_id)
                existing_node.attrs["confidence"] = Attribute(float(belief.confidence), self.agent_id)
                self.g.update_node(existing_node)
            else:
                # Create new node
                box_node = Node(agent_id=self.agent_id, type="box", name=box_name)
                box_node.attrs["pos_x"] = Attribute(float(belief.cx * 1000), self.agent_id)
                box_node.attrs["pos_y"] = Attribute(float(belief.cy * 1000), self.agent_id)
                box_node.attrs["width"] = Attribute(float(belief.width * 1000), self.agent_id)
                box_node.attrs["height"] = Attribute(float(belief.height * 1000), self.agent_id)
                box_node.attrs["depth"] = Attribute(float(belief.depth * 1000), self.agent_id)
                box_node.attrs["rotation_angle"] = Attribute(float(belief.angle), self.agent_id)
                box_node.attrs["confidence"] = Attribute(float(belief.confidence), self.agent_id)
                box_node.attrs["color"] = Attribute("Orange", self.agent_id)

                new_id = self.g.insert_node(box_node)
                # if new_id:
                #     console.print(f"[green]Inserted DSR node: {box_name}")

    def _delete_from_dsr(self, belief_id: int):
        """Delete a box node from DSR graph."""
        box_name = f"box_{belief_id}"
        self.g.delete_node(box_name)
