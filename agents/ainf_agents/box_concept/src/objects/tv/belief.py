#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
TV Belief - Belief class for TV (thin panel) obstacles.

State: [cx, cy, width, height, z_base, theta] (6 parameters)
- cx, cy: center position in XY plane
- width: horizontal extent (wide dimension)
- height: vertical extent (screen height)
- z_base: height from floor to bottom of TV (wall mount height)
- theta: rotation angle

Note: depth is fixed at 5cm (TV_FIXED_DEPTH) - not a free parameter.

TVs are characterized by:
- Being much wider than deep (thin panels)
- Having approximately 2:1 aspect ratio (width/height)
- Being aligned with walls
- Being mounted at a certain height from the floor
"""

import numpy as np
import torch
from typing import Optional, Tuple
from dataclasses import dataclass

from src.belief_core import Belief, BeliefConfig, DEVICE, DTYPE
from src.objects.tv.sdf import compute_tv_sdf, compute_tv_priors, TV_TYPICAL_ASPECT, TV_FIXED_DEPTH
from src.transforms import compute_jacobian_room_to_robot


@dataclass
class TVBeliefConfig(BeliefConfig):
    """Configuration for TV beliefs."""

    # Size priors - typical TV dimensions (1m x 0.5m)
    prior_width: float = 1.0          # Typical screen width (1m)
    prior_height: float = 0.5         # Typical screen height (0.5m)
    prior_size_std: float = 0.1       # Standard deviation for size priors

    # TV-specific shape constraints
    min_width: float = 0.4            # Minimum TV width (40cm)
    max_width: float = 1.2            # Maximum TV width (1.2m)
    min_height: float = 0.2           # Minimum TV height (20cm)
    max_height: float = 0.7           # Maximum TV height (70cm)
    min_z_base: float = 0.5           # Minimum mount height (50cm from floor)
    max_z_base: float = 1.5           # Maximum mount height (1.5m from floor)
    prior_z_base: float = 1.0         # Typical wall mount height (1m)

    # Aspect ratio constraints (width/height ~ 2:1 for 1m x 0.5m)
    target_screen_aspect: float = 2.0   # Target width/height ratio
    screen_aspect_tolerance: float = 0.4  # Tolerance for aspect ratio
    screen_aspect_weight: float = 0.5   # Weight for aspect ratio prior

    # Angle alignment (TVs align with walls)
    angle_alignment_weight: float = 1.5   # Stronger than boxes
    angle_alignment_sigma: float = 0.08   # Tighter alignment (~4.5°)

    # Clustering
    cluster_eps: float = 0.20
    min_cluster_points: int = 15

    # Association
    max_association_cost: float = 5.0
    max_association_distance: float = 1.5  # Larger for wall-mounted objects (was 1.0)
    wall_margin: float = 0.30

    # Historical points storage
    max_historical_points: int = 500
    sdf_threshold_for_storage: float = 0.06
    beta_sdf: float = 1.0

    # Binning for uniform surface coverage
    num_angle_bins: int = 24
    num_z_bins: int = 10
    edge_bonus_weight: float = 0.3
    edge_proximity_threshold: float = 0.04

    # RFE parameters
    rfe_alpha: float = 0.98
    rfe_max_threshold: float = 2.0

    # RFE classification thresholds
    rfe_trusted_threshold: float = 0.03
    rfe_good_threshold: float = 0.1
    rfe_moderate_threshold: float = 1.0

    # Edge/corner classification
    corner_score_threshold: float = 0.7
    edge_score_threshold: float = 0.3


class TVBelief(Belief):
    """Gaussian belief over TV parameters [cx, cy, width, height, z_base, theta]."""

    STATE_DIM = 6

    @property
    def state_dim(self) -> int:
        return self.STATE_DIM

    @property
    def position(self) -> Tuple[float, float]:
        return (self.mu[0].item(), self.mu[1].item())

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
        """Fixed depth (5cm) - not a free parameter."""
        return TV_FIXED_DEPTH

    @property
    def z_base(self) -> float:
        """Height from floor to bottom of TV."""
        return self.mu[4].item()

    @property
    def angle(self) -> float:
        return self.mu[5].item()

    def to_dict(self) -> dict:
        """Convert TV belief to dictionary for visualization/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'type': 'tv',
            'cx': self.cx,
            'cy': self.cy,
            'width': self.width,
            'height': self.height,
            'depth': self.depth,  # Fixed at TV_FIXED_DEPTH
            'z_base': self.z_base,
            'angle': self.angle,
        })
        return base_dict

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """Compute SDF for points relative to this TV."""
        assert points.shape[1] == 3, f"Points must be 3D, got {points.shape}"
        return compute_tv_sdf(points, self.mu)

    def compute_prior_term(self, mu: torch.Tensor, robot_pose: np.ndarray = None) -> torch.Tensor:
        """
        Compute TV-specific prior energy term.

        Includes:
        1. State transition prior (static object)
        2. TV shape priors (aspect ratio)
        3. Angle alignment prior

        Args:
            mu: Current state estimate [6] in robot frame
            robot_pose: Robot pose [x, y, theta]

        Returns:
            Prior energy term (scalar tensor)
        """
        total_prior = torch.tensor(0.0, dtype=mu.dtype, device=mu.device)

        # =================================================================
        # TV SHAPE PRIORS
        # =================================================================
        width, height = mu[2], mu[3]

        # Screen aspect ratio prior: width/height ≈ 2:1
        target_aspect = self.config.target_screen_aspect
        current_aspect = width / (height + 0.01)
        aspect_error = (current_aspect - target_aspect) / target_aspect
        if torch.abs(aspect_error) > self.config.screen_aspect_tolerance:
            total_prior = total_prior + self.config.screen_aspect_weight * (aspect_error ** 2)

        # =================================================================
        # ANGLE ALIGNMENT PRIOR: TVs align with walls
        # =================================================================
        theta = mu[5]  # theta is now index 5
        if robot_pose is not None:
            theta_room = theta + robot_pose[2]
        else:
            theta_room = theta

        # Normalize to [-π, π]
        theta_room = torch.atan2(torch.sin(theta_room), torch.cos(theta_room))

        sigma = self.config.angle_alignment_sigma
        precision = 1.0 / (sigma ** 2)

        # Distance to nearest aligned angle (0°, ±90°, 180°)
        dist_to_0 = theta_room ** 2
        dist_to_pos90 = (theta_room - np.pi/2) ** 2
        dist_to_neg90 = (theta_room + np.pi/2) ** 2
        dist_to_180 = (torch.abs(theta_room) - np.pi) ** 2

        min_dist = torch.minimum(
            torch.minimum(dist_to_0, dist_to_180),
            torch.minimum(dist_to_pos90, dist_to_neg90)
        )
        total_prior = total_prior + self.config.angle_alignment_weight * 0.5 * precision * min_dist

        # =================================================================
        # STATE TRANSITION PRIOR: penalize deviation from previous state
        # =================================================================
        if self.mu is not None and robot_pose is not None:
            from src.transforms import transform_object_to_robot_frame
            mu_prev_robot = transform_object_to_robot_frame(self.mu, robot_pose)

            lambda_pos = 0.05
            lambda_size = 0.03  # Slightly stronger for TV
            lambda_z_base = 0.02  # z_base should be stable
            lambda_angle = 0.02  # Stronger angle regularization for TV

            # Position difference (cx, cy)
            diff_pos = mu[:2] - mu_prev_robot[:2]
            total_prior = total_prior + lambda_pos * torch.sum(diff_pos ** 2)

            # Size differences (width, height)
            diff_size = mu[2:4] - mu_prev_robot[2:4]
            total_prior = total_prior + lambda_size * torch.sum(diff_size ** 2)

            # z_base difference (index 4)
            diff_z_base = mu[4] - mu_prev_robot[4]
            total_prior = total_prior + lambda_z_base * (diff_z_base ** 2)

            # Angle difference (index 5)
            diff_angle = mu[5] - mu_prev_robot[5]
            diff_angle = torch.atan2(torch.sin(diff_angle), torch.cos(diff_angle))
            total_prior = total_prior + lambda_angle * (diff_angle ** 2)

        return total_prior

    def _get_process_noise_variances(self) -> list:
        cfg = self.config
        return [
            cfg.sigma_process_xy**2,      # cx
            cfg.sigma_process_xy**2,      # cy
            cfg.sigma_process_size**2,    # width
            cfg.sigma_process_size**2,    # height
            cfg.sigma_process_size**2,    # z_base
            cfg.sigma_process_angle**2 * 0.5  # theta (less variance, TV is aligned)
        ]

    @classmethod
    def from_cluster(cls, belief_id: int, cluster: np.ndarray,
                     config: TVBeliefConfig, device: torch.device = DEVICE) -> Optional['TVBelief']:
        """Create a TV belief from a point cluster."""

        if len(cluster) < config.min_cluster_points:
            return None

        # Get cluster bounding box in XY
        pts_xy = cluster[:, :2]
        centroid = np.mean(pts_xy, axis=0)

        # Use PCA to find principal axes
        centered = pts_xy - centroid
        try:
            cov = np.cov(centered.T)
            if cov.ndim < 2:
                return None
            evals, evecs = np.linalg.eigh(cov)
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]

            # For TV, we WANT a linear structure (thin panel)
            # eigenratio close to 0 means linear
            eigenratio = evals[1] / (evals[0] + 1e-6)

            # Reject if too square (not TV-like)
            if eigenratio > 0.25:  # Allow some width but reject very square clusters
                print(f"[TVBelief] Rejected: eigenratio={eigenratio:.3f} (too square for TV)")
                return None

            angle = np.arctan2(evecs[1, 0], evecs[0, 0])
        except Exception as e:
            print(f"[TVBelief] Rejected: PCA failed - {e}")
            return None

        # Rotate points to principal axes to get dimensions
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rotated = np.column_stack([
            centered[:, 0] * cos_a - centered[:, 1] * sin_a,
            centered[:, 0] * sin_a + centered[:, 1] * cos_a
        ])

        min_xy, max_xy = np.min(rotated, axis=0), np.max(rotated, axis=0)
        dim1 = max(max_xy[0] - min_xy[0], 0.1)  # Along principal axis
        dim2 = max(max_xy[1] - min_xy[1], 0.03)  # Perpendicular

        # Width is the larger dimension
        width = max(dim1, dim2)

        # If dimensions were swapped, adjust angle
        if dim2 > dim1:
            angle = angle + np.pi/2
            while angle > np.pi: angle -= 2*np.pi
            while angle < -np.pi: angle += 2*np.pi

        # Validate width for TV range
        if width < config.min_width or width > config.max_width:
            print(f"[TVBelief] Rejected: width={width:.3f} out of range [{config.min_width}, {config.max_width}]")
            return None

        # Estimate height and z_base from Z values
        z_values = cluster[:, 2] if cluster.shape[1] >= 3 else np.array([config.prior_height])
        z_min = np.min(z_values)
        z_max = np.max(z_values)

        # z_base is the bottom of the TV (minimum Z of cluster)
        z_base = np.clip(z_min, config.min_z_base, config.max_z_base)

        # height is the vertical extent of the TV
        height = np.clip(z_max - z_min + 0.05, config.min_height, config.max_height)

        # DEBUG: Print initialization info (depth is fixed)
        print(f"[TVBelief] Created id={belief_id}: pos=({centroid[0]:.2f}, {centroid[1]:.2f}), "
              f"size=({width:.2f} x {height:.2f}), z_base={z_base:.2f}, θ={np.degrees(angle):.1f}°")

        # Initialize state [cx, cy, width, height, z_base, theta]
        mu = torch.tensor([centroid[0], centroid[1], width, height, z_base, angle],
                         dtype=DTYPE, device=device)

        # Initial covariance (6x6)
        Sigma = torch.diag(torch.tensor([
            config.initial_position_std**2,   # cx
            config.initial_position_std**2,   # cy
            config.initial_size_std**2,       # width
            config.initial_size_std**2,       # height
            config.initial_size_std**2,       # z_base
            config.initial_angle_std**2       # theta
        ], dtype=DTYPE, device=device))

        return cls(belief_id, mu, Sigma, config, config.initial_confidence)

    @staticmethod
    def _fit_rectangle_pca(points: np.ndarray, config) -> Optional[Tuple]:
        """Fit a rectangle to points using PCA. (Legacy method, kept for compatibility)"""
        if len(points) < 3:
            return None

        pts_xy = points[:, :2]
        centroid = np.mean(pts_xy, axis=0)
        centered = pts_xy - centroid

        try:
            cov = np.cov(centered.T)
            evals, evecs = np.linalg.eigh(cov)
            evals = np.sort(evals)[::-1]

            # For TV, we WANT a linear structure (thin panel)
            # Reject if it's too square
            if evals[0] > 0 and evals[1] / evals[0] > 0.3:  # Too square for a TV
                return None

            angle = np.arctan2(evecs[1, 1], evecs[0, 1])
        except:
            return None

        # Rotate points to principal axes
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rotated = np.column_stack([
            centered[:, 0] * cos_a - centered[:, 1] * sin_a,
            centered[:, 0] * sin_a + centered[:, 1] * cos_a
        ])

        min_xy, max_xy = np.min(rotated, axis=0), np.max(rotated, axis=0)
        w = max(max_xy[0] - min_xy[0], 0.1)  # Width (larger dimension)
        d = max(max_xy[1] - min_xy[1], 0.03)  # Depth (smaller dimension)

        return centroid[0], centroid[1], w, d, angle

    def _compute_edge_score(self, points: torch.Tensor) -> torch.Tensor:
        """Compute edge/corner score for points."""
        cx, cy = self.mu[0], self.mu[1]
        width, height, depth = self.mu[2], self.mu[3], self.mu[4]
        theta = self.mu[5]

        # Cache cos/sin
        theta_val = theta.item()
        if not hasattr(self, '_cached_theta') or self._cached_theta != theta_val:
            self._cached_theta = theta_val
            self._cached_cos = torch.cos(-theta)
            self._cached_sin = torch.sin(-theta)

        cos_t, sin_t = self._cached_cos, self._cached_sin

        px = points[:, 0] - cx
        py = points[:, 1] - cy

        local_x = px * cos_t - py * sin_t
        local_y = px * sin_t + py * cos_t
        local_z = points[:, 2] - height / 2

        half_w = width / 2
        half_d = depth / 2
        half_h = height / 2

        dist_x = torch.abs(torch.abs(local_x) - half_w)
        dist_y = torch.abs(torch.abs(local_y) - half_d)
        dist_z = torch.abs(torch.abs(local_z) - half_h)

        threshold = self.config.edge_proximity_threshold

        close_x = (dist_x < threshold).float()
        close_y = (dist_y < threshold).float()
        close_z = (dist_z < threshold).float()

        faces_close = close_x + close_y + close_z
        edge_score = (faces_close - 1).clamp(0, 2) / 2.0

        min_dist = torch.minimum(torch.minimum(dist_x, dist_y), dist_z)
        proximity_bonus = torch.exp(-min_dist / 0.02) * 0.2

        return (edge_score + proximity_bonus).clamp(0, 1)

    def add_historical_points(self, points_room, sdf_values, robot_cov):
        """Add points with low SDF to historical storage."""
        sdf_threshold = self.config.sdf_threshold_for_storage
        abs_sdf = torch.abs(sdf_values)
        mask = abs_sdf < sdf_threshold
        if not mask.any():
            return

        pts = points_room[mask]
        sdf = abs_sdf[mask]

        if pts.shape[1] == 2:
            pts = torch.cat([pts, torch.zeros(len(pts), 1, dtype=DTYPE, device=pts.device)], dim=1)
        elif pts.shape[1] > 3:
            pts = pts[:, :3]

        rob_cov = torch.tensor(robot_cov[:2, :2], dtype=DTYPE, device=self.mu.device)
        n = len(pts)
        I = torch.eye(2, dtype=DTYPE, device=self.mu.device)
        covs = rob_cov.unsqueeze(0).expand(n, -1, -1) + (self.config.beta_sdf * sdf**2).view(-1, 1, 1) * I.unsqueeze(0)
        rfe = torch.zeros(n, dtype=DTYPE, device=self.mu.device)
        self._add_to_bins(pts, covs, rfe)

    def _add_to_bins(self, pts, covs, rfe):
        """Add points to angular/height bins for uniform surface coverage."""
        cx, cy = self.mu[0].item(), self.mu[1].item()
        na = self.config.num_angle_bins
        nz = self.config.num_z_bins
        mpb = self.config.max_historical_points // (na * nz) + 1

        edge_scores = self._compute_edge_score(pts)
        edge_bonus_weight = self.config.edge_bonus_weight

        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        angles = torch.atan2(dy, dx)
        zb = (pts[:, 2] / 0.1).long().clamp(0, nz - 1)
        ab = ((angles + np.pi) / (2 * np.pi) * na).long() % na
        bidx = ab * nz + zb
        bins = {}

        if len(self.historical_points) > 0:
            existing_edge_scores = self._compute_edge_score(self.historical_points)
            for i in range(len(self.historical_points)):
                p, c, r = self.historical_points[i], self.historical_capture_covs[i], self.historical_rfe[i].item()
                dxh, dyh = p[0] - cx, p[1] - cy
                ah = torch.atan2(dyh, dxh)
                zbi = int((p[2] / 0.1).clamp(0, nz - 1).item())
                abi = int(((ah + np.pi) / (2 * np.pi) * na).clamp(0, na - 1).item())
                idx = abi * nz + zbi
                if idx not in bins:
                    bins[idx] = []
                edge_bonus = edge_bonus_weight * existing_edge_scores[i].item()
                quality = torch.trace(c).item() + r - edge_bonus
                bins[idx].append((p, c, r, quality))

        for i in range(len(pts)):
            idx = bidx[i].item()
            if idx not in bins:
                bins[idx] = []
            edge_bonus = edge_bonus_weight * edge_scores[i].item()
            quality = torch.trace(covs[i]).item() + rfe[i].item() - edge_bonus
            bins[idx].append((pts[i], covs[i], rfe[i].item(), quality))

        fp, fc, fr = [], [], []
        for contents in bins.values():
            for p, c, r, _ in sorted(contents, key=lambda x: x[3])[:mpb]:
                fp.append(p)
                fc.append(c)
                fr.append(r)

        if fp:
            self.historical_points = torch.stack(fp)
            self.historical_capture_covs = torch.stack(fc)
            self.historical_rfe = torch.tensor(fr, dtype=DTYPE, device=self.mu.device)

    def update_rfe(self, robot_cov: np.ndarray):
        """Update Remembered Free Energy for historical points."""
        if len(self.historical_points) == 0:
            return

        alpha = self.config.rfe_alpha
        max_rfe = self.config.rfe_max_threshold

        sdf_values = self.sdf(self.historical_points)
        sdf_squared = sdf_values ** 2

        robot_trace = np.trace(robot_cov)
        w_t = 1.0 / (1.0 + robot_trace)

        self.historical_rfe = alpha * self.historical_rfe + w_t * sdf_squared.detach()

        good_mask = self.historical_rfe < max_rfe
        if good_mask.sum() < len(self.historical_points):
            self.historical_points = self.historical_points[good_mask]
            self.historical_capture_covs = self.historical_capture_covs[good_mask]
            self.historical_rfe = self.historical_rfe[good_mask]

    def get_historical_points_in_robot_frame(self, robot_pose, robot_cov):
        """Transform historical points to robot frame with uncertainty."""
        if len(self.historical_points) == 0:
            return torch.empty((0, 3), dtype=DTYPE, device=self.mu.device), \
                   torch.empty(0, dtype=DTYPE, device=self.mu.device)

        rx, ry, rth = robot_pose
        ct, st = np.cos(-rth), np.sin(-rth)
        px = self.historical_points[:, 0] - rx
        py = self.historical_points[:, 1] - ry
        pts = torch.stack([px * ct - py * st, px * st + py * ct, self.historical_points[:, 2]], dim=1)

        rc = torch.tensor(robot_cov, dtype=DTYPE, device=self.mu.device)
        w = torch.zeros(len(self.historical_points), dtype=DTYPE, device=self.mu.device)

        for i in range(len(self.historical_points)):
            J = torch.tensor(
                compute_jacobian_room_to_robot(self.historical_points[i].cpu().numpy(), robot_pose),
                dtype=DTYPE, device=self.mu.device
            )
            st = self.historical_capture_covs[i] + J @ rc @ J.T
            w[i] = 1.0 / (1.0 + torch.trace(st) + self.historical_rfe[i])

        return pts, w


