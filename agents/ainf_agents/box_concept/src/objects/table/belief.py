#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Table Belief - Belief class for table objects.

State: [cx, cy, w, h, table_height, leg_length, theta]
- cx, cy: center position in XY plane
- w, h: width and depth of table top
- table_height: height of table surface from floor
- leg_length: length of legs
- theta: rotation angle around Z axis
"""

import numpy as np
import torch
from typing import Optional, Tuple
from dataclasses import dataclass

from src.belief_core import Belief, BeliefConfig, DEVICE, DTYPE
from src.objects.table.sdf import compute_table_sdf, TABLE_TOP_THICKNESS, TABLE_LEG_RADIUS
from src.transforms import compute_jacobian_room_to_robot


@dataclass
class TableBeliefConfig(BeliefConfig):
    """Configuration for table beliefs."""

    # Size priors - tables are larger than chairs
    prior_table_width: float = 1.0   # Standard table width
    prior_table_depth: float = 0.6   # Standard table depth
    prior_table_height: float = 0.75 # Standard table height from floor
    prior_size_std: float = 0.15     # Reduced variance (tighter prior)
    min_aspect_ratio: float = 0.3    # Tables can be more elongated

    # Size limits for tables (override base class)
    min_size: float = 0.50           # Tables are at least 50cm (smaller is likely a chair)
    max_size: float = 2.5            # Tables can be up to 2.5m

    # Clustering
    cluster_eps: float = 0.30        # Larger clustering for bigger objects
    min_cluster_points: int = 20     # More points expected for larger objects

    # Association
    max_association_cost: float = 5.0
    max_association_distance: float = 1.0
    wall_margin: float = 0.30

    # Historical points storage
    max_historical_points: int = 600
    sdf_threshold_for_storage: float = 0.08
    beta_sdf: float = 1.0

    # Slow accumulation of historical points (same as chair)
    max_new_points_per_frame: int = 5      # Max points to add per frame
    min_frames_before_historical: int = 10  # Wait N frames before adding historical points
    historical_warmup_frames: int = 50      # Gradually increase allowed points over this period

    # Binning for uniform surface coverage
    num_angle_bins: int = 24
    num_z_bins: int = 10
    edge_bonus_weight: float = 0.3
    edge_proximity_threshold: float = 0.05

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

    # Angle alignment prior
    angle_alignment_weight: float = 0.5
    angle_alignment_sigma: float = 0.1


class TableBelief(Belief):
    """Gaussian belief over table parameters [cx, cy, w, h, table_height, leg_length, theta]."""

    STATE_DIM = 7

    @property
    def state_dim(self) -> int:
        return self.STATE_DIM

    @property
    def position(self) -> Tuple[float, float]:
        return self.mu[0].item(), self.mu[1].item()

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
    def depth(self) -> float:
        return self.mu[3].item()

    @property
    def table_height(self) -> float:
        return self.mu[4].item()

    @property
    def leg_length(self) -> float:
        return self.mu[5].item()

    @property
    def angle(self) -> float:
        return self.mu[6].item()

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        assert points.shape[1] == 3, f"Points must be 3D, got {points.shape}"
        return compute_table_sdf(points, self.mu)

    def compute_prior_term(self, mu: torch.Tensor, robot_pose: np.ndarray = None) -> torch.Tensor:
        """
        Compute table-specific prior energy term.

        Since the table is static, the prior is simply that the state should not
        change much from the previous estimate. This acts as a soft regularizer.

        From OBJECT_INFERENCE_MATH.md:
        F_prior = (λ/2) × ||s - s_prev||²

        Args:
            mu: Current state estimate [7] in robot frame
            robot_pose: Robot pose [x, y, theta] for frame transformation

        Returns:
            Prior energy term (scalar tensor)
        """
        if self.mu is None or robot_pose is None:
            return torch.tensor(0.0, dtype=mu.dtype, device=mu.device)

        # Import here to avoid circular imports
        from src.transforms import transform_object_to_robot_frame

        # Transform self.mu (room frame) to robot frame for comparison
        mu_prev_robot = transform_object_to_robot_frame(self.mu, robot_pose)

        # =================================================================
        # STATE PRIOR: penalize deviation from previous state (static object)
        # Increased from 0.01/0.005 to 0.05/0.02 (2026-02-06)
        # =================================================================
        lambda_pos = 0.05     # Position regularization
        lambda_size = 0.02    # Size regularization
        lambda_angle = 0.01   # Angle regularization (very weak)

        # Position difference (cx, cy)
        diff_pos = mu[:2] - mu_prev_robot[:2]
        prior_pos = lambda_pos * torch.sum(diff_pos ** 2)

        # Size differences (w, h, table_height, leg_length)
        diff_size = mu[2:6] - mu_prev_robot[2:6]
        prior_size = lambda_size * torch.sum(diff_size ** 2)

        # Angle difference (normalized to [-pi, pi])
        diff_angle = mu[6] - mu_prev_robot[6]
        diff_angle = torch.atan2(torch.sin(diff_angle), torch.cos(diff_angle))
        prior_angle = lambda_angle * (diff_angle ** 2)

        total_prior = prior_pos + prior_size + prior_angle

        return total_prior

    def apply_constraints(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Apply table-specific physical constraints to the state vector.

        CRITICAL: leg_length (index 5) must not exceed table_height (index 4) - TOP_THICKNESS.
        Legs go from floor (z=0) up to the bottom of the table top.

        State: [cx, cy, w, h, table_height, leg_length, theta]
        """
        table_height = mu[4]
        max_leg_length = table_height - TABLE_TOP_THICKNESS

        # Ensure leg_length is in valid range [0.05, table_height - TOP_THICKNESS]
        mu[5] = torch.clamp(mu[5], min=0.05, max=max_leg_length.item())

        return mu

    def _get_process_noise_variances(self) -> list:
        cfg = self.config
        return [
            cfg.sigma_process_xy**2,      # cx
            cfg.sigma_process_xy**2,      # cy
            cfg.sigma_process_size**2,    # w
            cfg.sigma_process_size**2,    # h
            cfg.sigma_process_size**2,    # table_height
            cfg.sigma_process_size**2,    # leg_length
            cfg.sigma_process_angle**2    # theta
        ]

    @classmethod
    def from_cluster(cls, belief_id: int, cluster: np.ndarray,
                     config: TableBeliefConfig, device: torch.device = DEVICE) -> Optional['TableBelief']:
        """Create a table belief from a point cluster."""

        if len(cluster) < config.min_cluster_points:
            return None

        # Fit rectangle using PCA for position and orientation
        result = cls._fit_rectangle_pca(cluster, config)
        if result is None:
            return None

        cx, cy, w, h, angle = result

        # Validate dimensions
        if w < config.min_size or h < config.min_size:
            return None
        if w > config.max_size or h > config.max_size:
            return None
        if min(w, h) / max(w, h) < config.min_aspect_ratio:
            return None

        # Estimate table height from cluster Z values
        # For a table, points are mostly on the top surface
        z_values = cluster[:, 2] if cluster.shape[1] >= 3 else np.array([config.prior_table_height])
        table_height = np.percentile(z_values, 90)  # Top surface
        table_height = np.clip(table_height, 0.4, 1.2)  # Reasonable table height range

        # Leg length is table_height minus top thickness
        leg_length = max(table_height - TABLE_TOP_THICKNESS, 0.1)

        # Create state vector
        mu = torch.tensor([cx, cy, w, h, table_height, leg_length, angle],
                         dtype=DTYPE, device=device)

        # Create covariance matrix
        Sigma = torch.diag(torch.tensor([
            config.initial_position_std**2,  # cx
            config.initial_position_std**2,  # cy
            config.initial_size_std**2,      # w
            config.initial_size_std**2,      # h
            config.initial_size_std**2,      # table_height
            config.initial_size_std**2,      # leg_length
            config.initial_angle_std**2      # theta
        ], dtype=DTYPE, device=device))

        return cls(belief_id, mu, Sigma, config, config.initial_confidence)

    @staticmethod
    def _fit_rectangle_pca(points: np.ndarray, config) -> Optional[Tuple]:
        """Fit a rectangle to points using PCA."""
        if len(points) < 3:
            return None

        pts_xy = points[:, :2]
        centroid = np.mean(pts_xy, axis=0)
        centered = pts_xy - centroid

        try:
            cov = np.cov(centered.T)
            evals, evecs = np.linalg.eigh(cov)
            evals = np.sort(evals)[::-1]

            # Reject if too linear (not enough 2D spread for a table)
            if evals[0] > 0 and evals[1] / evals[0] < 0.1:
                return None

            angle = np.arctan2(evecs[1, 1], evecs[0, 1])
            while angle > np.pi/2: angle -= np.pi
            while angle < -np.pi/2: angle += np.pi
        except:
            angle = 0.0

        # Rotate points to aligned frame
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rotated = np.column_stack([
            centered[:, 0] * cos_a - centered[:, 1] * sin_a,
            centered[:, 0] * sin_a + centered[:, 1] * cos_a
        ])

        min_xy, max_xy = np.min(rotated, axis=0), np.max(rotated, axis=0)
        w = max(max_xy[0] - min_xy[0], 0.2)
        h = max(max_xy[1] - min_xy[1], 0.2)

        return centroid[0], centroid[1], w, h, angle

    def add_historical_points(self, points_room, sdf_values, robot_cov):
        """Add points with low SDF to historical storage.

        Points are added slowly to allow the pose to converge first:
        - Wait min_frames_before_historical frames before adding any
        - Gradually increase max points per frame during warmup period
        - Limit total points added per frame
        """
        # Don't add points until we have enough observations
        if self.observation_count < self.config.min_frames_before_historical:
            return

        # Calculate how many points we can add this frame (gradual warmup)
        warmup_progress = min(1.0, (self.observation_count - self.config.min_frames_before_historical)
                              / self.config.historical_warmup_frames)
        max_points_this_frame = max(1, int(warmup_progress * self.config.max_new_points_per_frame))

        sdf_threshold = self.config.sdf_threshold_for_storage
        mask = torch.abs(sdf_values) < sdf_threshold
        if not mask.any():
            return

        pts = points_room[mask]
        sdf = torch.abs(sdf_values[mask])

        # Limit number of new points
        if len(pts) > max_points_this_frame:
            # Select points with lowest SDF (best quality)
            _, indices = torch.topk(sdf, max_points_this_frame, largest=False)
            pts = pts[indices]
            sdf = sdf[indices]

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

    def _compute_edge_score(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute edge/corner score for points based on proximity to multiple faces.

        Points near edges/corners have high geometric information content.

        Returns:
            [N] scores in [0, 1], higher = more edge/corner-like
        """
        # Transform points to table-local frame
        cx, cy = self.mu[0], self.mu[1]
        w, h = self.mu[2], self.mu[3]
        table_height = self.mu[4]
        theta = self.mu[6]

        cos_t = torch.cos(-theta)
        sin_t = torch.sin(-theta)

        px = points[:, 0] - cx
        py = points[:, 1] - cy

        local_x = px * cos_t - py * sin_t
        local_y = px * sin_t + py * cos_t
        local_z = points[:, 2]

        # Distance to each face of table top
        half_w, half_h = w / 2, h / 2
        half_t = TABLE_TOP_THICKNESS / 2
        top_center_z = table_height - half_t

        dist_x = torch.abs(torch.abs(local_x) - half_w)
        dist_y = torch.abs(torch.abs(local_y) - half_h)
        dist_z = torch.abs(local_z - top_center_z)

        # Threshold for "close to face"
        threshold = self.config.edge_proximity_threshold

        close_x = (dist_x < threshold).float()
        close_y = (dist_y < threshold).float()
        close_z = (dist_z < half_t + threshold).float()  # Near top surface

        faces_close = close_x + close_y + close_z

        # Score: 0 = flat face, 0.5 = edge (2 faces), 1.0 = corner (3 faces)
        edge_score = (faces_close - 1).clamp(0, 2) / 2.0

        # Proximity bonus for being very close to edges
        min_dist = torch.minimum(dist_x, dist_y)
        proximity_bonus = torch.exp(-min_dist / 0.02) * 0.2

        return (edge_score + proximity_bonus).clamp(0, 1)

    def _add_to_bins(self, pts, covs, rfe):
        """
        Add points to angular/height bins for uniform surface coverage.

        Points are sorted by quality = trace(Σ) + RFE - edge_bonus
        Edge/corner points get priority (lower quality score = better).
        """
        cx, cy = self.mu[0].item(), self.mu[1].item()
        na = self.config.num_angle_bins
        nz = self.config.num_z_bins
        mpb = self.config.max_historical_points // (na * nz) + 1

        # Compute edge scores for new points (edge/corner priority)
        edge_scores = self._compute_edge_score(pts)
        edge_bonus_weight = self.config.edge_bonus_weight

        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        angles = torch.atan2(dy, dx)
        zb = (pts[:, 2] / 0.1).long().clamp(0, nz - 1)
        ab = ((angles + np.pi) / (2 * np.pi) * na).long() % na
        bidx = ab * nz + zb
        bins = {}

        # Add existing historical points (with their edge scores)
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
                # Quality: lower is better. Edge bonus REDUCES quality score (priority)
                edge_bonus = edge_bonus_weight * existing_edge_scores[i].item()
                quality = torch.trace(c).item() + r - edge_bonus
                bins[idx].append((p, c, r, quality))

        # Add new points (with edge scores)
        for i in range(len(pts)):
            idx = bidx[i].item()
            if idx not in bins:
                bins[idx] = []
            edge_bonus = edge_bonus_weight * edge_scores[i].item()
            quality = torch.trace(covs[i]).item() + rfe[i].item() - edge_bonus
            bins[idx].append((pts[i], covs[i], rfe[i].item(), quality))

        # Keep best points per bin
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
        """Get historical points transformed to robot frame with weights."""
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
            # J is [2x3], rc is [3x3], so J @ rc @ J.T gives [2x2]
            cov_from_robot = J @ rc @ J.T
            st = self.historical_capture_covs[i] + cov_from_robot
            w[i] = 1.0 / (1.0 + torch.trace(st) + self.historical_rfe[i])

        return pts, w

    def to_dict(self):
        """Convert belief to dictionary for visualization."""
        d = super().to_dict()

        # Compute RFE statistics for historical points
        rfe_stats = {}
        if len(self.historical_points) > 0:
            rfe = self.historical_rfe
            rfe_stats = {
                'mean': rfe.mean().item(),
                'min': rfe.min().item(),
                'max': rfe.max().item(),
                'trusted': (rfe < self.config.rfe_trusted_threshold).sum().item(),
                'good': ((rfe >= self.config.rfe_trusted_threshold) & (rfe < self.config.rfe_good_threshold)).sum().item(),
                'moderate': ((rfe >= self.config.rfe_good_threshold) & (rfe < self.config.rfe_moderate_threshold)).sum().item(),
                'unreliable': (rfe >= self.config.rfe_moderate_threshold).sum().item(),
            }

        d.update({
            'type': 'table',
            'cx': self.cx,
            'cy': self.cy,
            'width': self.width,
            'depth': self.depth,
            'table_height': self.table_height,
            'leg_length': self.leg_length,
            'angle': self.angle,
            'top_thickness': TABLE_TOP_THICKNESS,
            'leg_radius': TABLE_LEG_RADIUS,
            'num_historical_points': len(self.historical_points),
            'historical_rfe_stats': rfe_stats
        })
        return d
