#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Chair Belief - Belief class for chair objects.

State: [cx, cy, seat_w, seat_d, seat_h, back_h, theta]
- cx, cy: center position in XY plane
- seat_w: seat width
- seat_d: seat depth
- seat_h: seat height from floor
- back_h: backrest height above seat
- theta: rotation angle around Z axis

Fixed constants (in object_sdf_prior.py):
- CHAIR_SEAT_THICKNESS = 0.05m
- CHAIR_BACK_THICKNESS = 0.05m
"""

import numpy as np
import torch
from typing import Optional, Tuple
from dataclasses import dataclass

from src.belief_core import Belief, BeliefConfig, DEVICE, DTYPE
from src.objects.chair.sdf import compute_chair_sdf, CHAIR_SEAT_THICKNESS, CHAIR_BACK_THICKNESS
from src.transforms import compute_jacobian_room_to_robot


@dataclass
class ChairBeliefConfig(BeliefConfig):
    """Configuration for chair beliefs."""

    # Size priors
    prior_seat_width: float = 0.45
    prior_seat_depth: float = 0.45
    prior_seat_height: float = 0.45
    prior_back_height: float = 0.40
    prior_size_std: float = 0.1
    min_aspect_ratio: float = 0.5

    # Clustering
    cluster_eps: float = 0.25
    min_cluster_points: int = 20

    # Association
    max_association_cost: float = 5.0
    max_association_distance: float = 0.8
    wall_margin: float = 0.25

    # Historical points storage
    max_historical_points: int = 500
    sdf_threshold_for_storage: float = 0.06
    beta_sdf: float = 1.0

    # Slow accumulation of historical points
    max_new_points_per_frame: int = 5      # Max points to add per frame
    min_frames_before_historical: int = 10  # Wait N frames before adding historical points
    historical_warmup_frames: int = 50      # Gradually increase allowed points over this period

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

    # Angle alignment prior (encourage axis-aligned orientations)
    # Following the paper: E_theta = (lambda_theta/2) * min_k (theta - mu_k)^2
    # where mu_k in {0, pi/2, -pi/2, pi} and lambda_theta = 1/sigma^2
    angle_alignment_weight: float = 0.5  # gamma in the paper
    angle_alignment_sigma: float = 0.1   # ~5.7 degrees tolerance

    # Temporal continuity prior (prevents sudden angle flips)
    angle_continuity_weight: float = 2.0   # Moderate penalty for angle changes
    angle_continuity_sigma: float = 0.15   # ~8.6 degrees tolerance


class ChairBelief(Belief):
    """Gaussian belief over chair parameters [cx, cy, seat_w, seat_d, seat_h, back_h, theta]."""

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
    def seat_width(self) -> float:
        return self.mu[2].item()

    @property
    def seat_depth(self) -> float:
        return self.mu[3].item()

    @property
    def seat_height(self) -> float:
        return self.mu[4].item()

    @property
    def back_height(self) -> float:
        return self.mu[5].item()

    @property
    def angle(self) -> float:
        return self.mu[6].item()


    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        assert points.shape[1] == 3, f"Points must be 3D, got {points.shape}"
        return compute_chair_sdf(points, self.mu)

    def compute_prior_term(self, mu: torch.Tensor, robot_pose: np.ndarray = None) -> torch.Tensor:
        """
        Compute chair-specific prior energy term.

        Since the chair is static, the prior is simply that the state should not
        change much from the previous estimate. This acts as a soft regularizer.

        NOTE: Currently disabled to debug 90° flip issue.
        The problem is that mu is in robot frame but self.mu is in room frame.

        Args:
            mu: Current state estimate (in robot frame)
            robot_pose: Robot pose [x, y, theta] for frame transformation

        Returns:
            Prior energy term (scalar tensor)
        """
        # DISABLED - frame mismatch between mu (robot) and self.mu (room) causes 90° flip
        return torch.tensor(0.0, dtype=mu.dtype, device=mu.device)

    def _get_process_noise_variances(self) -> list:
        cfg = self.config
        return [
            cfg.sigma_process_xy**2,      # cx
            cfg.sigma_process_xy**2,      # cy
            cfg.sigma_process_size**2,    # seat_w
            cfg.sigma_process_size**2,    # seat_d
            cfg.sigma_process_size**2,    # seat_h
            cfg.sigma_process_size**2,    # back_h
            cfg.sigma_process_angle**2    # theta
        ]

    @classmethod
    def from_cluster(cls, belief_id: int, cluster: np.ndarray,
                     config: ChairBeliefConfig, device: torch.device = DEVICE) -> Optional['ChairBelief']:
        """Create a chair belief from a point cluster."""

        if len(cluster) < config.min_cluster_points:
            return None

        # Fit rectangle using PCA for position and orientation
        result = cls._fit_rectangle_pca(cluster, config)
        if result is None:
            return None

        cx, cy, w, d, angle = result

        # Validate dimensions
        if w < config.min_size or d < config.min_size:
            return None
        if w > config.max_size or d > config.max_size:
            return None

        # Estimate chair dimensions from cluster
        z_values = cluster[:, 2] if cluster.shape[1] >= 3 else np.array([config.prior_seat_height])

        # Seat height: lower part of the cluster
        seat_h = np.percentile(z_values, 30)
        seat_h = np.clip(seat_h, 0.35, 0.55)

        # Backrest height: difference between top and seat
        max_z = np.percentile(z_values, 95)
        back_h = max(max_z - seat_h, 0.2)
        back_h = np.clip(back_h, 0.2, 0.6)

        # Use priors for seat dimensions
        seat_w = np.clip(w, 0.3, 0.6)
        seat_d = np.clip(d, 0.3, 0.6)

        # =================================================================
        # TEST MULTIPLE ANGLES AND PICK THE ONE WITH LOWEST SDF
        # This is necessary because the SDF is nearly symmetric for square seats
        # Test every 45 degrees for better coverage
        # =================================================================
        cluster_tensor = torch.tensor(cluster[:, :3] if cluster.shape[1] >= 3 else
                                       np.column_stack([cluster, np.zeros(len(cluster))]),
                                       dtype=DTYPE, device=device)

        # Test every 45 degrees: 0°, 45°, 90°, 135°, 180°, -135°, -90°, -45°
        test_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]
        best_angle = angle
        best_sdf = float('inf')

        for test_angle in test_angles:
            test_params = torch.tensor([cx, cy, seat_w, seat_d, seat_h, back_h, test_angle],
                                        dtype=DTYPE, device=device)
            sdf_values = compute_chair_sdf(cluster_tensor, test_params)
            mean_sdf_sq = torch.mean(sdf_values**2).item()

            if mean_sdf_sq < best_sdf:
                best_sdf = mean_sdf_sq
                best_angle = test_angle

        print(f"[INIT] Tested 8 angles: best={np.degrees(best_angle):.0f}° (SDF²={best_sdf:.6f})")

        # Create state vector (7 parameters)
        mu = torch.tensor([cx, cy, seat_w, seat_d, seat_h, back_h, best_angle],
                         dtype=DTYPE, device=device)

        # Create covariance matrix
        Sigma = torch.diag(torch.tensor([
            config.initial_position_std**2,  # cx
            config.initial_position_std**2,  # cy
            config.initial_size_std**2,      # seat_w
            config.initial_size_std**2,      # seat_d
            config.initial_size_std**2,      # seat_h
            config.initial_size_std**2,      # back_h
            config.initial_angle_std**2      # theta
        ], dtype=DTYPE, device=device))

        return cls(belief_id, mu, Sigma, config, config.initial_confidence)

    @staticmethod
    def _fit_rectangle_pca(points: np.ndarray, config) -> Optional[Tuple]:
        """Fit a rectangle to points using PCA, orienting so +Y points to backrest."""
        if len(points) < 3:
            return None

        pts_xy = points[:, :2]
        centroid = np.mean(pts_xy, axis=0)
        centered = pts_xy - centroid

        try:
            cov = np.cov(centered.T)
            evals, evecs = np.linalg.eigh(cov)
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]

            if evals[0] > 0 and evals[1] / evals[0] < 0.1:
                return None

            # Initial angle from first principal component
            angle = np.arctan2(evecs[1, 0], evecs[0, 0])
        except:
            angle = 0.0

        # Rotate points to local frame
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rotated = np.column_stack([
            centered[:, 0] * cos_a - centered[:, 1] * sin_a,
            centered[:, 0] * sin_a + centered[:, 1] * cos_a
        ])

        min_xy, max_xy = np.min(rotated, axis=0), np.max(rotated, axis=0)
        w = max(max_xy[0] - min_xy[0], 0.2)
        d = max(max_xy[1] - min_xy[1], 0.2)

        # =========================================================
        # DETECT BACKREST SIDE using height distribution
        # The backrest has higher Z values than the seat
        # We want +Y to point TOWARD the backrest (backrest at +Y)
        # =========================================================
        backrest_detected = False
        if points.shape[1] >= 3:
            z_values = points[:, 2]
            z_threshold = np.percentile(z_values, 70)  # Higher threshold for backrest
            high_mask = z_values > z_threshold

            if high_mask.sum() > 5:  # Need enough high points
                # Get mean Y position of high points in local frame
                high_points_local_y = rotated[high_mask, 1]
                mean_high_y = np.mean(high_points_local_y)

                # Get mean Y of low points for comparison
                low_mask = z_values < np.percentile(z_values, 30)
                if low_mask.sum() > 5:
                    low_points_local_y = rotated[low_mask, 1]
                    mean_low_y = np.mean(low_points_local_y)

                    # Backrest should be clearly on one side
                    y_diff = mean_high_y - mean_low_y

                    # If high points are on -Y side (y_diff < 0), rotate 180°
                    # so backrest ends up at +Y
                    if y_diff < -0.02:  # 2cm threshold
                        angle += np.pi
                        backrest_detected = True
                    elif y_diff > 0.02:
                        backrest_detected = True
                        # Keep angle as-is, backrest is already on +Y

        # Normalize angle to [-pi, pi]
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi

        # Snap angle to nearest 90 degree multiple (0, 90, 180, -90)
        # This gives a good initial guess before testing all 4 angles
        angle_90 = np.round(angle / (np.pi / 2)) * (np.pi / 2)
        if angle_90 > np.pi:
            angle_90 -= 2 * np.pi
        elif angle_90 < -np.pi:
            angle_90 += 2 * np.pi

        angle = angle_90

        return centroid[0], centroid[1], w, d, angle

    def _compute_edge_score(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute edge/corner score for points.

        Returns:
            [N] scores in [0, 1], higher = more edge/corner-like
        """
        cx, cy = self.mu[0], self.mu[1]
        seat_w, seat_d = self.mu[2], self.mu[3]
        seat_h = self.mu[4]
        theta = self.mu[6]

        cos_t = torch.cos(-theta)
        sin_t = torch.sin(-theta)

        px = points[:, 0] - cx
        py = points[:, 1] - cy

        local_x = px * cos_t - py * sin_t
        local_y = px * sin_t + py * cos_t
        local_z = points[:, 2]

        # Distance to seat faces
        half_w, half_d = seat_w / 2, seat_d / 2
        half_t = CHAIR_SEAT_THICKNESS / 2
        seat_center_z = seat_h - half_t

        dist_x = torch.abs(torch.abs(local_x) - half_w)
        dist_y = torch.abs(torch.abs(local_y) - half_d)
        dist_z = torch.abs(local_z - seat_center_z)

        threshold = self.config.edge_proximity_threshold

        close_x = (dist_x < threshold).float()
        close_y = (dist_y < threshold).float()
        close_z = (dist_z < half_t + threshold).float()

        faces_close = close_x + close_y + close_z
        edge_score = (faces_close - 1).clamp(0, 2) / 2.0

        min_dist = torch.minimum(dist_x, dist_y)
        proximity_bonus = torch.exp(-min_dist / 0.02) * 0.2

        return (edge_score + proximity_bonus).clamp(0, 1)

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

        # Add existing historical points
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

        # Add new points
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
            cov_from_robot = J @ rc @ J.T
            st = self.historical_capture_covs[i] + cov_from_robot
            w[i] = 1.0 / (1.0 + torch.trace(st) + self.historical_rfe[i])

        return pts, w

    def to_dict(self):
        """Convert belief to dictionary for visualization."""
        d = super().to_dict()

        # Compute RFE statistics
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
            'type': 'chair',
            'cx': self.cx,
            'cy': self.cy,
            'seat_width': self.seat_width,
            'seat_depth': self.seat_depth,
            'seat_height': self.seat_height,
            'back_height': self.back_height,
            'back_thickness': CHAIR_BACK_THICKNESS,  # Fixed constant
            'angle': self.angle,
            'seat_thickness': CHAIR_SEAT_THICKNESS,
            'num_historical_points': len(self.historical_points),
            'historical_rfe_stats': rfe_stats
        })
        return d
