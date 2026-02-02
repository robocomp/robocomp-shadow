#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Box Belief - Belief class for 3D box obstacles. State: [cx, cy, w, h, d, theta]"""
import numpy as np
import torch
from typing import Optional, Tuple
from dataclasses import dataclass
from src.belief_core import Belief, BeliefConfig, DEVICE, DTYPE
from src.object_sdf_prior import compute_box_sdf, compute_box_priors
from src.transforms import compute_jacobian_room_to_robot
@dataclass
class BoxBeliefConfig(BeliefConfig):
    """Configuration for 3D box beliefs."""
    # Size priors
    prior_size_mean: float = 0.5
    prior_size_std: float = 0.2
    min_aspect_ratio: float = 0.35

    # Clustering
    cluster_eps: float = 0.25
    min_cluster_points: int = 12

    # Association
    max_association_cost: float = 5.0
    max_association_distance: float = 0.8
    wall_margin: float = 0.30

    # Historical points storage
    max_historical_points: int = 500
    sdf_threshold_for_storage: float = 0.07  # Max SDF to accept point (meters)
    beta_sdf: float = 1.0  # Weight for SDF in capture covariance

    # Binning for uniform surface coverage
    num_angle_bins: int = 24
    num_z_bins: int = 8
    edge_bonus_weight: float = 0.3  # Quality bonus for edge/corner points
    edge_proximity_threshold: float = 0.05  # Distance to consider "close to face" (meters)

    # RFE (Remembered Free Energy) parameters
    rfe_alpha: float = 0.98  # Temporal decay (higher = slower decay, preserves history)
    rfe_max_threshold: float = 2.0  # Points with RFE above this are pruned

    # RFE classification thresholds for statistics
    rfe_trusted_threshold: float = 0.03  # Below this = trusted
    rfe_good_threshold: float = 0.1  # Below this = good
    rfe_moderate_threshold: float = 1.0  # Below this = moderate, above = unreliable

    # Edge/corner classification thresholds
    corner_score_threshold: float = 0.7  # Above this = corner
    edge_score_threshold: float = 0.3  # Above this = edge

    # Angle alignment prior (boxes tend to align with room axes)
    angle_alignment_weight: float = 0.5  # Weight for alignment prior (0 = disabled)
    angle_alignment_sigma: float = 0.1  # Std dev around aligned angles (radians, ~6°)
class BoxBelief(Belief):
    """Gaussian belief over 3D box parameters [cx, cy, w, h, d, theta]."""
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
        return self.mu[4].item()
    @property
    def angle(self) -> float:
        return self.mu[5].item()
    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        assert points.shape[1] == 3, f"Points must be 3D, got {points.shape}"
        return compute_box_sdf(points, self.mu)

    def compute_angle_alignment_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute prior term for angle alignment with room axes.

        The prior is a mixture of Gaussians centered at aligned angles (0°, ±90°).
        Each Gaussian has:
        - Mean μ_k = k * π/2 (aligned angles)
        - Precision λ = 1/σ² (inverse variance)

        Prior energy: E(θ) = (λ/2) * min_k((θ - μ_k)²)

        This returns the prior term to be added to the VFE loss.
        When θ is aligned (0° or 90°), the term is zero.

        Args:
            theta: Angle in radians (normalized to [-π/2, π/2])

        Returns:
            Prior energy term (0 when aligned, positive otherwise)
        """
        sigma = self.config.angle_alignment_sigma  # Standard deviation
        precision = 1.0 / (sigma ** 2)  # λ = 1/σ²

        # Distance to nearest aligned angle (0 or ±π/2)
        # Since theta is in [-π/2, π/2], we check:
        # - Distance to 0 (axis-aligned)
        # - Distance to π/2 or -π/2 (90° rotated)
        dist_to_0 = theta ** 2
        dist_to_pos90 = (theta - np.pi/2) ** 2
        dist_to_neg90 = (theta + np.pi/2) ** 2

        # Minimum squared distance to any aligned angle
        min_dist_sq = torch.minimum(dist_to_0, torch.minimum(dist_to_pos90, dist_to_neg90))

        # Prior energy: (λ/2) * (θ - μ_nearest)²
        # This is zero when aligned, and increases with deviation
        prior_energy = 0.5 * precision * min_dist_sq

        return prior_energy
    def _get_process_noise_variances(self) -> list:
        cfg = self.config
        return [cfg.sigma_process_xy**2, cfg.sigma_process_xy**2,
                cfg.sigma_process_size**2, cfg.sigma_process_size**2,
                cfg.sigma_process_size**2, cfg.sigma_process_angle**2]
    @classmethod
    def from_cluster(cls, belief_id: int, cluster: np.ndarray, 
                     config: BoxBeliefConfig, device: torch.device = DEVICE) -> Optional['BoxBelief']:
        if len(cluster) < config.min_cluster_points:
            return None
        result = cls._fit_rectangle_pca(cluster, config)
        if result is None:
            return None
        cx, cy, w, h, angle = result
        if w < config.min_size or h < config.min_size or w > config.max_size or h > config.max_size:
            return None
        if min(w, h) / max(w, h) < config.min_aspect_ratio:
            return None
        d = max(np.max(cluster[:, 2]), config.min_size) if cluster.shape[1] >= 3 else config.prior_size_mean
        d = min(d, config.max_size)
        mu = torch.tensor([cx, cy, w, h, d, angle], dtype=DTYPE, device=device)
        Sigma = torch.diag(torch.tensor([config.initial_position_std**2]*2 + [config.initial_size_std**2]*3 + [config.initial_angle_std**2], dtype=DTYPE, device=device))
        return cls(belief_id, mu, Sigma, config, config.initial_confidence)
    @staticmethod
    def _fit_rectangle_pca(points: np.ndarray, config) -> Optional[Tuple]:
        if len(points) < 3:
            return None
        pts_xy = points[:, :2]
        centroid = np.mean(pts_xy, axis=0)
        centered = pts_xy - centroid
        try:
            cov = np.cov(centered.T)
            evals, evecs = np.linalg.eigh(cov)
            evals = np.sort(evals)[::-1]
            if evals[0] > 0 and (1.0 - evals[1]/evals[0]) > 0.85:
                return None
            angle = np.arctan2(evecs[1, 1], evecs[0, 1])
            while angle > np.pi/2: angle -= np.pi
            while angle < -np.pi/2: angle += np.pi
        except:
            angle = 0.0
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rotated = np.column_stack([centered[:,0]*cos_a - centered[:,1]*sin_a, centered[:,0]*sin_a + centered[:,1]*cos_a])
        min_xy, max_xy = np.min(rotated, axis=0), np.max(rotated, axis=0)
        w, h = max(max_xy[0] - min_xy[0], 0.1), max(max_xy[1] - min_xy[1], 0.1)
        # Note: We no longer swap w/h to force w >= h, as this prevents
        # proper convergence during optimization
        return centroid[0], centroid[1], w, h, angle

    def add_historical_points(self, points_room, sdf_values, robot_cov):
        """Add points with low SDF to historical storage."""
        sdf_threshold = self.config.sdf_threshold_for_storage
        mask = torch.abs(sdf_values) < sdf_threshold
        if not mask.any():
            return
        pts = points_room[mask]
        sdf = torch.abs(sdf_values[mask])

        # Ensure points are 3D
        if pts.shape[1] == 2:
            pts = torch.cat([pts, torch.zeros(len(pts), 1, dtype=DTYPE, device=pts.device)], dim=1)
        elif pts.shape[1] > 3:
            pts = pts[:, :3]  # Take only first 3 columns

        rob_cov = torch.tensor(robot_cov[:2, :2], dtype=DTYPE, device=self.mu.device)
        n = len(pts)
        I = torch.eye(2, dtype=DTYPE, device=self.mu.device)
        covs = rob_cov.unsqueeze(0).expand(n,-1,-1) + (self.config.beta_sdf * sdf**2).view(-1,1,1) * I.unsqueeze(0)
        rfe = torch.zeros(n, dtype=DTYPE, device=self.mu.device)
        self._add_to_bins(pts, covs, rfe)

    def _compute_edge_score(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute edge/corner score for points based on SDF gradient structure.

        Points near edges/corners have high geometric information content.
        We detect these by checking if the point is close to multiple faces.

        Returns:
            [N] scores in [0, 1], higher = more edge/corner-like
        """
        # Transform points to box-local frame
        cx, cy = self.mu[0], self.mu[1]
        w, h, d = self.mu[2], self.mu[3], self.mu[4]
        theta = self.mu[5]

        cos_t = torch.cos(-theta)
        sin_t = torch.sin(-theta)

        px = points[:, 0] - cx
        py = points[:, 1] - cy

        local_x = px * cos_t - py * sin_t
        local_y = px * sin_t + py * cos_t
        local_z = points[:, 2] - d / 2  # Center Z at box center

        # Distance to each face (absolute, in local frame)
        half_w, half_h, half_d = w / 2, h / 2, d / 2

        dist_x = torch.abs(torch.abs(local_x) - half_w)  # Distance to X faces
        dist_y = torch.abs(torch.abs(local_y) - half_h)  # Distance to Y faces
        dist_z = torch.abs(torch.abs(local_z) - half_d)  # Distance to Z faces

        # Threshold for "close to face"
        threshold = self.config.edge_proximity_threshold

        # Count how many faces the point is close to
        close_x = (dist_x < threshold).float()
        close_y = (dist_y < threshold).float()
        close_z = (dist_z < threshold).float()

        faces_close = close_x + close_y + close_z

        # Score: 0 = flat face, 0.5 = edge (2 faces), 1.0 = corner (3 faces)
        # Also give partial credit based on distance
        edge_score = (faces_close - 1).clamp(0, 2) / 2.0

        # Add gradient-based refinement: smaller distances = stronger edge
        min_dist = torch.minimum(torch.minimum(dist_x, dist_y), dist_z)
        proximity_bonus = torch.exp(-min_dist / 0.02) * 0.2  # Bonus for being very close

        return (edge_score + proximity_bonus).clamp(0, 1)

    def _add_to_bins(self, pts, covs, rfe):
        """
        Add points to angular/height bins for uniform surface coverage.

        Within each bin, points are sorted by quality = trace(Σ) + RFE - edge_bonus
        Lower quality score = better point:
        - Lower uncertainty (trace Σ)
        - Lower accumulated error (RFE)
        - Higher edge/corner score (bonus)

        Edge/corner points get a quality bonus because they provide more
        geometric information for constraining the box shape.
        """
        cx, cy = self.mu[0].item(), self.mu[1].item()
        na = self.config.num_angle_bins
        nz = self.config.num_z_bins
        mpb = self.config.max_historical_points // (na * nz) + 1

        # Compute edge scores for new points
        edge_scores = self._compute_edge_score(pts)
        edge_bonus_weight = self.config.edge_bonus_weight

        dx, dy = pts[:,0] - cx, pts[:,1] - cy
        angles = torch.atan2(dy, dx)
        zb = (pts[:,2] / 0.1).long().clamp(0, nz-1)
        ab = ((angles + np.pi) / (2*np.pi) * na).long() % na
        bidx = ab * nz + zb
        bins = {}

        # Add existing historical points to bins
        if len(self.historical_points) > 0:
            # Recompute edge scores for existing points (box may have moved)
            existing_edge_scores = self._compute_edge_score(self.historical_points)

            for i in range(len(self.historical_points)):
                p, c, r = self.historical_points[i], self.historical_capture_covs[i], self.historical_rfe[i].item()
                dxh, dyh = p[0]-cx, p[1]-cy
                ah = torch.atan2(dyh, dxh)
                zbi = int((p[2]/0.1).clamp(0,nz-1).item())
                abi = int(((ah+np.pi)/(2*np.pi)*na).clamp(0,na-1).item())
                idx = abi*nz + zbi
                if idx not in bins: bins[idx] = []

                # Quality metric: trace(cov) + RFE - edge_bonus (lower is better)
                edge_bonus = existing_edge_scores[i].item() * edge_bonus_weight
                quality = torch.trace(c).item() + r - edge_bonus
                bins[idx].append((p, c, r, quality))

        # Add new points to bins
        for i in range(len(pts)):
            idx = bidx[i].item()
            if idx not in bins: bins[idx] = []

            # Quality with edge bonus
            edge_bonus = edge_scores[i].item() * edge_bonus_weight
            quality = torch.trace(covs[i]).item() + rfe[i].item() - edge_bonus
            bins[idx].append((pts[i], covs[i], rfe[i].item(), quality))

        # Keep best points per bin (sorted by quality, lower = better)
        fp, fc, fr = [], [], []
        for contents in bins.values():
            for p, c, r, _ in sorted(contents, key=lambda x: x[3])[:mpb]:
                fp.append(p); fc.append(c); fr.append(r)
        if fp:
            self.historical_points = torch.stack(fp)
            self.historical_capture_covs = torch.stack(fc)
            self.historical_rfe = torch.tensor(fr, dtype=DTYPE, device=self.mu.device)

    def update_rfe(self, robot_cov: np.ndarray):
        """
        Update Remembered Free Energy for all historical points.

        RFE accumulates the SDF error weighted by robot certainty:
        RFE_i(t) = α * RFE_i(t-1) + w_t * SDF(p_i, s_t)²

        Points with consistently low SDF (RFE≈0) are "trusted memories".
        Points with high accumulated SDF error are unreliable.

        Args:
            robot_cov: Current robot pose covariance [3,3]
        """
        if len(self.historical_points) == 0:
            return

        alpha = self.config.rfe_alpha
        max_rfe = self.config.rfe_max_threshold

        # Compute current SDF for all historical points (in room frame)
        sdf_values = self.sdf(self.historical_points)
        sdf_squared = sdf_values ** 2

        # Weight based on current robot certainty
        robot_trace = np.trace(robot_cov)
        w_t = 1.0 / (1.0 + robot_trace)

        # Update RFE with temporal decay
        # RFE(t) = α * RFE(t-1) + w_t * SDF²
        self.historical_rfe = alpha * self.historical_rfe + w_t * sdf_squared.detach()

        # Prune points with very high RFE (consistently bad evidence)
        good_mask = self.historical_rfe < max_rfe

        if good_mask.sum() < len(self.historical_points):
            n_removed = len(self.historical_points) - good_mask.sum().item()
            self.historical_points = self.historical_points[good_mask]
            self.historical_capture_covs = self.historical_capture_covs[good_mask]
            self.historical_rfe = self.historical_rfe[good_mask]

    def get_historical_points_in_robot_frame(self, robot_pose, robot_cov):
        if len(self.historical_points) == 0:
            return torch.empty((0,3), dtype=DTYPE, device=self.mu.device), torch.empty(0, dtype=DTYPE, device=self.mu.device)
        rx, ry, rth = robot_pose
        ct, st = np.cos(-rth), np.sin(-rth)
        px, py = self.historical_points[:,0]-rx, self.historical_points[:,1]-ry
        pts = torch.stack([px*ct - py*st, px*st + py*ct, self.historical_points[:,2]], dim=1)
        rc = torch.tensor(robot_cov, dtype=DTYPE, device=self.mu.device)
        w = torch.zeros(len(self.historical_points), dtype=DTYPE, device=self.mu.device)
        for i in range(len(self.historical_points)):
            J = torch.tensor(compute_jacobian_room_to_robot(self.historical_points[i].cpu().numpy(), robot_pose), dtype=DTYPE, device=self.mu.device)
            st = self.historical_capture_covs[i] + J @ rc @ J.T
            w[i] = 1.0 / (1.0 + torch.trace(st) + self.historical_rfe[i])
        return pts, w
    def to_dict(self):
        d = super().to_dict()
        d.update({'cx': self.cx, 'cy': self.cy, 'width': self.width, 'height': self.height, 'depth': self.depth, 'angle': self.angle})
        return d
    @staticmethod
    def debug_vs_gt(belief, gt_cx=0., gt_cy=0., gt_w=0.5, gt_h=0.5, gt_d=0.5, gt_theta=0.):
        """Print debug info comparing belief against Ground Truth."""
        est_cx, est_cy = belief.cx, belief.cy
        est_w, est_h, est_d = belief.width, belief.height, belief.depth
        est_theta = belief.angle

        # Calculate errors
        pos_error = np.sqrt((est_cx - gt_cx)**2 + (est_cy - gt_cy)**2)
        w_error = est_w - gt_w
        h_error = est_h - gt_h
        d_error = est_d - gt_d
        theta_error = np.degrees(est_theta - gt_theta)

        # Volume ratio
        est_volume = est_w * est_h * est_d
        gt_volume = gt_w * gt_h * gt_d
        volume_ratio = est_volume / gt_volume if gt_volume > 0 else 0

        print(f"\n{'='*60}")
        print(f"BELIEF vs GROUND TRUTH (GT: {gt_w}x{gt_h}x{gt_d} at ({gt_cx},{gt_cy}))")
        print(f"{'='*60}")
        print(f"Position:  Est=({est_cx:.3f}, {est_cy:.3f})  GT=({gt_cx:.3f}, {gt_cy:.3f})  Error={pos_error:.4f}m")
        print(f"Width(X):  Est={est_w:.3f}m  GT={gt_w:.3f}m  Error={w_error:+.4f}m ({100*w_error/gt_w:+.1f}%)")
        print(f"Height(Y): Est={est_h:.3f}m  GT={gt_h:.3f}m  Error={h_error:+.4f}m ({100*h_error/gt_h:+.1f}%)")
        print(f"Depth(Z):  Est={est_d:.3f}m  GT={gt_d:.3f}m  Error={d_error:+.4f}m ({100*d_error/gt_d:+.1f}%)")
        print(f"Angle:     Est={np.degrees(est_theta):.1f}°  GT={np.degrees(gt_theta):.1f}°  Error={theta_error:+.1f}°")
        print(f"Volume:    Est={est_volume:.4f}m³  GT={gt_volume:.4f}m³  Ratio={volume_ratio:.2f}x")
        print(f"SDF mean:  {belief.last_sdf_mean:.4f}")

        # Show Z range of historical points and RFE statistics
        if belief.num_historical_points > 0:
            z_vals = belief.historical_points[:, 2].cpu().numpy()
            z_min, z_max = z_vals.min(), z_vals.max()
            rfe_vals = belief.historical_rfe.cpu().numpy()
            rfe_min, rfe_max, rfe_mean = rfe_vals.min(), rfe_vals.max(), rfe_vals.mean()

            print(f"Hist pts:  N={belief.num_historical_points}, Z=[{z_min:.3f}, {z_max:.3f}]m")
            print(f"           Box draws from z=0 to z={est_d:.3f}m, points go to z={z_max:.3f}m")
            print(f"RFE:       min={rfe_min:.4f}, max={rfe_max:.4f}, mean={rfe_mean:.4f}")

            # Count trusted vs unreliable points using config thresholds
            cfg = belief.config
            trusted = (rfe_vals < cfg.rfe_trusted_threshold).sum()
            good = ((rfe_vals >= cfg.rfe_trusted_threshold) & (rfe_vals < cfg.rfe_good_threshold)).sum()
            moderate = ((rfe_vals >= cfg.rfe_good_threshold) & (rfe_vals < cfg.rfe_moderate_threshold)).sum()
            unreliable = (rfe_vals >= cfg.rfe_moderate_threshold).sum()
            print(f"           Trusted(RFE<{cfg.rfe_trusted_threshold})={trusted}, Good={good}, Moderate={moderate}, Unreliable(RFE>{cfg.rfe_moderate_threshold})={unreliable}")

            # Edge/corner statistics using config thresholds
            edge_scores = belief._compute_edge_score(belief.historical_points).cpu().numpy()
            corners = (edge_scores > cfg.corner_score_threshold).sum()
            edges = ((edge_scores > cfg.edge_score_threshold) & (edge_scores <= cfg.corner_score_threshold)).sum()
            flat = (edge_scores <= cfg.edge_score_threshold).sum()
            print(f"Geometry:  Corners={corners}, Edges={edges}, Flat={flat}")

            if est_d > z_max * 1.5:
                print(f"           WARNING: Box height ({est_d:.3f}m) >> observed Z_max ({z_max:.3f}m)!")

        print(f"{'='*60}\n")
