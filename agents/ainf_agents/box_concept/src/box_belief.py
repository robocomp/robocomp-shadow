#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Box Belief - Belief class for 3D box obstacles. State: [cx, cy, w, h, d, theta]"""
import numpy as np
import torch
from typing import Optional, Tuple
from dataclasses import dataclass
from src.belief_core import Belief, BeliefConfig, DEVICE, DTYPE
from src.sdf_functions import compute_box_sdf
from src.transforms import compute_jacobian_room_to_robot
@dataclass
class BoxBeliefConfig(BeliefConfig):
    """Configuration for 3D box beliefs."""
    prior_size_mean: float = 0.5
    prior_size_std: float = 0.2
    min_aspect_ratio: float = 0.35
    cluster_eps: float = 0.25
    min_cluster_points: int = 12
    max_association_cost: float = 5.0
    max_association_distance: float = 0.8
    wall_margin: float = 0.30
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
        if h > w:
            w, h = h, w
            angle += np.pi/2
            while angle > np.pi/2: angle -= np.pi
        return centroid[0], centroid[1], w, h, angle
    def add_historical_points(self, points_room, sdf_values, robot_cov, sdf_threshold=0.05):
        mask = torch.abs(sdf_values) < sdf_threshold
        if not mask.any():
            return
        pts = points_room[mask]
        sdf = torch.abs(sdf_values[mask])
        if pts.shape[1] == 2:
            pts = torch.cat([pts, torch.zeros(len(pts), 1, dtype=DTYPE, device=pts.device)], dim=1)
        rob_cov = torch.tensor(robot_cov[:2, :2], dtype=DTYPE, device=self.mu.device)
        n = len(pts)
        I = torch.eye(2, dtype=DTYPE, device=self.mu.device)
        covs = rob_cov.unsqueeze(0).expand(n,-1,-1) + (self.beta_sdf * sdf**2).view(-1,1,1) * I.unsqueeze(0)
        rfe = torch.zeros(n, dtype=DTYPE, device=self.mu.device)
        self._add_to_bins(pts, covs, rfe)
    def _add_to_bins(self, pts, covs, rfe):
        cx, cy = self.mu[0].item(), self.mu[1].item()
        na, nz, mpb = 24, 8, self.max_historical_points // (24*8) + 1
        dx, dy = pts[:,0] - cx, pts[:,1] - cy
        angles = torch.atan2(dy, dx)
        zb = (pts[:,2] / 0.1).long().clamp(0, nz-1)
        ab = ((angles + np.pi) / (2*np.pi) * na).long() % na
        bidx = ab * nz + zb
        bins = {}
        if len(self.historical_points) > 0:
            for i in range(len(self.historical_points)):
                p, c, r = self.historical_points[i], self.historical_capture_covs[i], self.historical_rfe[i].item()
                dxh, dyh = p[0]-cx, p[1]-cy
                ah = torch.atan2(dyh, dxh)
                zbi = int((p[2]/0.1).clamp(0,nz-1).item())
                abi = int(((ah+np.pi)/(2*np.pi)*na).clamp(0,na-1).item())
                idx = abi*nz + zbi
                if idx not in bins: bins[idx] = []
                bins[idx].append((p, c, r, torch.trace(c).item()))
        for i in range(len(pts)):
            idx = bidx[i].item()
            if idx not in bins: bins[idx] = []
            bins[idx].append((pts[i], covs[i], rfe[i].item(), torch.trace(covs[i]).item()))
        fp, fc, fr = [], [], []
        for contents in bins.values():
            for p, c, r, _ in sorted(contents, key=lambda x: x[3])[:mpb]:
                fp.append(p); fc.append(c); fr.append(r)
        if fp:
            self.historical_points = torch.stack(fp)
            self.historical_capture_covs = torch.stack(fc)
            self.historical_rfe = torch.tensor(fr, dtype=DTYPE, device=self.mu.device)
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
        pe = np.sqrt((belief.cx-gt_cx)**2 + (belief.cy-gt_cy)**2)
        print(f"\n{'='*60}\nBELIEF vs GT ({gt_w}x{gt_h}x{gt_d})\nPos: ({belief.cx:.3f},{belief.cy:.3f}) err={pe:.4f}m")
        print(f"W:{belief.width:.3f} H:{belief.height:.3f} D:{belief.depth:.3f}")
        if belief.num_historical_points > 0:
            z = belief.historical_points[:,2].cpu().numpy()
            print(f"Hist: N={belief.num_historical_points} Z=[{z.min():.3f},{z.max():.3f}]")
        print(f"{'='*60}\n")
