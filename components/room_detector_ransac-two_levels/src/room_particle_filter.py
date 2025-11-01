#!/usr/bin/env python3
import math
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from dataclasses import dataclass


# ---------------------------------------------------------------------
# Utility: Signed distance to a centered rectangle (length, width)
# ---------------------------------------------------------------------
def sdf_rect(points, length, width):
    """
    Analytic 2D signed distance to an axis-aligned rectangle centered at (0,0)
    """
    half = torch.tensor([length / 2.0, width / 2.0], device=points.device)
    q = torch.abs(points) - half
    outside = torch.clamp(q, min=0.0)
    dist_out = outside.norm(dim=1)
    dist_in = torch.clamp(torch.max(q[:, 0], q[:, 1]), max=0.0)
    return dist_out + dist_in  # negative inside


# ---------------------------------------------------------------------
# Core smooth SDF loss (shared by PF & optimizer)
# ---------------------------------------------------------------------
def smooth_sdf_loss(points_room, L, W, margin=-0.02, delta=0.05):
    """
    Smooth, robust penalty for points outside the room rectangle.
    Points are given in the room frame.
    """
    if points_room.numel() == 0:
        return torch.tensor(0.0, device=points_room.device)

    sdf = sdf_rect(points_room, L, W)
    r = torch.relu(sdf - margin)  # penalize outside walls
    absr = torch.abs(r)
    quad = 0.5 * absr**2
    lin = delta * (absr - 0.5 * delta)
    hub = torch.where(absr <= delta, quad, lin)
    return hub.mean()


# ---------------------------------------------------------------------
# Particle definition
# ---------------------------------------------------------------------
@dataclass
class Particle:
    x: float
    y: float
    theta: float
    length: float
    width: float
    weight: float = 1.0
    height: float = 2.5
    z_center: float = 1.25

# ---------------------------------------------------------------------
# RoomParticleFilter class
# ---------------------------------------------------------------------
class RoomParticleFilter:
    def __init__(
        self,
        num_particles,
        initial_hypothesis,
        device="cpu",
        use_gradient_refinement=True,
    ):
        self.device = device
        self.num_particles = num_particles
        self.use_gradient_refinement = use_gradient_refinement
        self.tau = 0.03  # temperature for weight scaling
        self.particles = [
            deepcopy(initial_hypothesis) for _ in range(num_particles)
        ]

        # Noise parameters
        self.trans_noise = 0.02
        self.rot_noise = 0.02

        # Gradient refinement settings
        self.lr = 0.01
        self.num_steps = 10
        self.top_n = 3
        self.pose_lambda = 1e-2
        self.size_lambda = 1e-4

    # -----------------------------------------------------------------
    def predict(self, odometry_delta):
        """Apply motion model with noise to all particles."""
        dx, dy, dtheta = odometry_delta
        for p in self.particles:
            ndx = dx + np.random.normal(0, self.trans_noise)
            ndy = dy + np.random.normal(0, self.trans_noise)
            ndtheta = dtheta + np.random.normal(0, self.rot_noise)
            c, s = math.cos(p.theta), math.sin(p.theta)
            p.x += ndx * c - ndy * s
            p.y += ndx * s + ndy * c
            p.theta = (p.theta + ndtheta + math.pi) % (2 * math.pi) - math.pi

    # -----------------------------------------------------------------
    def compute_fitness_loss(self, particle, plane_detector, pcd):
        """Compute the smooth SDF loss for a given particle."""
        # --- 1. Get wall inlier points (fallback to all points)
        try:
            wall_points = plane_detector.get_wall_inlier_points(pcd)
            if wall_points is None or len(wall_points) == 0:
                wall_points = np.asarray(pcd.points)
        except Exception:
            wall_points = np.asarray(pcd.points)

        if len(wall_points) == 0:
            return 0.0

        points = torch.from_numpy(wall_points[:, :2]).float().to(self.device)

        # --- 2. Transform points into the room frame of the particle
        c, s = math.cos(-particle.theta), math.sin(-particle.theta)
        R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32, device=self.device)
        t = torch.tensor([particle.x, particle.y], dtype=torch.float32, device=self.device)
        points_room = (points - t) @ R.T

        # --- 3. Smooth loss
        loss = smooth_sdf_loss(points_room, particle.length, particle.width)
        return loss.item()

    # -----------------------------------------------------------------
    def update(self, plane_detector, pcd):
        """Compute weights from SDF-based loss and normalize."""
        losses = []
        for p in self.particles:
            L = self.compute_fitness_loss(p, plane_detector, pcd)
            losses.append(L)

        # Convert losses to weights (lower loss -> higher weight)
        weights = np.exp(-np.array(losses) / self.tau)
        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
        weights /= np.sum(weights)

        for p, w in zip(self.particles, weights):
            p.weight = w

    # -----------------------------------------------------------------
    def resample(self):
        """Systematic resampling."""
        weights = np.array([p.weight for p in self.particles])
        N = len(self.particles)
        positions = (np.arange(N) + np.random.rand()) / N
        indexes = np.zeros(N, dtype=int)
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles = [deepcopy(self.particles[k]) for k in indexes]

    # -----------------------------------------------------------------
    def refine_best_particles_gradient(self, plane_detector, pcd):
        """Gradient-based refinement of top-N particles."""
        # pick top-n by weight
        top_particles = sorted(self.particles, key=lambda p: p.weight, reverse=True)[: self.top_n]

        for p in top_particles:
            # initialize tensors
            x = torch.tensor(p.x, requires_grad=True, dtype=torch.float32, device=self.device)
            y = torch.tensor(p.y, requires_grad=True, dtype=torch.float32, device=self.device)
            theta = torch.tensor(p.theta, requires_grad=True, dtype=torch.float32, device=self.device)
            L = torch.tensor(p.length, requires_grad=True, dtype=torch.float32, device=self.device)
            W = torch.tensor(p.width, requires_grad=True, dtype=torch.float32, device=self.device)

            opt = torch.optim.Adam([x, y, theta, L, W], lr=self.lr)

            # get wall points once (fallback safe)
            try:
                wall_points = plane_detector.get_wall_inlier_points(pcd)
                if wall_points is None or len(wall_points) == 0:
                    wall_points = np.asarray(pcd.points)
            except Exception:
                wall_points = np.asarray(pcd.points)
            points = torch.from_numpy(wall_points[:, :2]).float().to(self.device)

            x0, y0, theta0 = (
                torch.tensor(p.x, device=self.device),
                torch.tensor(p.y, device=self.device),
                torch.tensor(p.theta, device=self.device),
            )
            L0 = torch.tensor(p.length, device=self.device)
            W0 = torch.tensor(p.width, device=self.device)

            for _ in range(self.num_steps):
                opt.zero_grad()
                c, s = torch.cos(-theta), torch.sin(-theta)
                R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
                t = torch.stack([x, y])
                points_room = (points - t) @ R.T
                loss_data = smooth_sdf_loss(points_room, L, W)
                # small priors to keep stability
                dth = ((theta - theta0 + math.pi) % (2 * math.pi)) - math.pi
                loss = (
                    loss_data
                    + self.pose_lambda * ((x - x0) ** 2 + (y - y0) ** 2 + dth**2)
                    + self.size_lambda * ((L - L0) ** 2 + (W - W0) ** 2)
                )
                loss.backward()
                opt.step()

                with torch.no_grad():
                    theta.data = (theta.data + math.pi) % (2 * math.pi) - math.pi
                    L.data = torch.clamp(L.data, min=1.0)
                    W.data = torch.clamp(W.data, min=1.0)

            # commit updates
            p.x = float(x.item())
            p.y = float(y.item())
            p.theta = float(theta.item())
            p.length = float(L.item())
            p.width = float(W.item())

    # -----------------------------------------------------------------
    def step(self, odometry_delta, plane_detector, pcd):
        """Main PF step: predict, update, refine, resample."""
        self.predict(odometry_delta)
        self.update(plane_detector, pcd)
        if self.use_gradient_refinement:
            self.refine_best_particles_gradient(plane_detector, pcd)
        self.resample()

    # -----------------------------------------------------------------
    def best_particle(self):
        """Return particle with max weight."""
        return max(self.particles, key=lambda p: p.weight)
