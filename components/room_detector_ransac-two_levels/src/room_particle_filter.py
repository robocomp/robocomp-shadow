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
    #half = torch.tensor([length / 2.0, width / 2.0], device=points.device)
    half = torch.tensor([
        (length / 2.0).detach() if isinstance(length, torch.Tensor) else length / 2.0,
        (width / 2.0).detach() if isinstance(width, torch.Tensor) else width / 2.0
    ], device=points.device)
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

        # initialize particles around the initial hypothesis
        base = deepcopy(initial_hypothesis)
        rng = np.random.default_rng()

        def jitter():
            # very small spread; adjust if needed
            return (rng.normal(0, 0.05),  # x [m]
                    rng.normal(0, 0.05),  # y [m]
                    rng.normal(0, 0.05))  # theta [rad]

        self.particles = []
        for _ in range(self.num_particles):
            dx, dy, dth = jitter()
            p = deepcopy(base)
            p.x += dx
            p.y += dy
            p.theta = ((p.theta + dth + np.pi) % (2 * np.pi)) - np.pi
            p.weight = 1.0 / self.num_particles
            self.particles.append(p)

        # Noise parameters
        self.trans_noise = 0.01
        self.rot_noise = 0.01

        # Resampling settings
        self.ess_frac = 0.75  # resample when ESS < 0.5*N
        self.ess = float(getattr(self, "num_particles", 1))  # start with a sane value
        self.ess_pct = None

        # entropy-based diversity
        self.diversity = 0

        # Gradient refinement settings
        self.lr = 0.01              #
        self.num_steps = 10         # gradient steps per refinement
        self.top_n = 3              # refine top-N particles
        self.pose_lambda = 1e-2     # pose prior strength: constraint to original pose
        self.size_lambda = 1e-4     # size prior strength

        # loss history
        self.history = {
            "tick": [],
            "loss_best": [],
            "num_features": [],
            "ess": [],
            "births": [],
            "deaths": [],
            "n_particles": [],
            "ess_pct": [],
            "weight_entropy": []
        }
        self._tick = getattr(self, "_tick", 0)

    # -----------------------------------------------------------------
    def step(self, odometry_delta, plane_detector, pcd):
        """Main PF step: predict, update, refine, resample."""
        self.predict(odometry_delta)
        self.update(plane_detector, pcd)
        if self.use_gradient_refinement:
            self.refine_best_particles_gradient(plane_detector, pcd)

        # resample only if ESS below threshold
        if self.ess < self.ess_frac * len(self.particles):
            self.resample()
            self.roughen_after_resample(sigma_xy=0.003, sigma_th=0.004)

        self.log_history( float(self.compute_fitness_loss(self.best_particle(), plane_detector, pcd)))

    # -----------------------------------------------------------------
    def predict(self, odometry_delta):
        """Apply motion model with noise to all particles."""
        dx, dy, dtheta = odometry_delta
        for p in self.particles:
            speed = abs(dx) + abs(dy) + 0.2 * abs(dtheta)
            if speed < 1e-3:  # stationary
                tn, rn = 0.003, 0.004
            else:  # moving
                tn, rn = 0.01, 0.01     #TODO: move to class params
            ndx = dx + np.random.normal(0, tn)
            ndy = dy + np.random.normal(0, tn)
            ndtheta = dtheta + np.random.normal(0, rn)
            c, s = math.cos(p.theta), math.sin(p.theta)
            # subtract because joystick velocities in webots are inverted
            p.x -= ndx * c - ndy * s
            p.y -= ndx * s + ndy * c
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

        losses = np.asarray(losses, dtype=np.float64)
        med = np.median(losses)
        mad = np.median(np.abs(losses - med))
        eps = 1e-12

        if mad < eps:
            weights = np.ones_like(losses) / len(losses)
        else:
            alpha = 0.5
            w = np.exp(- (losses - med) / (alpha * mad))
            weights = w / w.sum()

        # ---- entropy-based diversity in [0, 1] ----
        eps = 1e-12
        H = -np.sum(weights * np.log(weights + eps))  # nats
        H_max = np.log(len(weights) + eps)
        self.diversity = float(H / max(H_max, eps)) # ∈ [0,1]

        # Compute effective sample size (ESS)
        self.ess = float(1.0 / np.maximum(1e-12, np.sum(weights * weights)))
        self.ess_pct = 100.0 * self.ess / len(self.particles)

        # assign weights to particles
        for p, w in zip(self.particles, weights):
            p.weight = float(w)


    # -----------------------------------------------------------------
    def resample(self):
        """Systematic resampling."""
        N = len(self.particles)
        weights = np.array([p.weight for p in self.particles], dtype=float)

        # normalize defensively
        wsum = weights.sum()
        if wsum <= 0:
            weights = np.ones(N, dtype=float) / N
        else:
            weights /= wsum

        positions = (np.arange(N) + np.random.rand()) / N
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0  # guard against roundoff

        indexes = np.zeros(N, dtype=int)
        i = j = 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        # clone and RESET weights uniformly
        new_particles = [deepcopy(self.particles[k]) for k in indexes]
        u = 1.0 / N
        for p in new_particles:
            p.weight = u
        self.particles = new_particles

    # -----------------------------------------------------------------
    def refine_best_particles_gradient(self, plane_detector, pcd):
        """Gradient-based refinement of top-N particles."""
        # pick top-n by weight
        top_particles = sorted(self.particles, key=lambda p: p.weight, reverse=True)[: self.top_n]

        # get wall points once (fallback safe)
        try:
            wall_points = plane_detector.get_wall_inlier_points(pcd)
            if wall_points is None or len(wall_points) == 0:
                wall_points = np.asarray(pcd.points)
        except Exception:
            wall_points = np.asarray(pcd.points)
        points = torch.from_numpy(wall_points[:, :2]).float().to(self.device)
        
        for p in top_particles:
            # initialize tensors
            x = torch.tensor(p.x, requires_grad=True, dtype=torch.float32, device=self.device)
            y = torch.tensor(p.y, requires_grad=True, dtype=torch.float32, device=self.device)
            theta = torch.tensor(p.theta, requires_grad=True, dtype=torch.float32, device=self.device)
            L = torch.tensor(p.length, requires_grad=True, dtype=torch.float32, device=self.device)
            W = torch.tensor(p.width, requires_grad=True, dtype=torch.float32, device=self.device)

            opt = torch.optim.Adam([x, y, theta, L, W], lr=self.lr)

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
    def roughen_after_resample(self, sigma_xy=0.01, sigma_th=0.01):
        for p in self.particles:
            p.x += np.random.normal(0, sigma_xy)
            p.y += np.random.normal(0, sigma_xy)
            p.theta = ((p.theta + np.random.normal(0, sigma_th) + np.pi) % (2 * np.pi)) - np.pi

    # -----------------------------------------------------------------
    def best_particle(self):
        """Return particle with max weight."""
        return max(self.particles, key=lambda p: p.weight)

    # -----------------------------------------------------------------
    def get_history(self):
        """
        Return diagnostic history for plotting.
        The dict always has these keys, even if features are disabled.
        """
        # ensure keys exist so the plotter never crashes
        for k in ["tick", "loss_best", "num_features", "ess", "births", "deaths", "n_particles", "ess_pct", "weight_entropy"]:
            if k not in self.history:
                self.history[k] = []
        return self.history

    def log_history(self, loss_best=None):
        self._tick += 1
        self.history["tick"].append(self._tick)
        self.history["loss_best"].append(loss_best)
        self.history["num_features"].append(len(getattr(self.map, "features", [])) if hasattr(self, "map") else 0)
        self.history["ess"].append(self.ess)
        self.history["births"].append(0)
        self.history["deaths"].append(0)
        self.history.setdefault("n_particles", []).append(int(len(self.particles)))
        self.history.setdefault("weight_entropy", []).append(self.diversity)
        self.history.setdefault("ess_pct", []).append(self.ess_pct)




