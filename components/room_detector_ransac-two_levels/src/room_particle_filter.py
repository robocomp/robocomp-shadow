#!/usr/bin/env python3
import math
import numpy as np
import torch
from copy import deepcopy
from dataclasses import dataclass
from src.params import Pose2D
from src.helios_model import RealisticHeliosRaycaster
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
            *,
            device="cpu",
            use_gradient_refinement=True,
            adaptive_particles=True,
            min_particles=20,
            max_particles=300,
            elite_count=5,  # Number of best particles to preserve during resampling

            # NEW: motion noise (moving vs. stationary)
            trans_noise=0.01,
            rot_noise=0.01,
            trans_noise_stationary=0.00,
            rot_noise_stationary=0.00,

            # NEW: resampling/ESS
            ess_frac=0.15,  # resample when ESS < ess_frac * N
            target_ess_pct=75.0,  # target ESS% for adaptive particles

            # NEW: gradient refinement
            lr=0.05,
            num_steps=25,
            top_n=3,
            pose_lambda=1e-2,
            size_lambda=1e-3,
    ):
        assert num_particles > 0, "num_particles must be positive"
        assert initial_hypothesis.length > 0, "length must be positive"
        assert initial_hypothesis.width > 0, "width must be positive"
        assert min_particles > 0, "min_particles must be positive"
        assert max_particles >= min_particles, "max_particles must be >= min_particles"

        self.device = device
        self.num_particles = num_particles
        self.use_gradient_refinement = use_gradient_refinement

        # Adaptive particle settings
        self.adaptive_particles = adaptive_particles
        self.min_particles = min_particles
        self.max_particles = max_particles
        self.target_ess_pct = target_ess_pct
        self.last_adaptation_tick = -10
        self.elite_count = elite_count

        # initialize particles around the initial hypothesis
        base = deepcopy(initial_hypothesis)
        rng = np.random.default_rng()

        def jitter():
            return (rng.normal(0, 0.05), rng.normal(0, 0.05), rng.normal(0, 0.05))

        self.particles = []
        for _ in range(self.num_particles):
            dx, dy, dth = jitter()
            p = deepcopy(base)
            p.x += dx
            p.y += dy
            p.theta = ((p.theta + dth + np.pi) % (2 * np.pi)) - np.pi
            p.weight = 1.0 / self.num_particles
            self.particles.append(p)

        # Motion noise
        self.trans_noise = float(trans_noise)
        self.rot_noise = float(rot_noise)
        self.trans_noise_stationary = float(trans_noise_stationary)
        self.rot_noise_stationary = float(rot_noise_stationary)

        # Resampling settings / diagnostics
        self.ess_frac = float(ess_frac)
        self.ess = float(getattr(self, "num_particles", 1))
        self.ess_pct = None
        self.weight_entropy = 0.0

        # Gradient refinement
        self.lr = float(lr)
        self.num_steps = int(num_steps)
        self.top_n = int(top_n)
        self.top_n_max = 3 * int(top_n)  # for exploration phase
        self.pose_lambda = float(pose_lambda)
        self.size_lambda = float(size_lambda)

        # loss history
        self.history = {
            "tick": [], "loss_best": [], "num_features": [], "ess": [],
            "births": [], "deaths": [], "n_particles": [], "ess_pct": [],
            "weight_entropy": [],  "x_std": [], "y_std": [], "theta_std": [],
        }
        self._tick = getattr(self, "_tick", 0)

        # smoother
        self.smooth_particle = None

        # Helios raycaster for visibility checking
        self.visibility_checker = RealisticHeliosRaycaster(
            num_horizontal_rays=180,  # 2° resolution
            num_vertical_samples=8,  # Sample 32 beams with 8
            lidar_height=1.1,  # 160cm above floor
            min_z=1.5,  # Filter z<1.5m
            max_range=10.0,  # Max range
            min_pitch_deg=-11.0,  # Bottom beam angle
            max_pitch_deg=55.0,  # Top beam angle
            min_ray_hits=10  # Need 10+ valid hits
        )

    # -----------------------------------------------------------------
    def step(self, odometry_delta, wall_points, period):
        """Main PF step: predict, update, refine, resample.

        Args:
            odometry_delta: (dx, dy, dtheta) motion since last step
            wall_points: torch.Tensor of shape (N, 2) containing wall points in world frame
        """
        self.predict(odometry_delta)
        if self.use_gradient_refinement:
            # Adapt top_n based on convergence
            pose_diversity = self.compute_pose_diversity()  # std of x, y, theta
            if pose_diversity > 0.1:  # Exploration phase
                top_n_adaptive = self.top_n_max  # Refine many to collapse distribution
            else:  # Converged phase
                top_n_adaptive = self.top_n  # Only refine top few (fast)
            self.refine_best_particles_gradient(wall_points, top_n=top_n_adaptive)
        self.update(wall_points)

        best_loss = self.compute_fitness_loss_with_points(self.best_particle()[0], wall_points)
        is_converged = (best_loss < 0.001 and self.weight_entropy < 0.2)  # Converged threshold
        # More lenient ESS threshold when converged
        ess_threshold = 0.3 if is_converged else 0.9
        if self.ess < self.ess_frac * len(self.particles):
            self.resample()

            if not is_converged:
                self.roughen_after_resample(sigma_xy=0.001, sigma_th=0.002)  # Much smaller noise
                #print(f"[Resample] tick={self._tick}, ESS={self.ess:.1f} < {self.ess_frac * len(self.particles):.1f}")

            # Adaptive particle count adjustment
            if self.adaptive_particles:
                self.adapt_particle_count()
        elif self._tick % 20 == 0:
            # print(f"[No Resample] tick={self._tick}, ESS={self.ess:.1f} >= {self.ess_frac * len(self.particles):.1f}")
            pass

        # Compute loss for logging
        self.history.setdefault("period", []).append(float(period))
        best_p, smoothed_best_p = self.best_particle()
        loss_best = self.compute_fitness_loss_with_points(smoothed_best_p, wall_points)
        self.log_history(float(loss_best))
        return float(loss_best), best_p, smoothed_best_p

    # -----------------------------------------------------------------
    def predict(self, odometry_delta):
        """Apply motion model with noise to all particles."""
        dx, dy, dtheta = odometry_delta
        for p in self.particles:
            speed = abs(dx) + abs(dy) + 0.2 * abs(dtheta)
            if speed < 1e-3:  # stationary
                tn, rn = self.trans_noise_stationary, self.rot_noise_stationary
            else:  # moving
                tn, rn = self.trans_noise, self.rot_noise
            ndx = dx + np.random.normal(0, tn)
            ndy = dy + np.random.normal(0, tn)
            ndtheta = dtheta + np.random.normal(0, rn)
            c, s = math.cos(p.theta), math.sin(p.theta)
            # subtract because joystick velocities in webots are inverted
            p.x -= ndx * c - ndy * s
            p.y -= ndx * s + ndy * c
            p.theta = (p.theta + ndtheta + math.pi) % (2 * math.pi) - math.pi

    # -----------------------------------------------------------------
    def compute_fitness_loss_with_points(self, particle, points):
        """Compute the smooth SDF loss for a given particle."""

        # Transform points into the room frame of the particle
        c, s = math.cos(-particle.theta), math.sin(-particle.theta)
        R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32, device=self.device)
        t = torch.tensor([particle.x, particle.y], dtype=torch.float32, device=self.device)
        points_room = (points - t) @ R.T

        #  Smooth loss
        loss = self.smooth_sdf_loss(points_room, particle.length, particle.width)
        return loss.item()

    # def compute_fitness_loss_with_points(self, particle, points):
    #     if points.numel() == 0:
    #         return 0.0
    #
    #     return  self.visibility_checker.compute_wall_specific_loss(
    #         particle, points, robot_pose_world=(0.0, 0.0, 0.0)
    #     )

    # -----------------------------------------------------------------
    def update(self, wall_points):
        """Compute weights from SDF-based loss and normalize.

        Args:
            wall_points: torch.Tensor of shape (N, 2) containing wall points
        """
        # Compute losses for all particles
        losses = []
        for p in self.particles:
            L = self.compute_fitness_loss_with_points(p, wall_points)
            losses.append(L)

        losses = np.asarray(losses, dtype=np.float64)
        med = np.median(losses)
        mad = np.median(np.abs(losses - med))
        eps = 1e-12
        alpha = 0.05   # smaller alpha -> more peaked weights -> more aggressive resampling -> faster convergence
        scale = max(mad, eps)
        z = - (losses - med) / (alpha * scale)
        z = np.clip(z, -50.0, 50.0)
        w = np.exp(z)
        w_sum = w.sum()
        weights = (w / w_sum) if w_sum > 0 else np.ones_like(w) / len(w)

        # ---- entropy-based diversity in [0, 1] : 0 -> low diversity, 1 -> high diversity ----
        eps = 1e-12
        H = -np.sum(weights * np.log(weights + eps))  # nats
        H_max = np.log(len(weights) + eps)
        self.weight_entropy = float(H / max(H_max, eps))  # ∈ [0,1]

        # Compute effective sample size (ESS)
        self.ess = float(1.0 / np.maximum(1e-12, np.sum(weights * weights)))
        self.ess_pct = 100.0 * self.ess / len(self.particles)

        # assign weights to particles
        for p, w in zip(self.particles, weights):
            p.weight = float(w)

    # -----------------------------------------------------------------
    def resample(self):
        """Systematic resampling with elitism."""
        N = len(self.particles)
        weights = np.array([p.weight for p in self.particles], dtype=float)

        # weights come normalized from update(), but guard anyway
        wsum = weights.sum()
        if wsum <= 0:
            weights = np.ones(N, dtype=float) / N

        # ELITISM: Keep top elite_count particles
        elite_count = min(self.elite_count, N // 4)  # Cap at 25% of population

        if elite_count > 0:
            # Sort by weight and get top particles
            sorted_indices = np.argsort(weights)[::-1]  # Descending order
            elite_indices = sorted_indices[:elite_count]
            elite_particles = [deepcopy(self.particles[i]) for i in elite_indices]
        else:
            elite_particles = []

        # Resample remaining (N - elite_count) slots
        n_resample = N - elite_count

        if n_resample > 0:
            positions = (np.arange(n_resample) + np.random.rand()) / n_resample
            cumulative_sum = np.cumsum(weights)
            cumulative_sum[-1] = 1.0  # guard against roundoff

            indexes = np.zeros(n_resample, dtype=int)
            i = j = 0
            while i < n_resample:
                if positions[i] < cumulative_sum[j]:
                    indexes[i] = j
                    i += 1
                else:
                    j += 1

            # Clone resampled particles
            resampled_particles = [deepcopy(self.particles[k]) for k in indexes]
        else:
            resampled_particles = []

        # Combine elite + resampled
        new_particles = elite_particles + resampled_particles

        # RESET weights uniformly
        u = 1.0 / N
        for p in new_particles:
            p.weight = u

        self.particles = new_particles

    # -----------------------------------------------------------------
    def refine_best_particles_gradient(self, wall_points, top_n=None):
        """Gradient-based refinement of top-N particles.

        Args:
            wall_points: torch.Tensor of shape (N, 2) containing wall points
        """
        if top_n is None:
            top_n = self.top_n

        # Check if already converged - reduce refinement intensity
        best_loss = self.compute_fitness_loss_with_points(self.best_particle()[0], wall_points)
        if best_loss < 0.0005:  # Already excellent
            num_steps_adaptive = 5  # Minimal refinement
            lr_adaptive = 0.01  # Smaller steps
        else:
            num_steps_adaptive = self.num_steps
            lr_adaptive = self.lr

        # pick top-n by weight
        top_particles = sorted(self.particles, key=lambda p: p.weight, reverse=True)[: top_n]

        for p in top_particles:
            # initialize tensors
            x = torch.tensor(p.x, requires_grad=True, dtype=torch.float32, device=self.device)
            y = torch.tensor(p.y, requires_grad=True, dtype=torch.float32, device=self.device)
            theta = torch.tensor(p.theta, requires_grad=True, dtype=torch.float32, device=self.device)
            L = torch.tensor(p.length, requires_grad=True, dtype=torch.float32, device=self.device)
            W = torch.tensor(p.width, requires_grad=True, dtype=torch.float32, device=self.device)

            opt = torch.optim.Adam([x, y, theta, L, W], lr=lr_adaptive)

            x0, y0, theta0 = (
                torch.tensor(p.x, device=self.device),
                torch.tensor(p.y, device=self.device),
                torch.tensor(p.theta, device=self.device),
            )
            L0 = torch.tensor(p.length, device=self.device)
            W0 = torch.tensor(p.width, device=self.device)

            for _ in range(num_steps_adaptive):
                opt.zero_grad()
                c, s = torch.cos(-theta), torch.sin(-theta)
                R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
                t = torch.stack([x, y])
                points_room = (wall_points - t) @ R.T
                loss_data = self.smooth_sdf_loss(points_room, L, W)
                # small priors to keep stability
                dth = ((theta - theta0 + math.pi) % (2 * math.pi)) - math.pi
                loss = (
                        loss_data
                        + self.pose_lambda * ((x - x0) ** 2 + (y - y0) ** 2 + dth ** 2)
                        + self.size_lambda * ((L - L0) ** 2 + (W - W0) ** 2)
                )
                loss.backward()
                # print if grads are non zero
                # if(L.grad > 0 and W.grad > 0 and x.grad >0 and y.grad >0 and theta.grad >0):
                #     print(f"  Grad L: {L.grad.item():.6f}, W: {W.grad.item():.6f}, x: {x.grad.item():.6f}, y: {y.grad.item():.6f}, theta: {theta.grad.item():.6f}")
                opt.step()

                with torch.no_grad():
                    theta.data = (theta.data + math.pi) % (2 * math.pi) - math.pi
                    L.data = torch.clamp(L.data, min=1.0)
                    W.data = torch.clamp(W.data, min=1.0)

                # In refine_best_particles_gradient(), after the optimization loop:
                # print(f"Refined: L={L.item():.3f}, W={W.item():.3f}, loss={loss.item():.6f}")

            # commit updates
            p.x = float(x.item())
            p.y = float(y.item())
            p.theta = float(theta.item())
            p.length = float(L.item())
            p.width = float(W.item())

    # -----------------------------------------------------------------
    # AUXILIARY METHODS -----------------------------------------------
    # -----------------------------------------------------------------
    def compute_pose_diversity(self):
        xs = [p.x for p in self.particles]
        ys = [p.y for p in self.particles]
        thetas = [p.theta for p in self.particles]
        return np.std(xs) + np.std(ys) + np.std(thetas)

    # -----------------------------------------------------------------
    def roughen_after_resample(self, sigma_xy=0.01, sigma_th=0.01):
        for p in self.particles:
            p.x += np.random.normal(0, sigma_xy)
            p.y += np.random.normal(0, sigma_xy)
            p.theta = ((p.theta + np.random.normal(0, sigma_th) + np.pi) % (2 * np.pi)) - np.pi

    # ---------------------------------------------------------------------
    # Utility: Signed distance to a centered rectangle (length, width)
    # ---------------------------------------------------------------------
    def sdf_rect(self, points, length, width):
        """
        Analytic 2D signed distance to an axis-aligned rectangle centered at (0,0)
        """
        # Line 463, replace with:
        if isinstance(length, torch.Tensor):
            half = torch.stack([length / 2.0, width / 2.0])
        else:
            half = torch.tensor([length / 2.0, width / 2.0], device=points.device)
        q = torch.abs(points) - half
        outside = torch.clamp(q, min=0.0)
        dist_out = outside.norm(dim=1)
        dist_in = torch.clamp(torch.max(q[:, 0], q[:, 1]), max=0.0)
        return dist_out + dist_in  # negative inside

    # ---------------------------------------------------------------------
    # Core smooth SDF loss (shared by PF & optimizer)
    # ---------------------------------------------------------------------
    def smooth_sdf_loss(self, points_room, L, W, margin=-0.05, delta=0.03):
        """
        Smooth, robust penalty for points inside and outside the room rectangle.
        Points are given in the room frame.
        """
        if points_room.numel() == 0:
            return torch.tensor(0.0, device=points_room.device)

        sdf = self.sdf_rect(points_room, L, W)  # signed distance to rectangle
        # penalize inside and outside points
        # r = torch.relu(sdf - margin)  # penalize outside walls
        # absr = torch.abs(sdf)
        absr = sdf
        quad = 0.5 * absr ** 2
        lin = delta * (absr - 0.5 * delta)
        hub = torch.where(absr <= delta, quad, lin)
        return hub.mean()

    # -----------------------------------------------------------------
    def best_particle(self):
        """Return particles with max weight and time smoothed version"""
        best = max(self.particles, key=lambda p: p.weight)

        # Smooth with previous best
        if self.smooth_particle is None:
            self.smooth_particle = deepcopy(best)
        else:
            alpha = 0.5  # smoothing factor
            self.smooth_particle.x = alpha * best.x + (1 - alpha) * self.smooth_particle.x
            self.smooth_particle.y = alpha * best.y + (1 - alpha) * self.smooth_particle.y
            # Handle theta wrap-around
            dtheta = ((best.theta - self.smooth_particle.theta + math.pi) % (2 * math.pi)) - math.pi
            self.smooth_particle.theta += alpha * dtheta
            self.smooth_particle.length = alpha * best.length + (1 - alpha) * self.smooth_particle.length
            self.smooth_particle.width = alpha * best.width + (1 - alpha) * self.smooth_particle.width
        return best, self.smooth_particle

        # -----------------------------------------------------------------

    def mean_weighted_particle(self):
        """Return the weighted mean particle."""
        N = len(self.particles)
        if N == 0:
            return None

        x_mean = sum(p.x * p.weight for p in self.particles)
        y_mean = sum(p.y * p.weight for p in self.particles)

        # For theta, use circular mean
        sin_sum = sum(math.sin(p.theta) * p.weight for p in self.particles)
        cos_sum = sum(math.cos(p.theta) * p.weight for p in self.particles)
        theta_mean = math.atan2(sin_sum, cos_sum)

        length_mean = sum(p.length * p.weight for p in self.particles)
        width_mean = sum(p.width * p.weight for p in self.particles)

        return Particle(
            x=x_mean,
            y=y_mean,
            theta=theta_mean,
            length=length_mean,
            width=width_mean,
            weight=1.0 / N
        )

    # -----------------------------------------------------------------
    @staticmethod
    def particle_to_pose(particle) -> Pose2D | None:
        if particle is None:
            return None
        return Pose2D(
            x=particle.x,
            y=particle.y,
            theta=particle.theta,
            length=particle.length,
            width=particle.width,
        )

    # -----------------------------------------------------------------
    def get_history(self):
        """
        Return diagnostic history for plotting.
        The dict always has these keys, even if features are disabled.
        """
        # ensure keys exist so the plotter never crashes
        for k in ["tick", "loss_best", "num_features", "ess", "births", "deaths", "n_particles", "ess_pct",
                  "weight_entropy"]:
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
        self.history.setdefault("weight_entropy", []).append(self.weight_entropy)
        self.history.setdefault("ess_pct", []).append(self.ess_pct)
        # Add pose diversity metrics
        xs = [p.x for p in self.particles]
        ys = [p.y for p in self.particles]
        thetas = [p.theta for p in self.particles]
        self.history.setdefault("x_std", []).append(float(np.std(xs)))
        self.history.setdefault("y_std", []).append(float(np.std(ys)))
        self.history.setdefault("theta_std", []).append(float(np.std(thetas)))