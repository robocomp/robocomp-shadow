"""
Concept: Rectangle - Rectangular Obstacle Belief Implementation

This module implements rectangular obstacle beliefs and their lifecycle management.
The RectangleManager class handles all instance creation, update, and removal.
"""
import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from obstacle_base import (
    ObstacleBeliefBase,
    ObstacleBeliefConfig,
    ObstacleManager,
    ObstacleType,
    DEVICE,
    DTYPE
)


@dataclass
class RectangleObstacleConfig(ObstacleBeliefConfig):
    """Configuration specific to rectangular obstacles."""
    prior_size_std: float = 5.0
    size_prior_mean: float = 40.0
    min_size: float = 15.0
    max_size: float = 100.0
    initial_size_std: float = 20.0
    initial_angle_std: float = 0.5


class RectangleManager(ObstacleManager):
    """
    Manages the lifecycle of RectangleObstacle instances.

    Handles:
    - Creating rectangles from point clusters (PCA-based fit)
    - Matching clusters to existing rectangles
    - Updating rectangles with new observations
    - Removing rectangles based on visibility logic
    - Filtering out wall false positives
    """

    def __init__(self, config: RectangleObstacleConfig = None,
                 room_width: float = 800.0, room_height: float = 600.0):
        super().__init__(config or RectangleObstacleConfig())
        self.rect_config = config or RectangleObstacleConfig()
        self.room_width = room_width
        self.room_height = room_height
        self.wall_margin = 30.0  # Reject detections this close to walls

    def _create_from_points(self, points: np.ndarray, robot_pose: np.ndarray) -> Optional['RectangleObstacle']:
        """
        Create a new RectangleObstacle from a point cluster using PCA fit.
        """
        if len(points) < 3:
            return None

        cx, cy, width, height, angle = self._fit_rectangle(points)

        # Reject if too close to walls (likely wall false positive)
        if (cx < self.wall_margin or cx > self.room_width - self.wall_margin or
            cy < self.wall_margin or cy > self.room_height - self.wall_margin):
            return None

        # Validate dimensions
        if width < self.rect_config.min_size or height < self.rect_config.min_size:
            return None
        if width > self.rect_config.max_size or height > self.rect_config.max_size:
            return None

        rect = RectangleObstacle(cx, cy, width, height, angle, self.rect_config)
        rect.confidence = 0.4
        return rect

    @staticmethod
    def _fit_rectangle(points: np.ndarray) -> Tuple[float, float, float, float, float]:
        """
        Fit oriented rectangle to points using PCA.

        Returns:
            (cx, cy, width, height, angle)
        """
        if len(points) < 3:
            cx, cy = np.mean(points, axis=0)
            return cx, cy, 50.0, 30.0, 0.0

        # Centroid
        centroid = np.mean(points, axis=0)

        # PCA for orientation
        centered = points - centroid
        cov = np.cov(centered.T)

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Principal axis is eigenvector with largest eigenvalue
            angle = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
            # Normalize to [-pi/2, pi/2]
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

        # Bounding box in rotated frame
        min_xy = np.min(rotated, axis=0)
        max_xy = np.max(rotated, axis=0)

        width = max(max_xy[0] - min_xy[0], 20.0)
        height = max(max_xy[1] - min_xy[1], 15.0)

        # Ensure width >= height (convention)
        if height > width:
            width, height = height, width
            angle += np.pi/2
            while angle > np.pi/2:
                angle -= np.pi
            while angle < -np.pi/2:
                angle += np.pi

        return centroid[0], centroid[1], width, height, angle


class RectangleObstacle(ObstacleBeliefBase):
    """Rectangular obstacle belief. State: [cx, cy, w, h, angle]"""
    def __init__(self, cx: float, cy: float, w: float, h: float, angle: float = 0.0,
                 config: Optional[RectangleObstacleConfig] = None,
                 device: torch.device = DEVICE):
        config = config or RectangleObstacleConfig()
        super().__init__(ObstacleType.RECTANGLE, config, device)
        self.rect_config = config
        self.mu = torch.tensor([cx, cy, w, h, angle], dtype=DTYPE, device=device)
        self.Sigma = torch.diag(torch.tensor([
            config.initial_position_std ** 2,
            config.initial_position_std ** 2,
            config.initial_size_std ** 2,
            config.initial_size_std ** 2,
            config.initial_angle_std ** 2
        ], dtype=DTYPE, device=device))
    def _transform_to_local(self, points: torch.Tensor) -> torch.Tensor:
        """Transform world points to rectangle's local frame."""
        center = self.mu[:2]
        angle = self.mu[4]
        translated = points - center
        cos_a, sin_a = torch.cos(-angle), torch.sin(-angle)
        local_x = translated[:, 0] * cos_a - translated[:, 1] * sin_a
        local_y = translated[:, 0] * sin_a + translated[:, 1] * cos_a
        return torch.stack([local_x, local_y], dim=1)
    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """SDF for oriented rectangle."""
        w, h = self.mu[2], self.mu[3]
        local_points = self._transform_to_local(points)
        half_w, half_h = w / 2, h / 2
        dx = torch.abs(local_points[:, 0]) - half_w
        dy = torch.abs(local_points[:, 1]) - half_h
        outside_x = torch.clamp(dx, min=0)
        outside_y = torch.clamp(dy, min=0)
        outside_dist = torch.sqrt(outside_x**2 + outside_y**2)
        inside_dist = torch.max(dx, dy)
        is_outside = (dx > 0) | (dy > 0)
        return torch.where(is_outside, outside_dist, inside_dist)
    def predict_range(self, ray_origin: torch.Tensor, ray_dirs: torch.Tensor) -> torch.Tensor:
        """Ray-rectangle intersection using slab method."""
        center = self.mu[:2]
        w, h, angle = self.mu[2], self.mu[3], self.mu[4]
        half_w, half_h = w / 2, h / 2
        K = ray_dirs.shape[0]
        cos_a, sin_a = torch.cos(-angle), torch.sin(-angle)
        local_origin = ray_origin - center
        lo_x = local_origin[0] * cos_a - local_origin[1] * sin_a
        lo_y = local_origin[0] * sin_a + local_origin[1] * cos_a
        ld_x = ray_dirs[:, 0] * cos_a - ray_dirs[:, 1] * sin_a
        ld_y = ray_dirs[:, 0] * sin_a + ray_dirs[:, 1] * cos_a
        ranges = torch.full((K,), 1e6, dtype=DTYPE, device=self.device)
        eps = 1e-10
        for k in range(K):
            dx, dy = ld_x[k], ld_y[k]
            if torch.abs(dx) > eps:
                tx1, tx2 = (-half_w - lo_x) / dx, (half_w - lo_x) / dx
                tx_min, tx_max = torch.min(tx1, tx2), torch.max(tx1, tx2)
            else:
                if lo_x < -half_w or lo_x > half_w:
                    continue
                tx_min, tx_max = torch.tensor(-1e10), torch.tensor(1e10)
            if torch.abs(dy) > eps:
                ty1, ty2 = (-half_h - lo_y) / dy, (half_h - lo_y) / dy
                ty_min, ty_max = torch.min(ty1, ty2), torch.max(ty1, ty2)
            else:
                if lo_y < -half_h or lo_y > half_h:
                    continue
                ty_min, ty_max = torch.tensor(-1e10), torch.tensor(1e10)
            t_enter = torch.max(tx_min, ty_min)
            t_exit = torch.min(tx_max, ty_max)
            if t_enter < t_exit and t_exit > 0:
                t = t_enter if t_enter > 0 else t_exit
                if t > 0:
                    ranges[k] = t
        return ranges
    def get_params_tuple(self) -> Tuple[float, float, float, float, float]:
        return tuple(self.mu[i].item() for i in range(5))
    def get_center(self) -> torch.Tensor:
        return self.mu[:2]
    def get_bounding_radius(self) -> float:
        w, h = self.mu[2].item(), self.mu[3].item()
        return np.sqrt((w/2)**2 + (h/2)**2)

    def compute_vfe(self, points: np.ndarray, pose: np.ndarray) -> float:
        """
        Compute Variational Free Energy for rectangular obstacle model.

        From paper Eq. obstacle_objective:

        F(s) = (1/2σ²_o) Σᵢ dᵢ(s)² + (1/2)(s-μₚ)ᵀΣₚ⁻¹(s-μₚ)

        where:
        - dᵢ(s) = SDF_rect(pᵢ, s)  (SDF for oriented rectangle)
        - First term: prediction error (negative accuracy)
        - Second term: complexity (prior regulariser / KL divergence)

        For model selection, rectangles have 5 parameters vs 3 for circles,
        resulting in higher complexity cost (Occam's razor).

        Args:
            points: LIDAR points [N, 2] numpy array
            pose: Robot pose [x, y, theta] numpy array

        Returns:
            VFE value (lower = better fit)
        """
        if len(points) == 0:
            return 1e6

        # Convert to tensor
        points_t = torch.tensor(points, dtype=DTYPE, device=self.device)

        # =====================================================================
        # PREDICTION ERROR TERM (Eq. cluster_likelihood extended to rectangles)
        # -ln p(C|s) ∝ (1/2σ²_o) Σᵢ dᵢ(s)²
        # =====================================================================
        sdf_values = self.sdf(points_t)  # SDF for oriented rectangle
        sigma_o = 5.0  # Observation noise σ_o
        prediction_error = torch.sum(sdf_values ** 2) / (2.0 * sigma_o ** 2)

        # =====================================================================
        # COMPLEXITY TERM (Prior regulariser / KL divergence)
        # Rectangle has MORE parameters than circle = higher complexity
        # This implements Occam's razor: prefer simpler models
        # =====================================================================
        n_params = 5  # Rectangle: (cx, cy, w, h, angle)

        # Log-determinant of covariance (uncertainty measure)
        log_det = torch.logdet(self.Sigma + 1e-6 * torch.eye(n_params, device=self.device))

        # Complexity = Occam factor + uncertainty penalty
        # Higher n_params = higher penalty (favors simpler circle model when fit is equal)
        complexity = 0.1 * n_params + 0.05 * torch.clamp(log_det, min=0.0)

        vfe = prediction_error.item() + complexity.item()
        return vfe
    def clone(self) -> 'RectangleObstacle':
        params = self.get_params_tuple()
        new_obs = RectangleObstacle(*params, self.rect_config, self.device)
        new_obs.mu = self.mu.clone()
        new_obs.Sigma = self.Sigma.clone()
        new_obs.confidence = self.confidence
        new_obs.age = self.age
        new_obs.last_seen = self.last_seen
        new_obs.observation_count = self.observation_count
        if self.historical_points is not None:
            new_obs.historical_points = self.historical_points.clone()
        return new_obs

    def optimize(self, points: np.ndarray, num_iters: int = 15, lr: float = 0.1):
        """
        Optimize rectangle parameters via gradient descent on SDF loss.

        Uses PyTorch autograd for gradient computation.
        Angle is stabilized with lower learning rate and momentum prior.

        F = (1/2σ²) Σ SDF² + angle_prior + angle_momentum + size_reg

        Args:
            points: LIDAR points [N, 2] numpy array
            num_iters: Number of gradient descent iterations
            lr: Learning rate
        """
        if len(points) < 3:
            return

        points_t = torch.tensor(points, dtype=DTYPE, device=self.device)

        # Store previous angle for momentum
        prev_angle = self.mu[4].item()

        # Make parameters require gradients
        mu = self.mu.clone().detach().requires_grad_(True)

        sigma_sq = 25.0  # Observation noise variance

        # Per-parameter learning rates: lower for angle to stabilize
        lr_pos = lr          # Position learning rate
        lr_size = lr         # Size learning rate
        lr_angle = lr * 0.3  # Angle learning rate (much lower for stability)

        for _ in range(num_iters):
            # Compute SDF with current parameters
            cx, cy, w, h, angle = mu[0], mu[1], mu[2], mu[3], mu[4]

            # Transform points to local frame
            cos_a, sin_a = torch.cos(-angle), torch.sin(-angle)
            local = points_t - torch.stack([cx, cy])
            local_x = local[:, 0] * cos_a - local[:, 1] * sin_a
            local_y = local[:, 0] * sin_a + local[:, 1] * cos_a

            # Rectangle SDF
            half_w, half_h = w / 2, h / 2
            dx = torch.abs(local_x) - half_w
            dy = torch.abs(local_y) - half_h

            outside = (dx > 0) | (dy > 0)
            sdf = torch.where(
                outside,
                torch.sqrt(torch.clamp(dx, min=0)**2 + torch.clamp(dy, min=0)**2 + 1e-6),
                torch.maximum(dx, dy)
            )

            # =====================================================================
            # LOSS = Prediction Error + Angle Priors + Size Regularization
            # =====================================================================

            # Prediction error (SDF loss)
            prediction_error = torch.sum(sdf ** 2) / (2.0 * sigma_sq)

            # Angle prior 1: prefer axis-aligned (0, ±90 degrees)
            # sin(2*angle)^2 has minima at 0, ±45, ±90 degrees
            axis_prior = 0.05 * torch.sin(2 * angle) ** 2

            # Angle prior 2: momentum - resist change from previous angle
            # This stabilizes the angle estimate over time
            angle_diff = angle - prev_angle
            # Wrap angle difference to [-pi/2, pi/2]
            while angle_diff > np.pi/2:
                angle_diff = angle_diff - np.pi
            while angle_diff < -np.pi/2:
                angle_diff = angle_diff + np.pi
            momentum_prior = 0.5 * angle_diff ** 2  # Strong resistance to change

            # Size regularization - prevent extreme sizes
            size_reg = 0.001 * (w**2 + h**2)

            # Total loss
            loss = prediction_error + axis_prior + momentum_prior + size_reg

            # Gradient via autograd
            if mu.grad is not None:
                mu.grad.zero_()
            loss.backward()

            # Update with per-parameter learning rates
            with torch.no_grad():
                grad = mu.grad
                if grad is not None and not torch.isnan(grad).any():
                    # Clip gradients
                    grad = torch.clamp(grad, -10.0, 10.0)
                    # Apply per-parameter learning rates
                    mu[0] -= lr_pos * grad[0]    # cx
                    mu[1] -= lr_pos * grad[1]    # cy
                    mu[2] -= lr_size * grad[2]   # w
                    mu[3] -= lr_size * grad[3]   # h
                    mu[4] -= lr_angle * grad[4]  # angle (lower LR)

                # Clamp dimensions
                mu[2] = torch.clamp(mu[2], self.rect_config.min_size, self.rect_config.max_size)
                mu[3] = torch.clamp(mu[3], self.rect_config.min_size, self.rect_config.max_size)

                # Normalize angle to [-pi/2, pi/2]
                while mu[4] > np.pi/2:
                    mu[4] -= np.pi
                while mu[4] < -np.pi/2:
                    mu[4] += np.pi

                # Check for NaN and reset if needed
                if torch.isnan(mu).any():
                    mu = self.mu.clone().detach()
                    break

            mu.requires_grad_(True)

        # Apply updates with EMA smoothing for angle
        new_mu = mu.detach()

        # Enforce convention: width >= height
        # If height > width, swap them and rotate by 90°
        if new_mu[3] > new_mu[2]:  # h > w
            new_mu[2], new_mu[3] = new_mu[3].clone(), new_mu[2].clone()  # swap w, h
            new_mu[4] = new_mu[4] + np.pi/2  # rotate 90°

        # EMA for angle: new_angle = alpha * optimized + (1-alpha) * previous
        alpha = 0.3  # Low alpha = more smoothing
        old_angle = self.mu[4].item()
        new_angle = new_mu[4].item()

        # Wrap difference for proper interpolation
        angle_diff = new_angle - old_angle
        while angle_diff > np.pi/2:
            angle_diff -= np.pi
        while angle_diff < -np.pi/2:
            angle_diff += np.pi

        smoothed_angle = old_angle + alpha * angle_diff
        # Normalize to [-pi/2, pi/2]
        while smoothed_angle > np.pi/2:
            smoothed_angle -= np.pi
        while smoothed_angle < -np.pi/2:
            smoothed_angle += np.pi

        new_mu[4] = smoothed_angle
        self.mu = new_mu

    def get_corners(self) -> torch.Tensor:
        """Get the 4 corners in world frame."""
        cx, cy = self.mu[0], self.mu[1]
        half_w, half_h = self.mu[2] / 2, self.mu[3] / 2
        angle = self.mu[4]
        local_corners = torch.tensor([
            [-half_w, -half_h], [half_w, -half_h],
            [half_w, half_h], [-half_w, half_h]
        ], dtype=DTYPE, device=self.device)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        rotated_x = local_corners[:, 0] * cos_a - local_corners[:, 1] * sin_a
        rotated_y = local_corners[:, 0] * sin_a + local_corners[:, 1] * cos_a
        return torch.stack([rotated_x + cx, rotated_y + cy], dim=1)
    @property
    def width(self) -> float:
        return self.mu[2].item()
    @property
    def height(self) -> float:
        return self.mu[3].item()
    @property
    def angle(self) -> float:
        return self.mu[4].item()


# =============================================================================
# REGISTRATION - Register rectangle as an obstacle type
# =============================================================================

def fit_rectangle_from_points(points: np.ndarray, robot_pose: np.ndarray = None) -> Optional[RectangleObstacle]:
    """
    Fit a RectangleObstacle from points using PCA fit.

    This is the factory function used by TrackedObstacle to create rectangle hypotheses.
    """
    if len(points) < 3:
        return None

    config = RectangleObstacleConfig()
    cx, cy, width, height, angle = RectangleManager._fit_rectangle(points)

    # Validate dimensions
    if width < config.min_size or height < config.min_size:
        return None
    if width > config.max_size or height > config.max_size:
        return None

    rect = RectangleObstacle(cx, cy, width, height, angle, config)
    rect.confidence = 0.4
    return rect


# Register rectangle type in the global registry
from obstacle_base import register_obstacle_type
register_obstacle_type(
    name='rectangle',
    cls=RectangleObstacle,
    config_class=RectangleObstacleConfig,
    fit_func=fit_rectangle_from_points,
    prior=1.0  # Equal prior with circles
)


if __name__ == "__main__":
    print("Testing RectangleObstacle...")
    rect = RectangleObstacle(100.0, 100.0, 60.0, 40.0, 0.0)
    print(f"Created: {rect}")
    test_points = torch.tensor([[100.0, 100.0], [130.0, 100.0]], dtype=DTYPE, device=rect.device)
    print(f"SDF: {rect.sdf(test_points).tolist()}")
    ray_origin = torch.tensor([0.0, 100.0], dtype=DTYPE, device=rect.device)
    ray_dirs = torch.tensor([[1.0, 0.0]], dtype=DTYPE, device=rect.device)
    print(f"Range: {rect.predict_range(ray_origin, ray_dirs).tolist()}")
    print("OK!")
