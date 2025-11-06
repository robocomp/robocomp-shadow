"""
Realistic ROBOSENSE Helios 3D Lidar Simulation
===============================================

Accurate simulation of ROBOSENSE Helios lidar specs:
- 32 vertical beams distributed from -11° to +55° pitch
- Horizontal 360° scanning (sampled at 180 rays = 2° resolution)
- Ray origin at lidar position (0, 0, lidar_z) not floor
- Only count wall hits where z > min_z (1.5m filter)

This gives 180 × 8 = 1440 raycasts per frame (sampling 32 beams with 8)
"""

import torch
import math
import numpy as np
from rich.console import Console

console = Console(highlight=False)


class RealisticHeliosRaycaster:
    """
    Realistic simulation of ROBOSENSE Helios 3D lidar.
    """

    def __init__(self,
                 num_horizontal_rays=180,  # One ray per 2 degrees
                 num_vertical_samples=8,  # Sample 32 beams with 8
                 lidar_height=1.1,  # Lidar Z position (160cm)
                 lidar_y_offset=0.0,  # Lidar Y offset from robot center
                 lidar_x_offset=0.0,  # Lidar X offset from robot center
                 min_z=1.5,  # Min z for filtered points
                 max_range=10.0,  # Max lidar range
                 min_pitch_deg=-11.0,  # Bottom beam angle
                 max_pitch_deg=55.0,  # Top beam angle
                 min_ray_hits=10):  # Min hits for wall visibility
        """
        Args:
            num_horizontal_rays: Number of rays in horizontal sweep (180 = 2° resolution)
            num_vertical_samples: Number of pitch angles to sample (8 samples of 32 beams)
            lidar_height: Height of lidar above floor (meters)
            min_z: Minimum z for point filtering (meters)
            max_range: Maximum lidar range (meters)
            min_pitch_deg: Minimum pitch angle (degrees, negative = downward)
            max_pitch_deg: Maximum pitch angle (degrees, positive = upward)
            min_ray_hits: Minimum ray hits for wall to be visible
        """
        self.num_horizontal_rays = num_horizontal_rays
        self.num_vertical_samples = num_vertical_samples
        self.lidar_height = lidar_height
        self.lidar_y_offset = lidar_y_offset
        self.lidar_x_offset = lidar_x_offset
        self.min_z = min_z
        self.max_range = max_range
        self.min_ray_hits = min_ray_hits

        # Convert pitch angles to radians
        self.min_pitch = math.radians(min_pitch_deg)
        self.max_pitch = math.radians(max_pitch_deg)

        # Pre-compute pitch angles (vertical beam angles)
        self.pitch_angles = np.linspace(
            self.min_pitch,
            self.max_pitch,
            num_vertical_samples
        )

        console.log(
            f"[cyan]Helios Raycaster: {num_horizontal_rays}×{num_vertical_samples} rays, "
            f"pitch [{min_pitch_deg:.1f}°, {max_pitch_deg:.1f}°][/cyan]"
        )

    def raycast_wall_visibility(self, particle, robot_pose_world=None):
        """
        Cast rays in 3D to simulate Helios lidar and count wall hits.

        Args:
            particle: Particle with room geometry (L, W, height, x, y, theta)
            robot_pose_world: (x, y, theta) in world frame, or None for origin

        Returns:
            wall_ray_hits: [int, int, int, int] - Valid hits per wall
            walls_visible: [bool, bool, bool, bool] - Visibility flags
        """
        if robot_pose_world is None:
            robot_pose_world = (0.0, 0.0, 0.0)

        rx_world, ry_world, rtheta_world = robot_pose_world

        # Ray origin in world frame: robot position + lidar height
        ray_origin_world = np.array([self.lidar_x_offset, self.lidar_y_offset, self.lidar_height])

        # Transform robot to room frame for wall intersection tests
        c = math.cos(-particle.theta)
        s = math.sin(-particle.theta)
        rx_room = c * (rx_world - particle.x) - s * (ry_world - particle.y)
        ry_room = s * (rx_world - particle.x) + c * (ry_world - particle.y)

        # Ray origin in room frame (for 2D wall tests)
        ray_origin_room = np.array([rx_room, ry_room])

        L, W = particle.length, particle.width
        room_height = getattr(particle, 'height', 2.5)

        # Count valid hits per wall
        wall_ray_hits = [0, 0, 0, 0]  # [+x, -x, +y, -y]
        hit_points = []

        # For each vertical pitch angle (8 samples of 32 beams)
        for pitch in self.pitch_angles:
            cos_pitch = math.cos(pitch)
            sin_pitch = math.sin(pitch)

            # For each horizontal direction (180 rays = 2° resolution)
            for i in range(self.num_horizontal_rays):
                # Horizontal angle (0 to 2π)
                yaw = 2.0 * math.pi * i / self.num_horizontal_rays

                # Ray direction in 3D (world frame, robot-relative)
                # First rotate around Z (yaw), then tilt (pitch)
                dx_2d = math.cos(yaw) * cos_pitch
                dy_2d = math.sin(yaw) * cos_pitch
                dz = sin_pitch

                # Apply robot's yaw to get world frame direction
                c_robot = math.cos(rtheta_world)
                s_robot = math.sin(rtheta_world)
                dx_world = dx_2d * c_robot - dy_2d * s_robot
                dy_world = dx_2d * s_robot + dy_2d * c_robot

                ray_dir_world = np.array([dx_world, dy_world, dz])

                # Transform ray direction to room frame (for 2D wall tests)
                dx_room = c * dx_world - s * dy_world
                dy_room = s * dx_world + c * dy_world
                ray_dir_room_2d = np.array([dx_room, dy_room])

                # Find which wall this ray hits (2D intersection)
                wall_hit, hit_distance_2d, hit_point_room_2d = self._raycast_to_walls_2d(
                    ray_origin_room, ray_dir_room_2d, L, W
                )

                if wall_hit is not None:
                    # Compute 3D distance and hit point
                    # We need actual 3D distance, not just 2D projection
                    t_3d = hit_distance_2d / cos_pitch if abs(cos_pitch) > 1e-6 else float('inf')

                    if t_3d <= self.max_range:
                        # Compute 3D hit point in world frame
                        hit_point_world = ray_origin_world + t_3d * ray_dir_world
                        hit_z = hit_point_world[2]
                        # Only count AND append if hit is within valid z-range
                        if hit_z >= self.min_z and hit_z <= room_height:
                            wall_ray_hits[wall_hit] += 1
                            hit_points.append(hit_point_world)

                        # Only count if hit is above min_z (simulates point filter)
                        if hit_z >= self.min_z and hit_z <= room_height:
                            wall_ray_hits[wall_hit] += 1

        # A wall is "visible" if it receives enough valid ray hits
        walls_visible = [hits >= self.min_ray_hits for hits in wall_ray_hits]

        return wall_ray_hits, walls_visible, hit_points

    def _raycast_to_walls_2d(self, ray_origin, ray_dir, L, W):
        """
        Cast a ray in 2D (XY plane) and find which wall it hits first.

        Args:
            ray_origin: [rx, ry] in room frame
            ray_dir: [dx, dy] in room frame (normalized)
            L, W: Room dimensions

        Returns:
            (wall_idx, distance, hit_point_2d) or (None, inf, None)
            wall_idx: 0=+x, 1=-x, 2=+y, 3=-y
        """
        rx, ry = ray_origin
        dx, dy = ray_dir

        min_distance = float('inf')
        hit_wall = None
        hit_point = None

        epsilon = 1e-6

        # Right wall (x = L/2)
        if abs(dx) > epsilon:
            t = (L / 2 - rx) / dx
            if t > 0:
                hit_y = ry + t * dy
                if -W / 2 <= hit_y <= W / 2:
                    if t < min_distance:
                        min_distance = t
                        hit_wall = 0
                        hit_point = np.array([L / 2, hit_y])

        # Left wall (x = -L/2)
        if abs(dx) > epsilon:
            t = (-L / 2 - rx) / dx
            if t > 0:
                hit_y = ry + t * dy
                if -W / 2 <= hit_y <= W / 2:
                    if t < min_distance:
                        min_distance = t
                        hit_wall = 1
                        hit_point = np.array([-L / 2, hit_y])

        # Front wall (y = W/2)
        if abs(dy) > epsilon:
            t = (W / 2 - ry) / dy
            if t > 0:
                hit_x = rx + t * dx
                if -L / 2 <= hit_x <= L / 2:
                    if t < min_distance:
                        min_distance = t
                        hit_wall = 2
                        hit_point = np.array([hit_x, W / 2])

        # Back wall (y = -W/2)
        if abs(dy) > epsilon:
            t = (-W / 2 - ry) / dy
            if t > 0:
                hit_x = rx + t * dx
                if -L / 2 <= hit_x <= L / 2:
                    if t < min_distance:
                        min_distance = t
                        hit_wall = 3
                        hit_point = np.array([hit_x, -W / 2])

        return hit_wall, min_distance, hit_point

    def compute_wall_specific_loss(self, particle, points, robot_pose_world=None):
        """
        Compute fitness loss with realistic Helios lidar visibility.

        Only penalize walls that:
        1. Receive valid ray hits (accounting for pitch, z-filter, range)
        2. Have sufficient point observations OR should have them

        Args:
            particle: Particle with room geometry
            points: torch.Tensor (N, 2) of points in world frame (XY only)
            robot_pose_world: (x, y, theta) in world frame

        Returns:
            float: Weighted loss from visible walls only
        """
        if points.numel() == 0:
            return 0.0

        # Raycast to determine which walls are visible
        wall_ray_hits, walls_visible, _ = self.raycast_wall_visibility(
            particle, robot_pose_world
        )

        # Transform points to room frame
        c = math.cos(-particle.theta)
        s = math.sin(-particle.theta)
        R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32, device=points.device)
        t = torch.tensor([particle.x, particle.y], dtype=torch.float32, device=points.device)
        points_room = (points - t) @ R.T

        # Compute loss per wall
        wall_losses, wall_counts = self._compute_per_wall_loss(
            points_room, particle.length, particle.width
        )

        # Only include walls with valid ray hits
        min_observations = 20
        valid_losses = []
        total_points = 0

        wall_names = ["Right(+x)", "Left(-x)", "Front(+y)", "Back(-y)"]

        for i, (loss, count, visible, ray_hits, name) in enumerate(
                zip(wall_losses, wall_counts, walls_visible, wall_ray_hits, wall_names)
        ):
            if visible:  # Wall received valid ray hits
                if count >= min_observations:
                    # Wall is visible and has observations
                    valid_losses.append(loss * count)
                    total_points += count
                else:
                    # Wall should be visible but has few observations
                    # Small penalty to encourage adjustment
                    penalty = 0.003
                    valid_losses.append(penalty * 10)
                    total_points += 10

                    if self._should_log():
                        console.log(
                            f"[yellow]⚠ {name}: {ray_hits} ray hits, only {count} point obs[/yellow]"
                        )
            # else: wall received no valid ray hits, don't penalize

        if total_points == 0:
            # No visible walls? Fall back to global loss
            return self._global_sdf_loss(points_room, particle.length, particle.width)

        return float(sum(valid_losses) / total_points)

    def _compute_per_wall_loss(self, points_room, L, W):
        """Compute loss and observation count for each wall."""
        band_width = 0.40

        wall_losses = []
        wall_counts = []

        # Wall 0: Right (+x at x=L/2)
        mask = (points_room[:, 0] > L / 2 - band_width) & (points_room[:, 0] < L / 2 + band_width)
        mask = mask & (points_room[:, 1] > -W / 2 - 0.2) & (points_room[:, 1] < W / 2 + 0.2)
        if mask.any():
            dist = torch.abs(points_room[mask, 0] - L / 2)
            loss = self._huber_loss(dist, delta=0.05).mean().item()
            wall_losses.append(loss)
            wall_counts.append(mask.sum().item())
        else:
            wall_losses.append(0.0)
            wall_counts.append(0)

        # Wall 1: Left (-x at x=-L/2)
        mask = (points_room[:, 0] > -L / 2 - band_width) & (points_room[:, 0] < -L / 2 + band_width)
        mask = mask & (points_room[:, 1] > -W / 2 - 0.2) & (points_room[:, 1] < W / 2 + 0.2)
        if mask.any():
            dist = torch.abs(points_room[mask, 0] + L / 2)
            loss = self._huber_loss(dist, delta=0.05).mean().item()
            wall_losses.append(loss)
            wall_counts.append(mask.sum().item())
        else:
            wall_losses.append(0.0)
            wall_counts.append(0)

        # Wall 2: Front (+y at y=W/2)
        mask = (points_room[:, 1] > W / 2 - band_width) & (points_room[:, 1] < W / 2 + band_width)
        mask = mask & (points_room[:, 0] > -L / 2 - 0.2) & (points_room[:, 0] < L / 2 + 0.2)
        if mask.any():
            dist = torch.abs(points_room[mask, 1] - W / 2)
            loss = self._huber_loss(dist, delta=0.05).mean().item()
            wall_losses.append(loss)
            wall_counts.append(mask.sum().item())
        else:
            wall_losses.append(0.0)
            wall_counts.append(0)

        # Wall 3: Back (-y at y=-W/2)
        mask = (points_room[:, 1] > -W / 2 - band_width) & (points_room[:, 1] < -W / 2 + band_width)
        mask = mask & (points_room[:, 0] > -L / 2 - 0.2) & (points_room[:, 0] < L / 2 + 0.2)
        if mask.any():
            dist = torch.abs(points_room[mask, 1] + W / 2)
            loss = self._huber_loss(dist, delta=0.05).mean().item()
            wall_losses.append(loss)
            wall_counts.append(mask.sum().item())
        else:
            wall_losses.append(0.0)
            wall_counts.append(0)

        return wall_losses, wall_counts

    def _huber_loss(self, x, delta=0.05):
        """Huber loss: quadratic near zero, linear far away."""
        abs_x = torch.abs(x)
        quadratic = 0.5 * abs_x ** 2
        linear = delta * (abs_x - 0.5 * delta)
        return torch.where(abs_x <= delta, quadratic, linear)

    def _global_sdf_loss(self, points_room, L, W):
        """Fallback: global SDF loss."""
        from room_particle_filter import smooth_sdf_loss
        return smooth_sdf_loss(points_room, L, W, margin=-0.02, delta=0.05).item()

    def _should_log(self):
        """Rate-limit logging."""
        self._log_counter = getattr(self, '_log_counter', 0) + 1
        return self._log_counter % 50 == 0

    def log_visibility_status(self, particle, robot_pose_world=None):
        """Log raycast results for debugging."""
        wall_ray_hits, walls_visible = self.raycast_wall_visibility(
            particle, robot_pose_world
        )

        wall_names = ["Right(+x)", "Left(-x)", "Front(+y)", "Back(-y)"]

        visibility_str = ", ".join([
            f"{name}={hits}hits{'✓' if vis else '✗'}"
            for name, hits, vis in zip(wall_names, wall_ray_hits, walls_visible)
        ])

        console.log(
            f"[cyan]Helios raycasts ({self.num_horizontal_rays}×{self.num_vertical_samples}={self.num_horizontal_rays * self.num_vertical_samples} rays): "
            f"{visibility_str}[/cyan]"
        )

        # Show occluded walls
        occluded = [name for name, vis in zip(wall_names, walls_visible) if not vis]
        if occluded:
            console.log(f"[yellow]  Occluded (no valid hits): {', '.join(occluded)}[/yellow]")

