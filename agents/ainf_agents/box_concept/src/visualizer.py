#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Visualizer for Box Concept Agent using DearPyGui

Displays:
- Room boundaries
- LIDAR points (raw and filtered)
- DBSCAN clusters with different colors
- Detected box beliefs
"""

import dearpygui.dearpygui as dpg
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import threading
import time


@dataclass
class VisualizerData:
    """Data container for visualization updates."""
    room_dims: Tuple[float, float] = (6.0, 6.0)  # (width, depth) in meters
    robot_pose: np.ndarray = None  # [x, y, theta]
    lidar_points_raw: np.ndarray = None  # Raw LIDAR points in room frame
    lidar_points_filtered: np.ndarray = None  # After wall filtering
    clusters: List[np.ndarray] = None  # List of point clusters
    beliefs: List[dict] = None  # List of belief info dicts

    def __post_init__(self):
        if self.robot_pose is None:
            self.robot_pose = np.array([0.0, 0.0, 0.0])
        if self.lidar_points_raw is None:
            self.lidar_points_raw = np.array([])
        if self.lidar_points_filtered is None:
            self.lidar_points_filtered = np.array([])
        if self.clusters is None:
            self.clusters = []
        if self.beliefs is None:
            self.beliefs = []


class BoxConceptVisualizer:
    """DearPyGui-based visualizer for the box concept agent."""

    # Color palette for clusters (RGB, 0-255)
    CLUSTER_COLORS = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (0, 255, 128),    # Spring Green
        (255, 0, 128),    # Rose
    ]

    def __init__(self, width: int = 800, height: int = 800):
        """Initialize the visualizer."""
        self.width = width
        self.height = height
        self.data = VisualizerData()
        self.lock = threading.Lock()
        self.running = False
        self.scale = 100  # pixels per meter
        self.center_x = width // 2
        self.center_y = height // 2

        self._setup_dpg()

    def _setup_dpg(self):
        """Setup DearPyGui context and window."""
        dpg.create_context()

        # Create main window
        with dpg.window(label="Box Concept Visualizer", tag="main_window",
                       width=self.width, height=self.height + 100):

            # Info text
            dpg.add_text("Room and LIDAR Visualization", tag="title_text")
            dpg.add_text("", tag="info_text")
            dpg.add_separator()

            # Drawing canvas
            with dpg.drawlist(width=self.width - 20, height=self.height - 50, tag="canvas"):
                pass

            # Controls
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_slider_float(label="Scale", default_value=100, min_value=20,
                                    max_value=200, tag="scale_slider", width=200,
                                    callback=self._on_scale_change)
                dpg.add_checkbox(label="Show Raw Points", default_value=True,
                               tag="show_raw_cb")
                dpg.add_checkbox(label="Show Filtered Points", default_value=True,
                               tag="show_filtered_cb")
                dpg.add_checkbox(label="Show Clusters", default_value=True,
                               tag="show_clusters_cb")
                dpg.add_checkbox(label="Show Beliefs", default_value=True,
                               tag="show_beliefs_cb")

        dpg.create_viewport(title="Box Concept Agent - Visualizer",
                           width=self.width + 20, height=self.height + 150)
        dpg.setup_dearpygui()

    def _on_scale_change(self, sender, app_data):
        """Handle scale slider change."""
        self.scale = app_data
        self._redraw()

    def _world_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        """Convert world coordinates (meters) to screen coordinates (pixels)."""
        screen_x = self.center_x + x * self.scale
        screen_y = self.center_y - y * self.scale  # Y is inverted in screen coords
        return screen_x, screen_y

    def _draw_room(self):
        """Draw room boundaries."""
        width, depth = self.data.room_dims
        half_w, half_d = width / 2, depth / 2

        # Room corners in world coordinates
        corners = [
            (-half_w, -half_d),
            (half_w, -half_d),
            (half_w, half_d),
            (-half_w, half_d),
            (-half_w, -half_d)  # Close the rectangle
        ]

        # Convert to screen coordinates
        screen_corners = [self._world_to_screen(x, y) for x, y in corners]

        # Draw room outline
        for i in range(len(screen_corners) - 1):
            dpg.draw_line(screen_corners[i], screen_corners[i + 1],
                         color=(100, 100, 100, 255), thickness=2, parent="canvas")

        # Draw axes
        origin = self._world_to_screen(0, 0)
        x_axis = self._world_to_screen(0.5, 0)
        y_axis = self._world_to_screen(0, 0.5)
        dpg.draw_arrow(x_axis, origin, color=(255, 0, 0, 255), thickness=2, parent="canvas")
        dpg.draw_arrow(y_axis, origin, color=(0, 255, 0, 255), thickness=2, parent="canvas")
        dpg.draw_text((origin[0] + 5, origin[1] + 5), "O", color=(255, 255, 255, 255),
                     size=12, parent="canvas")

    def _draw_robot(self):
        """Draw robot position and orientation.

        Convention: θ=0 means robot faces +Y direction.
        """
        if self.data.robot_pose is None:
            return

        x, y, theta = self.data.robot_pose
        pos = self._world_to_screen(x, y)

        # Draw robot as a triangle
        robot_size = 0.2 * self.scale  # Robot size in pixels

        # Adjust angle: robot convention is θ=0 facing +Y, but screen Y is inverted
        # So we need to rotate by 90° and account for screen Y inversion
        screen_theta = theta + np.pi / 2  # Rotate 90° to align with +Y convention
        cos_t, sin_t = np.cos(screen_theta), np.sin(screen_theta)

        # Front point (in direction of theta)
        front = (pos[0] + robot_size * cos_t, pos[1] - robot_size * sin_t)
        # Back left
        back_left = (pos[0] - robot_size * 0.5 * cos_t + robot_size * 0.4 * sin_t,
                    pos[1] + robot_size * 0.5 * sin_t + robot_size * 0.4 * cos_t)
        # Back right
        back_right = (pos[0] - robot_size * 0.5 * cos_t - robot_size * 0.4 * sin_t,
                     pos[1] + robot_size * 0.5 * sin_t - robot_size * 0.4 * cos_t)

        dpg.draw_triangle(front, back_left, back_right,
                         color=(0, 200, 255, 255), fill=(0, 150, 200, 200),
                         parent="canvas")

    def _draw_points(self, points: np.ndarray, color: Tuple[int, int, int],
                     radius: float = 2, max_points: int = 500):
        """Draw a set of points (subsampled for performance)."""
        if points is None or len(points) == 0:
            return

        # Subsample if too many points
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]

        for point in points:
            if len(point) >= 2:
                screen_pos = self._world_to_screen(point[0], point[1])
                dpg.draw_circle(screen_pos, radius, color=(*color, 255),
                               fill=(*color, 200), parent="canvas")

    def _draw_clusters(self):
        """Draw DBSCAN clusters with different colors."""
        if not self.data.clusters:
            return

        for i, cluster in enumerate(self.data.clusters):
            if cluster is None or len(cluster) == 0:
                continue

            color = self.CLUSTER_COLORS[i % len(self.CLUSTER_COLORS)]
            self._draw_points(cluster, color, radius=4)

            # Draw cluster centroid
            centroid = np.mean(cluster, axis=0)
            screen_pos = self._world_to_screen(centroid[0], centroid[1])
            dpg.draw_circle(screen_pos, 8, color=(*color, 255),
                           fill=(0, 0, 0, 0), thickness=2, parent="canvas")
            dpg.draw_text((screen_pos[0] + 10, screen_pos[1] - 10),
                         f"C{i}", color=(*color, 255), size=14, parent="canvas")

    def _draw_beliefs(self):
        """Draw detected box beliefs."""
        if not self.data.beliefs:
            return

        for belief in self.data.beliefs:
            cx = belief.get('cx', 0)
            cy = belief.get('cy', 0)
            w = belief.get('width', 0.5)
            h = belief.get('height', 0.5)
            angle = belief.get('angle', 0)
            confidence = belief.get('confidence', 0)
            belief_id = belief.get('id', -1)

            # Compute corners
            half_w, half_h = w / 2, h / 2
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            # Local corners
            local_corners = [
                (-half_w, -half_h),
                (half_w, -half_h),
                (half_w, half_h),
                (-half_w, half_h),
            ]

            # Transform to world and then to screen
            screen_corners = []
            for lx, ly in local_corners:
                wx = cx + lx * cos_a - ly * sin_a
                wy = cy + lx * sin_a + ly * cos_a
                screen_corners.append(self._world_to_screen(wx, wy))

            # Draw box outline
            alpha = int(255 * confidence)
            for i in range(4):
                dpg.draw_line(screen_corners[i], screen_corners[(i + 1) % 4],
                             color=(255, 165, 0, alpha), thickness=3, parent="canvas")

            # Draw center and label
            center_screen = self._world_to_screen(cx, cy)
            dpg.draw_circle(center_screen, 5, color=(255, 165, 0, 255),
                           fill=(255, 165, 0, 200), parent="canvas")
            dpg.draw_text((center_screen[0] + 10, center_screen[1] - 20),
                         f"Box {belief_id}\nconf: {confidence:.2f}",
                         color=(255, 200, 100, 255), size=12, parent="canvas")

    def _redraw(self):
        """Redraw all elements."""
        # Clear canvas
        dpg.delete_item("canvas", children_only=True)

        # Update scale from slider
        self.scale = dpg.get_value("scale_slider")

        # Draw elements
        self._draw_room()
        self._draw_robot()

        # Draw raw points (gray, small)
        if dpg.get_value("show_raw_cb"):
            self._draw_points(self.data.lidar_points_raw, (128, 128, 128), radius=1)

        # Draw filtered points (white, medium)
        if dpg.get_value("show_filtered_cb"):
            self._draw_points(self.data.lidar_points_filtered, (200, 200, 200), radius=2)

        # Draw clusters
        if dpg.get_value("show_clusters_cb"):
            self._draw_clusters()

        # Draw beliefs
        if dpg.get_value("show_beliefs_cb"):
            self._draw_beliefs()

        # Update info text
        with self.lock:
            n_raw = len(self.data.lidar_points_raw) if self.data.lidar_points_raw is not None else 0
            n_filtered = len(self.data.lidar_points_filtered) if self.data.lidar_points_filtered is not None else 0
            n_clusters = len(self.data.clusters) if self.data.clusters else 0
            n_beliefs = len(self.data.beliefs) if self.data.beliefs else 0

            info = f"Raw: {n_raw} | Filtered: {n_filtered} | Clusters: {n_clusters} | Beliefs: {n_beliefs}"
            info += f" | Room: {self.data.room_dims[0]:.1f}x{self.data.room_dims[1]:.1f}m"
            if self.data.robot_pose is not None:
                info += f" | Robot: ({self.data.robot_pose[0]:.2f}, {self.data.robot_pose[1]:.2f})"

        dpg.set_value("info_text", info)

    def update(self, room_dims: Tuple[float, float] = None,
               robot_pose: np.ndarray = None,
               lidar_points_raw: np.ndarray = None,
               lidar_points_filtered: np.ndarray = None,
               clusters: List[np.ndarray] = None,
               beliefs: List[dict] = None):
        """Update visualization data (thread-safe)."""
        with self.lock:
            if room_dims is not None:
                self.data.room_dims = room_dims
            if robot_pose is not None:
                self.data.robot_pose = robot_pose.copy() if isinstance(robot_pose, np.ndarray) else robot_pose
            if lidar_points_raw is not None:
                self.data.lidar_points_raw = lidar_points_raw.copy() if len(lidar_points_raw) > 0 else np.array([])
            if lidar_points_filtered is not None:
                self.data.lidar_points_filtered = lidar_points_filtered.copy() if len(lidar_points_filtered) > 0 else np.array([])
            if clusters is not None:
                self.data.clusters = [c.copy() for c in clusters if c is not None and len(c) > 0]
            if beliefs is not None:
                self.data.beliefs = beliefs.copy()

    def start(self):
        """Start the visualizer (blocking)."""
        self.running = True
        dpg.show_viewport()

        target_fps = 15  # Limit to 15 FPS to save CPU
        frame_time = 1.0 / target_fps

        while dpg.is_dearpygui_running() and self.running:
            start_time = time.time()
            self._redraw()
            dpg.render_dearpygui_frame()

            # Limit FPS
            elapsed = time.time() - start_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

        dpg.destroy_context()

    def start_async(self):
        """Start the visualizer in a separate thread."""
        self.thread = threading.Thread(target=self.start, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the visualizer."""
        self.running = False


# Singleton instance for easy access
_visualizer_instance: Optional[BoxConceptVisualizer] = None


def get_visualizer() -> BoxConceptVisualizer:
    """Get or create the global visualizer instance."""
    global _visualizer_instance
    if _visualizer_instance is None:
        _visualizer_instance = BoxConceptVisualizer()
    return _visualizer_instance


def start_visualizer_async():
    """Start the visualizer in a background thread."""
    viz = get_visualizer()
    viz.start_async()
    return viz


# Test code
if __name__ == "__main__":
    import random

    viz = BoxConceptVisualizer()

    # Simulate some data
    room_dims = (6.0, 5.0)
    robot_pose = np.array([0.5, 0.5, 0.3])

    # Generate random LIDAR points
    n_points = 200
    angles = np.linspace(0, 2 * np.pi, n_points)
    distances = 2.0 + 0.5 * np.random.randn(n_points)
    raw_points = np.column_stack([
        robot_pose[0] + distances * np.cos(angles + robot_pose[2]),
        robot_pose[1] + distances * np.sin(angles + robot_pose[2])
    ])

    # Simulate filtered points (subset)
    filtered_points = raw_points[::2]

    # Simulate clusters
    clusters = [
        np.array([[1.0, 1.0], [1.1, 1.0], [1.0, 1.1], [1.05, 1.05]]),
        np.array([[-1.0, 0.5], [-1.1, 0.5], [-1.0, 0.6], [-0.95, 0.55]]),
    ]

    # Simulate beliefs
    beliefs = [
        {'id': 0, 'cx': 1.0, 'cy': 1.0, 'width': 0.4, 'height': 0.3, 'angle': 0.1, 'confidence': 0.8},
        {'id': 1, 'cx': -1.0, 'cy': 0.5, 'width': 0.35, 'height': 0.35, 'angle': -0.2, 'confidence': 0.5},
    ]

    viz.update(room_dims=room_dims, robot_pose=robot_pose,
              lidar_points_raw=raw_points, lidar_points_filtered=filtered_points,
              clusters=clusters, beliefs=beliefs)

    viz.start()
