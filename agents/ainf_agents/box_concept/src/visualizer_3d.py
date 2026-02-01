#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
3D Visualizer for Box Concept Agent using Open3D

Displays:
- Room boundaries (floor and walls)
- Robot position and orientation
- LIDAR points (raw and filtered)
- DBSCAN clusters with different colors
- Detected box beliefs as 3D boxes
- SDF values and confidence
"""

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import threading
import time


@dataclass
class Visualizer3DData:
    """Data container for 3D visualization updates."""
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


class BoxConceptVisualizer3D:
    """Open3D-based 3D visualizer for the box concept agent."""

    # Color palette for clusters (RGB, 0-1)
    CLUSTER_COLORS = [
        [1.0, 0.3, 0.3],      # Red
        [0.3, 1.0, 0.3],      # Green
        [0.3, 0.3, 1.0],      # Blue
        [1.0, 1.0, 0.3],      # Yellow
        [1.0, 0.3, 1.0],      # Magenta
        [0.3, 1.0, 1.0],      # Cyan
        [1.0, 0.6, 0.2],      # Orange
        [0.6, 0.2, 1.0],      # Purple
        [0.2, 1.0, 0.6],      # Spring Green
        [1.0, 0.2, 0.6],      # Rose
    ]

    def __init__(self, width: int = 640, height: int = 400):
        """Initialize the 3D visualizer."""
        self.width = width
        self.height = height
        self.data = Visualizer3DData()
        self.lock = threading.Lock()
        self.running = False
        self.vis = None
        self.geometries = {}
        self.needs_update = True

        # Settings
        self.show_raw_points = False
        self.show_filtered_points = True
        self.show_clusters = True
        self.show_beliefs = True
        self.show_room = True
        self.point_size = 3.0

    def _create_room_geometry(self) -> List[o3d.geometry.LineSet]:
        """Create room floor and wall wireframes."""
        width, depth = self.data.room_dims
        half_w, half_d = width / 2, depth / 2
        wall_height = 0.5  # meters

        geometries = []

        # Floor outline
        floor_points = np.array([
            [-half_w, -half_d, 0],
            [half_w, -half_d, 0],
            [half_w, half_d, 0],
            [-half_w, half_d, 0],
        ])
        floor_lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
        floor = o3d.geometry.LineSet()
        floor.points = o3d.utility.Vector3dVector(floor_points)
        floor.lines = o3d.utility.Vector2iVector(floor_lines)
        floor.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]] * 4)
        geometries.append(floor)

        # Wall outlines (vertical lines at corners)
        for corner in floor_points:
            wall_line = o3d.geometry.LineSet()
            wall_line.points = o3d.utility.Vector3dVector([
                corner,
                corner + np.array([0, 0, wall_height])
            ])
            wall_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            wall_line.colors = o3d.utility.Vector3dVector([[0.4, 0.4, 0.4]])
            geometries.append(wall_line)

        # Top of walls
        top_points = floor_points + np.array([0, 0, wall_height])
        top = o3d.geometry.LineSet()
        top.points = o3d.utility.Vector3dVector(top_points)
        top.lines = o3d.utility.Vector2iVector(floor_lines)
        top.colors = o3d.utility.Vector3dVector([[0.4, 0.4, 0.4]] * 4)
        geometries.append(top)

        # Origin axes
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        geometries.append(axes)

        return geometries

    def _create_robot_geometry(self) -> o3d.geometry.TriangleMesh:
        """Create robot representation as a cone/arrow."""
        if self.data.robot_pose is None:
            return None

        x, y, theta = self.data.robot_pose

        # Create a cone pointing in the direction of theta
        robot = o3d.geometry.TriangleMesh.create_cone(radius=0.15, height=0.3)
        robot.compute_vertex_normals()

        # Rotate to point in XY plane (cone starts pointing up in Z)
        # Add pi/2 to align with robot convention (theta=0 means facing +Y)
        R = robot.get_rotation_matrix_from_xyz((np.pi/2, 0, theta + np.pi/2))
        robot.rotate(R, center=(0, 0, 0))

        # Translate to robot position
        robot.translate([x, y, 0.15])

        # Color: cyan
        robot.paint_uniform_color([0.0, 0.8, 1.0])

        return robot

    def _create_points_geometry(self, points: np.ndarray, color: List[float],
                                  height: float = 0.1) -> o3d.geometry.PointCloud:
        """Create point cloud geometry from 2D points."""
        if points is None or len(points) == 0:
            return None

        # Add Z coordinate (height)
        points_3d = np.column_stack([points[:, 0], points[:, 1],
                                      np.full(len(points), height)])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector([color] * len(points))

        return pcd

    def _create_cluster_geometries(self) -> List[o3d.geometry.PointCloud]:
        """Create point clouds for each cluster with different colors."""
        geometries = []

        if not self.data.clusters:
            return geometries

        for i, cluster in enumerate(self.data.clusters):
            if cluster is None or len(cluster) == 0:
                continue

            color = self.CLUSTER_COLORS[i % len(self.CLUSTER_COLORS)]
            # Use slightly different height for visibility
            pcd = self._create_points_geometry(cluster, color, height=0.05)
            if pcd:
                geometries.append(pcd)

        return geometries

    def _create_box_geometry(self, belief: dict) -> List[o3d.geometry.Geometry]:
        """Create 3D box geometry for a belief."""
        geometries = []

        cx = belief.get('cx', 0)
        cy = belief.get('cy', 0)
        w = belief.get('width', 0.5)
        h = belief.get('height', 0.5)
        d = belief.get('depth', 0.5)
        angle = belief.get('angle', 0)
        confidence = belief.get('confidence', 0)
        sdf_mean = belief.get('sdf_mean', 0)

        # Create oriented bounding box
        # Box sits on the ground (z=0 to z=d)
        center = np.array([cx, cy, d/2])
        extent = np.array([w, h, d])

        # Rotation around Z axis
        R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz((0, 0, angle))

        obb = o3d.geometry.OrientedBoundingBox(center, R, extent)

        # Color based on SDF quality (green = good fit, red = poor fit)
        sdf_quality = max(0, min(1, 1.0 - sdf_mean * 5))
        box_color = [1.0 - sdf_quality, sdf_quality, 0.0]
        obb.color = box_color

        geometries.append(obb)

        # Create a solid semi-transparent box mesh for better visibility
        box_mesh = o3d.geometry.TriangleMesh.create_box(w, h, d)
        box_mesh.translate([-w/2, -h/2, 0])  # Center at origin
        box_mesh.rotate(R, center=(0, 0, 0))
        box_mesh.translate([cx, cy, 0])
        box_mesh.compute_vertex_normals()

        # Color with alpha based on confidence
        alpha_color = [box_color[0] * confidence,
                       box_color[1] * confidence,
                       box_color[2] * 0.5 + 0.2]
        box_mesh.paint_uniform_color(alpha_color)

        geometries.append(box_mesh)

        return geometries

    def _create_belief_geometries(self) -> List[o3d.geometry.Geometry]:
        """Create geometries for all beliefs."""
        geometries = []

        if not self.data.beliefs:
            return geometries

        for belief in self.data.beliefs:
            box_geoms = self._create_box_geometry(belief)
            geometries.extend(box_geoms)

        return geometries

    def _rebuild_scene(self):
        """Rebuild all geometries."""
        all_geometries = []

        with self.lock:
            # Room
            if self.show_room:
                all_geometries.extend(self._create_room_geometry())

            # Robot
            robot = self._create_robot_geometry()
            if robot:
                all_geometries.append(robot)

            # Raw points
            if self.show_raw_points and self.data.lidar_points_raw is not None:
                pcd = self._create_points_geometry(self.data.lidar_points_raw,
                                                    [0.5, 0.5, 0.5], height=0.02)
                if pcd:
                    all_geometries.append(pcd)

            # Filtered points
            if self.show_filtered_points and self.data.lidar_points_filtered is not None:
                pcd = self._create_points_geometry(self.data.lidar_points_filtered,
                                                    [0.8, 0.8, 0.8], height=0.08)
                if pcd:
                    all_geometries.append(pcd)

            # Clusters
            if self.show_clusters:
                all_geometries.extend(self._create_cluster_geometries())

            # Beliefs (boxes)
            if self.show_beliefs:
                all_geometries.extend(self._create_belief_geometries())

        return all_geometries

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
                self.data.beliefs = beliefs.copy() if beliefs else []

            self.needs_update = True

    def start(self):
        """Start the visualizer (blocking)."""
        self.running = True
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Box Concept 3D Visualizer", self.width, self.height)

        # Set render options
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.15])
        opt.point_size = self.point_size
        opt.line_width = 2.0

        # Initial geometry
        geometries = self._rebuild_scene()
        for geom in geometries:
            self.vis.add_geometry(geom)

        # Set initial viewpoint (bird's eye view)
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.5)
        ctr.set_front([0, -0.3, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])

        # Main loop
        frame_count = 0
        update_interval = 5  # Update geometries every N frames

        while self.running:
            frame_count += 1

            # Update geometries periodically
            if self.needs_update and frame_count % update_interval == 0:
                self.needs_update = False

                # Clear and rebuild
                self.vis.clear_geometries()
                geometries = self._rebuild_scene()
                for geom in geometries:
                    self.vis.add_geometry(geom)

            # Poll events and update renderer
            if not self.vis.poll_events():
                break
            self.vis.update_renderer()

            time.sleep(0.016)  # ~60 FPS limit

        self.vis.destroy_window()

    def start_async(self):
        """Start the visualizer in a separate thread."""
        self.thread = threading.Thread(target=self.start, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the visualizer."""
        self.running = False


# Convenience functions
_visualizer_3d_instance: Optional[BoxConceptVisualizer3D] = None


def get_visualizer_3d() -> BoxConceptVisualizer3D:
    """Get or create the global 3D visualizer instance."""
    global _visualizer_3d_instance
    if _visualizer_3d_instance is None:
        _visualizer_3d_instance = BoxConceptVisualizer3D()
    return _visualizer_3d_instance


def start_visualizer_3d_async():
    """Start the 3D visualizer in a background thread."""
    viz = get_visualizer_3d()
    viz.start_async()
    return viz


# Test code
if __name__ == "__main__":
    viz = BoxConceptVisualizer3D()

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
        np.array([[1.0, 1.0], [1.1, 1.0], [1.0, 1.1], [1.05, 1.05],
                  [1.0, 0.9], [0.9, 1.0], [1.1, 1.1], [0.95, 0.95]]),
        np.array([[-1.0, 0.5], [-1.1, 0.5], [-1.0, 0.6], [-0.95, 0.55],
                  [-1.05, 0.45], [-0.9, 0.5], [-1.0, 0.4]]),
    ]

    # Simulate beliefs
    beliefs = [
        {'id': 0, 'cx': 1.0, 'cy': 1.0, 'width': 0.4, 'height': 0.3, 'depth': 0.5,
         'angle': 0.1, 'confidence': 0.8, 'sdf_mean': 0.02},
        {'id': 1, 'cx': -1.0, 'cy': 0.5, 'width': 0.35, 'height': 0.35, 'depth': 0.4,
         'angle': -0.2, 'confidence': 0.5, 'sdf_mean': 0.15},
    ]

    viz.update(room_dims=room_dims, robot_pose=robot_pose,
               lidar_points_raw=raw_points, lidar_points_filtered=filtered_points,
               clusters=clusters, beliefs=beliefs)

    print("Starting 3D visualizer...")
    print("Controls:")
    print("  - Left mouse: Rotate")
    print("  - Right mouse: Pan")
    print("  - Scroll: Zoom")
    print("  - Close window to exit")

    viz.start()
