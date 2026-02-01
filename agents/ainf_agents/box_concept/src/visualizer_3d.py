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
import os
import json
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import threading
import time

# Path to the robot mesh
ROBOT_MESH_PATH = os.path.join(os.path.dirname(__file__), "meshes", "shadow.obj")
# Path to save camera settings
CAMERA_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), ".camera_settings.json")


@dataclass
class Visualizer3DData:
    """Data container for 3D visualization updates."""
    room_dims: Tuple[float, float] = (6.0, 6.0)  # (width, depth) in meters
    robot_pose: np.ndarray = None  # [x, y, theta]
    lidar_points_raw: np.ndarray = None  # Raw LIDAR points in room frame
    lidar_points_filtered: np.ndarray = None  # After wall filtering
    clusters: List[np.ndarray] = None  # List of point clusters
    beliefs: List[dict] = None  # List of belief info dicts
    historical_points: Dict[int, np.ndarray] = None  # Per-belief historical points

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
        if self.historical_points is None:
            self.historical_points = {}


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
        self.current_geometries = []  # Track current geometries for removal
        self.needs_update = True

        # Settings
        self.show_raw_points = False
        self.show_filtered_points = True
        self.show_clusters = True
        self.show_beliefs = True
        self.show_room = True
        self.show_historical_points = True
        self.point_size = 2.0

    def _create_room_geometry(self) -> List[o3d.geometry.LineSet]:
        """Create room floor with grid and wall wireframes."""
        width, depth = self.data.room_dims
        half_w, half_d = width / 2, depth / 2
        wall_height = 0.5  # meters

        geometries = []

        # Create floor grid
        grid_spacing = 0.5  # meters between grid lines
        grid_points = []
        grid_lines = []
        point_idx = 0

        # Horizontal lines (along X axis)
        y = -half_d
        while y <= half_d + 0.01:
            grid_points.append([-half_w, y, 0])
            grid_points.append([half_w, y, 0])
            grid_lines.append([point_idx, point_idx + 1])
            point_idx += 2
            y += grid_spacing

        # Vertical lines (along Y axis)
        x = -half_w
        while x <= half_w + 0.01:
            grid_points.append([x, -half_d, 0])
            grid_points.append([x, half_d, 0])
            grid_lines.append([point_idx, point_idx + 1])
            point_idx += 2
            x += grid_spacing

        floor_grid = o3d.geometry.LineSet()
        floor_grid.points = o3d.utility.Vector3dVector(np.array(grid_points))
        floor_grid.lines = o3d.utility.Vector2iVector(grid_lines)
        floor_grid.colors = o3d.utility.Vector3dVector([[0.3, 0.3, 0.35]] * len(grid_lines))
        geometries.append(floor_grid)

        # Floor outline (thicker border)
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
        floor.colors = o3d.utility.Vector3dVector([[0.6, 0.6, 0.6]] * 4)
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
        """Create robot representation using shadow.obj mesh."""
        if self.data.robot_pose is None:
            return None

        x, y, theta = self.data.robot_pose

        # Try to load the robot mesh
        if os.path.exists(ROBOT_MESH_PATH):
            try:
                robot = o3d.io.read_triangle_mesh(ROBOT_MESH_PATH)
                robot.compute_vertex_normals()

                # The mesh is already in meters (about 0.48m x 0.48m x 1.4m)
                # Scale it down to a reasonable size for visualization (e.g., 0.3m wide)
                current_size = robot.get_axis_aligned_bounding_box().get_extent()
                scale_factor = 0.3 / max(current_size[0], current_size[1])  # Scale to ~30cm width
                robot.scale(scale_factor, center=(0, 0, 0))

                # Move to sit on floor (translate so bottom is at z=0)
                bounds = robot.get_axis_aligned_bounding_box()
                min_z = bounds.min_bound[2]
                robot.translate([0, 0, -min_z])

                # Rotate to align with robot orientation
                R = robot.get_rotation_matrix_from_xyz((0, 0, theta))
                robot.rotate(R, center=(0, 0, 0))

                # Translate to robot position
                robot.translate([x, y, 0])

                # Color: cyan
                robot.paint_uniform_color([0.0, 0.8, 1.0])

                return robot
            except Exception as e:
                print(f"Failed to load robot mesh: {e}")

        # Fallback to cone if mesh loading fails
        robot = o3d.geometry.TriangleMesh.create_cone(radius=0.08, height=0.15)
        robot.compute_vertex_normals()
        R = robot.get_rotation_matrix_from_xyz((np.pi/2, 0, theta))
        robot.rotate(R, center=(0, 0, 0))
        robot.translate([x, y, 0.08])
        robot.paint_uniform_color([0.0, 0.8, 1.0])

        return robot

    def _create_points_geometry(self, points: np.ndarray, color: List[float],
                                  height: float = 0.1) -> o3d.geometry.PointCloud:
        """Create point cloud geometry from 2D or 3D points.

        For 2D points [N, 2]: adds height as Z coordinate.
        For 3D points [N, 3]: uses the Z coordinate directly.
        """
        if points is None or len(points) == 0:
            return None

        # Check if points are already 3D
        if points.shape[1] >= 3:
            # Use the actual Z coordinates
            points_3d = points[:, :3]
        else:
            # Add Z coordinate (height) for 2D points
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
            # Use slightly lighter/muted color for cluster points
            muted_color = [c * 0.6 for c in color]  # Darker/lighter version
            pcd = self._create_points_geometry(cluster, muted_color, height=0.05)
            if pcd:
                geometries.append(pcd)

        return geometries

    def _create_box_geometry(self, belief: dict) -> List[o3d.geometry.Geometry]:
        """Create 3D box geometry for a belief (semi-transparent wireframe)."""
        geometries = []

        cx = belief.get('cx', 0)
        cy = belief.get('cy', 0)
        w = belief.get('width', 0.5)
        h = belief.get('height', 0.5)
        d = belief.get('depth', 0.5)
        angle = belief.get('angle', 0)
        confidence = belief.get('confidence', 0)
        sdf_mean = belief.get('sdf_mean', 0)

        # Color based on SDF quality (green = good fit, red = poor fit)
        sdf_quality = max(0.0, min(1.0, 1.0 - sdf_mean * 5))
        box_color = [1.0 - sdf_quality, sdf_quality, 0.0]

        # Create wireframe box using LineSet for transparency effect
        half_w, half_h = w / 2, h / 2
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # 8 corners of the box in local frame
        local_corners = np.array([
            [-half_w, -half_h, 0],
            [half_w, -half_h, 0],
            [half_w, half_h, 0],
            [-half_w, half_h, 0],
            [-half_w, -half_h, d],
            [half_w, -half_h, d],
            [half_w, half_h, d],
            [-half_w, half_h, d],
        ])

        # Rotate and translate corners
        world_corners = []
        for corner in local_corners:
            rx = corner[0] * cos_a - corner[1] * sin_a + cx
            ry = corner[0] * sin_a + corner[1] * cos_a + cy
            rz = corner[2]
            world_corners.append([rx, ry, rz])
        world_corners = np.array(world_corners)

        # 12 edges of the box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
        ]

        # Create wireframe
        wireframe = o3d.geometry.LineSet()
        wireframe.points = o3d.utility.Vector3dVector(world_corners)
        wireframe.lines = o3d.utility.Vector2iVector(edges)
        # Color intensity based on confidence
        edge_color = [c * (0.5 + 0.5 * confidence) for c in box_color]
        wireframe.colors = o3d.utility.Vector3dVector([edge_color] * len(edges))
        geometries.append(wireframe)

        # Add semi-transparent faces using very light colored mesh
        # Only show top face for less visual clutter
        if confidence > 0.3:
            # Top face as a thin quad
            top_mesh = o3d.geometry.TriangleMesh()
            top_mesh.vertices = o3d.utility.Vector3dVector(world_corners[4:8])
            top_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
            top_mesh.compute_vertex_normals()
            # Very light color for transparency effect
            alpha = 0.2 + 0.3 * confidence
            top_mesh.paint_uniform_color([box_color[0] * alpha, box_color[1] * alpha, box_color[2] * alpha + 0.1])
            geometries.append(top_mesh)

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

    def _create_historical_points_geometry(self) -> List[o3d.geometry.PointCloud]:
        """Create point clouds for historical points of each belief.

        Historical points are 3D (x, y, z) and already contain the original
        Z coordinate from the lidar scan, so we use them directly.
        """
        geometries = []

        if not self.data.historical_points:
            return geometries

        for belief_id, points in self.data.historical_points.items():
            if points is None or len(points) == 0:
                continue

            # Bright yellow/gold color for historical points to stand out
            hist_color = [1.0, 0.9, 0.2]  # Bright yellow

            # Points should be [N, 3] with real Z coordinates from lidar
            if points.shape[1] == 3:
                # Use the 3D points directly
                points_3d = points
            else:
                # Fallback: if only 2D, place at lidar height
                lidar_height = 0.15
                points_3d = np.column_stack([points[:, 0], points[:, 1],
                                              np.full(len(points), lidar_height)])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            pcd.colors = o3d.utility.Vector3dVector([hist_color] * len(points_3d))
            geometries.append(pcd)

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

            # Historical points (accumulated evidence)
            if self.show_historical_points:
                all_geometries.extend(self._create_historical_points_geometry())

            # Beliefs (boxes)
            if self.show_beliefs:
                all_geometries.extend(self._create_belief_geometries())

        return all_geometries

    def update(self, room_dims: Tuple[float, float] = None,
               robot_pose: np.ndarray = None,
               lidar_points_raw: np.ndarray = None,
               lidar_points_filtered: np.ndarray = None,
               clusters: List[np.ndarray] = None,
               beliefs: List[dict] = None,
               historical_points: Dict[int, np.ndarray] = None):
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
            if historical_points is not None:
                self.data.historical_points = {k: v.copy() for k, v in historical_points.items()}
            if beliefs is not None:
                self.data.beliefs = beliefs.copy() if beliefs else []

            self.needs_update = True

    def start(self):
        """Start the visualizer (blocking)."""
        self.running = True
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("Box Concept 3D Visualizer", self.width, self.height)

        # Register key callbacks
        # Press 'R' to reset view
        self.vis.register_key_callback(ord('R'), self._reset_view_callback)
        # Press 'S' to save camera settings
        self.vis.register_key_callback(ord('S'), self._save_camera_callback)

        # Set render options
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.15])
        opt.point_size = self.point_size
        opt.line_width = 2.0

        # Set mouse mode to rotate around the scene center (like Webots)
        # This makes rotation orbit around the lookat point
        ctr = self.vis.get_view_control()
        ctr.set_constant_z_near(0.01)
        ctr.set_constant_z_far(100.0)

        # Initial geometry
        self.current_geometries = self._rebuild_scene()
        for geom in self.current_geometries:
            self.vis.add_geometry(geom)

        # Set initial viewpoint: top-down view from above
        # First reset to default view that fits all geometry
        self.vis.reset_view_point(True)

        # Try to load saved camera settings, otherwise use default
        saved_settings = self._load_camera_settings()
        if saved_settings:
            self._apply_camera_settings(saved_settings)
            print("Loaded saved camera settings")
        else:
            ctr = self.vis.get_view_control()
            # Default camera: rotate to look from above
            ctr.rotate(0.0, -300.0)
            ctr.scale(15.0)

        # Main loop
        frame_count = 0
        update_interval = 5  # Update geometries every N frames
        save_interval = 60  # Save camera settings every N frames (~1 second)

        while self.running:
            frame_count += 1

            # Update geometries periodically
            if self.needs_update and frame_count % update_interval == 0:
                self.needs_update = False

                # Remove old geometries
                for geom in self.current_geometries:
                    self.vis.remove_geometry(geom, reset_bounding_box=False)

                # Rebuild scene
                self.current_geometries = self._rebuild_scene()

                # Add new geometries without resetting bounding box
                for geom in self.current_geometries:
                    self.vis.add_geometry(geom, reset_bounding_box=False)

            # Save camera settings periodically
            if frame_count % save_interval == 0:
                self._save_camera_settings()

            # Poll events and update renderer
            if not self.vis.poll_events():
                break
            self.vis.update_renderer()

            time.sleep(0.016)  # ~60 FPS limit

        # Save camera settings before closing
        self._save_camera_settings()
        print("Camera settings saved")

        self.vis.destroy_window()

    def start_async(self):
        """Start the visualizer in a separate thread."""
        self.thread = threading.Thread(target=self.start, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the visualizer."""
        self.running = False

    def _reset_view_callback(self, vis):
        """Callback to reset view when 'R' is pressed."""
        self.vis.reset_view_point(True)
        ctr = self.vis.get_view_control()
        ctr.rotate(0.0, -300.0)
        ctr.scale(15.0)
        print("View reset")
        return False

    def _save_camera_callback(self, vis):
        """Callback to save camera settings when 'S' is pressed."""
        self._save_camera_settings()
        print("Camera settings saved manually")
        return False

    def _save_camera_settings(self):
        """Save current camera settings to a JSON file."""
        if self.vis is None:
            return
        try:
            ctr = self.vis.get_view_control()
            # Try to get pinhole camera parameters
            try:
                cam_params = ctr.convert_to_pinhole_camera_parameters()
                # Save intrinsic and extrinsic matrices
                settings = {
                    'extrinsic': cam_params.extrinsic.tolist(),
                    'intrinsic_width': cam_params.intrinsic.width,
                    'intrinsic_height': cam_params.intrinsic.height,
                    'intrinsic_matrix': cam_params.intrinsic.intrinsic_matrix.tolist()
                }
                with open(CAMERA_SETTINGS_PATH, 'w') as f:
                    json.dump(settings, f, indent=2)
            except Exception as e:
                # If pinhole conversion fails (orthographic mode), skip saving
                pass
        except Exception as e:
            print(f"Failed to save camera settings: {e}")

    def _load_camera_settings(self) -> Optional[dict]:
        """Load camera settings from JSON file."""
        if not os.path.exists(CAMERA_SETTINGS_PATH):
            return None
        try:
            with open(CAMERA_SETTINGS_PATH, 'r') as f:
                settings = json.load(f)
            return settings
        except Exception as e:
            print(f"Failed to load camera settings: {e}")
            return None

    def _apply_camera_settings(self, settings: dict):
        """Apply saved camera settings."""
        if self.vis is None or settings is None:
            return
        try:
            ctr = self.vis.get_view_control()
            # Create pinhole camera parameters
            cam_params = ctr.convert_to_pinhole_camera_parameters()

            # Apply saved extrinsic matrix
            cam_params.extrinsic = np.array(settings['extrinsic'])

            # Apply saved intrinsic if dimensions match
            if (cam_params.intrinsic.width == settings['intrinsic_width'] and
                cam_params.intrinsic.height == settings['intrinsic_height']):
                cam_params.intrinsic.intrinsic_matrix = np.array(settings['intrinsic_matrix'])

            ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        except Exception as e:
            # If applying fails, just use default view
            print(f"Failed to apply camera settings: {e}")


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
    print("Controls (Webots-style):")
    print("  - Left mouse + drag: Rotate around center")
    print("  - Right mouse + drag: Pan")
    print("  - Scroll: Zoom in/out")
    print("  - Middle mouse + drag: Tilt")
    print("  - R: Reset view")
    print("  - S: Save camera settings")
    print("  - Q or close window: Exit")

    viz.start()
