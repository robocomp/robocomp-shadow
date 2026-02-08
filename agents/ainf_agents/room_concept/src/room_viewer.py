#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Room Viewer using DearPyGui with Observer Pattern

This module provides a real-time visualization of the room estimation process,
showing both the estimated robot pose and the ground truth pose.

Uses the Observer pattern to decouple the viewer from the SpecificWorker.
"""

import dearpygui.dearpygui as dpg
import numpy as np
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class RobotState:
    """State of a robot for visualization"""
    x: float = 0.0           # Position x (meters)
    y: float = 0.0           # Position y (meters)
    theta: float = 0.0       # Heading angle (radians)


@dataclass
class RoomState:
    """State of the room for visualization"""
    width: float = 6.0       # Room width (meters)
    length: float = 4.0      # Room length (meters)


@dataclass
class ViewerData:
    """Complete data package for the viewer"""
    estimated_pose: RobotState = None
    ground_truth_pose: RobotState = None
    room: RoomState = None
    lidar_points: np.ndarray = None  # [N, 2] points in robot frame
    phase: str = "init"
    sdf_error: float = 0.0
    pose_error: float = 0.0
    angle_error: float = 0.0   # Angle error in degrees (vs GT)
    error_x: float = 0.0       # Position error in x (meters, vs GT)
    error_y: float = 0.0       # Position error in y (meters, vs GT)
    step: int = 0
    # Innovation (prediction error from motion model)
    innovation_x: float = 0.0      # Innovation in x (meters)
    innovation_y: float = 0.0      # Innovation in y (meters)
    innovation_theta: float = 0.0  # Innovation in theta (degrees)
    prior_precision: float = 0.0   # Adaptive prior precision
    # Free Energy components (Active Inference)
    f_likelihood: float = 0.0      # F_likelihood = mean(SDF²) - accuracy term
    f_prior: float = 0.0           # F_prior = (s-s_pred)ᵀΣ⁻¹(s-s_pred) - complexity term
    vfe: float = 0.0               # Variational Free Energy after optimization
    # Ground truth room size
    gt_room_width: float = 6.0     # GT room width (meters)
    gt_room_length: float = 4.0    # GT room length (meters)
    # Pose covariance for uncertainty ellipse (2x2 for x,y)
    pose_covariance: np.ndarray = None  # 2x2 covariance matrix [x,y]
    pose_covariance_full: np.ndarray = None  # 3x3 covariance matrix [x,y,theta]
    # CPU/Performance stats
    lidar_subsample_factor: int = 1    # Current LIDAR subsampling factor
    optimizer_iterations: int = 0       # Iterations used in optimizer
    compute_time_ms: float = 0.0        # Total compute time in ms
    cpu_percent: float = 0.0            # CPU usage percentage
    # Velocity-adaptive weights
    velocity_weights: np.ndarray = None  # [w_x, w_y, w_theta] velocity-based optimization weights
    # Uncertainty-based speed modulation
    speed_factor: float = 1.0          # Speed modulation factor based on uncertainty
    # Velocity calibration
    velocity_scale: float = 1.0        # Learned velocity calibration factor (k)
    # Commanded velocities
    cmd_adv_x: float = 0.0             # Commanded forward velocity (mm/s)
    cmd_adv_y: float = 0.0             # Commanded lateral velocity (mm/s)
    cmd_rot: float = 0.0               # Commanded rotation velocity (rad/s)

    def __post_init__(self):
        if self.estimated_pose is None:
            self.estimated_pose = RobotState()
        if self.ground_truth_pose is None:
            self.ground_truth_pose = RobotState()
        if self.room is None:
            self.room = RoomState()
        if self.pose_covariance is None:
            self.pose_covariance = np.eye(2) * 0.01  # Default small covariance
        if self.pose_covariance_full is None:
            self.pose_covariance_full = np.eye(3) * 0.01  # Default small covariance 3x3
        if self.velocity_weights is None:
            self.velocity_weights = np.array([1.0, 1.0, 1.0])  # Default uniform weights


class RoomObserver(ABC):
    """Abstract observer interface for room estimation updates"""

    @abstractmethod
    def on_update(self, data: ViewerData):
        """Called when room/pose data is updated"""
        pass


class RoomViewerDPG(RoomObserver):
    """
    DearPyGui-based room viewer implementing the Observer pattern.

    Coordinate System:
    - World frame: origin at room center
      * X-axis: points right (room width direction)
      * Y-axis: points up (room length direction)
      * θ=0: robot facing +Y direction (up)
    - Room: rectangular boundaries at ±width/2, ±length/2
    - Screen: standard pixel coordinates (Y increases downward)

    Displays:
    - Room boundaries aligned with window axes
    - Estimated robot pose (blue)
    - Ground truth robot pose (green)
    - LIDAR points (optional, transformed to world frame)
    - Statistics panel with errors and metrics
    """

    def __init__(self,
                 window_width: int = 900,
                 window_height: int = 900,
                 margin: float = 1.0,
                 show_lidar: bool = True,
                 dsr_viewer=None):
        """
        Initialize the room viewer.

        Args:
            window_width: Window width in pixels
            window_height: Window height in pixels
            margin: Extra margin around room in meters
            show_lidar: Whether to display LIDAR points
            dsr_viewer: Optional DSRGraphViewerDPG instance to integrate side-by-side
        """
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        self.show_lidar = show_lidar
        self.dsr_viewer = dsr_viewer

        # Plan running state (starts stopped)
        self.plan_running = False
        self._on_plan_toggle = None  # External callback
        self._on_plan_selected = None  # Callback for plan selection
        self.plan_names = ["Basic (Static)", "Circle (0.75m + 360°)", "Ocho (Figure-8)", "Circle Center (r=1.5m)"]  # Available plans


        # Current data
        self.data: ViewerData = ViewerData()
        self.data_lock = threading.Lock()

        # Drawing area dimensions - NEW LAYOUT:
        # Top row: Room canvas + DSR panel (side by side)
        # Bottom row: Stats panel (horizontal, full width)
        dsr_width = self.dsr_viewer.canvas_width if dsr_viewer else 0
        stats_height = 320  # Height of stats panel (two rows)
        self.draw_width = window_width - dsr_width - 20  # Room canvas width
        self.dsr_width = dsr_width  # DSR panel width
        self.stats_height = stats_height
        self.draw_height = window_height - stats_height - 50  # Room canvas height

        # DPG context
        self.is_running = False
        self.dpg_thread: Optional[threading.Thread] = None

        # Robot drawing parameters
        self.robot_radius = 0.25  # Robot radius in meters
        self.robot_arrow_length = 0.4  # Arrow length in meters

        # Error history for plotting
        self.pose_error_history: List[float] = []
        self.sdf_error_history: List[float] = []
        self.max_history = 500

    def set_plan_toggle_callback(self, callback):
        """Set callback invoked when the Start/Stop button is pressed.
        callback receives a single bool argument: True if plan is now running."""
        self._on_plan_toggle = callback

    def set_plan_selected_callback(self, callback):
        """Set callback for plan selection from dropdown.
        Callback receives plan index (0-based)"""
        self._on_plan_selected = callback

    def start(self):
        """Start the viewer in a separate thread"""
        if self.is_running:
            return

        self.is_running = True
        self.dpg_thread = threading.Thread(target=self._run_dpg, daemon=True)
        self.dpg_thread.start()

    def stop(self):
        """Stop the viewer"""
        self.is_running = False
        if dpg.is_dearpygui_running():
            dpg.stop_dearpygui()

    def on_update(self, data: ViewerData):
        """Observer callback - called when data is updated"""
        with self.data_lock:
            self.data = data
            # Update error history
            if data.pose_error > 0:
                self.pose_error_history.append(data.pose_error)
                if len(self.pose_error_history) > self.max_history:
                    self.pose_error_history.pop(0)
            if data.sdf_error > 0:
                self.sdf_error_history.append(data.sdf_error)
                if len(self.sdf_error_history) > self.max_history:
                    self.sdf_error_history.pop(0)

    def _run_dpg(self):
        """Main DearPyGui loop (runs in separate thread)"""
        dpg.create_context()
        dpg.create_viewport(title="Room Pose Estimator Viewer",
                           width=self.window_width,
                           height=self.window_height)

        # Create main window
        with dpg.window(label="Room Viewer", tag="main_window",
                       width=self.window_width, height=self.window_height,
                       no_title_bar=True, no_resize=True, no_move=True):

            # TOP ROW: Room canvas + DSR panel (side by side)
            with dpg.group(horizontal=True):
                # Drawing canvas (room)
                with dpg.drawlist(width=self.draw_width, height=self.draw_height,
                                 tag="canvas"):
                    pass

                # DSR Graph panel (if dsr_viewer provided)
                if self.dsr_viewer:
                    with dpg.child_window(width=self.dsr_width - 10, height=self.draw_height, tag="dsr_panel"):
                        dpg.add_text("DSR GRAPH", color=(100, 255, 100))
                        dpg.add_separator()
                        # Create drawlist for DSR graph
                        with dpg.drawlist(width=self.dsr_width - 20, height=self.draw_height - 60,
                                        tag="dsr_canvas"):
                            pass

            # BOTTOM: Stats panel (two rows)
            with dpg.child_window(width=-1, height=self.stats_height):
                # ROW 1: Status, Motion, Estimated, GT, Errors
                with dpg.group(horizontal=True):
                    # Column 1: Phase, Step, Room
                    with dpg.group():
                        dpg.add_text("STATUS", color=(255, 255, 0))
                        with dpg.group(horizontal=True):
                            dpg.add_text("Phase:", color=(200, 200, 200))
                            dpg.add_text("INIT", tag="phase_text", color=(0, 255, 255))
                        with dpg.group(horizontal=True):
                            dpg.add_text("Step:", color=(200, 200, 200))
                            dpg.add_text("0", tag="step_text")
                        dpg.add_spacer(height=3)
                        dpg.add_text("ROOM", color=(255, 255, 0))
                        dpg.add_text("Est: 0.0 x 0.0 m", tag="room_size_text")
                        dpg.add_text("GT:  0.0 x 0.0 m", tag="room_gt_size_text", color=(100, 255, 100))

                    dpg.add_spacer(width=20)

                    # Column 2: Motion Model
                    with dpg.group():
                        dpg.add_text("MOTION MODEL", color=(255, 200, 100))
                        dpg.add_text("dx: 0.0 cm", tag="innov_x_text")
                        dpg.add_text("dy: 0.0 cm", tag="innov_y_text")
                        dpg.add_text("dθ: 0.0°", tag="innov_theta_text")
                        dpg.add_text("π: 0.000", tag="prior_precision_text")

                    dpg.add_spacer(width=20)

                    # Column 3: Estimated Pose
                    with dpg.group():
                        dpg.add_text("ESTIMATED", color=(100, 150, 255))
                        dpg.add_text("X: 0.000 m", tag="est_x_text")
                        dpg.add_text("Y: 0.000 m", tag="est_y_text")
                        dpg.add_text("θ: 0.0°", tag="est_theta_text")

                    dpg.add_spacer(width=20)

                    # Column 4: Ground Truth
                    with dpg.group():
                        dpg.add_text("GROUND TRUTH", color=(100, 255, 100))
                        dpg.add_text("X: 0.000 m", tag="gt_x_text")
                        dpg.add_text("Y: 0.000 m", tag="gt_y_text")
                        dpg.add_text("θ: 0.0°", tag="gt_theta_text")

                    dpg.add_spacer(width=20)

                    # Column 5: Errors
                    with dpg.group():
                        dpg.add_text("ERRORS (vs GT)", color=(255, 255, 0))
                        dpg.add_text("X err: 0.0 cm", tag="err_x_text")
                        dpg.add_text("Y err: 0.0 cm", tag="err_y_text")
                        dpg.add_text("θ err: 0.0°", tag="err_theta_text")
                        with dpg.group(horizontal=True):
                            dpg.add_text("Pose:", color=(200, 200, 200))
                            dpg.add_text("0.0 cm", tag="pose_error_text")
                        with dpg.group(horizontal=True):
                            dpg.add_text("SDF:", color=(200, 200, 200))
                            dpg.add_text("0.000 m", tag="sdf_error_text")

                    dpg.add_spacer(width=20)

                    # Column 6: Pose Covariance
                    with dpg.group():
                        dpg.add_text("POSE COV (σ)", color=(255, 180, 100))
                        dpg.add_text("σx: 0.000 m", tag="pose_cov_x_text")
                        dpg.add_text("σy: 0.000 m", tag="pose_cov_y_text")
                        dpg.add_text("σθ: 0.0°", tag="pose_cov_theta_text")
                        with dpg.group(horizontal=True):
                            dpg.add_text("σpos:", color=(200, 200, 200))
                            dpg.add_text("0.000 m", tag="pose_cov_total_text")

                dpg.add_separator()

                # ROW 2: Free Energy, Legend, CPU Stats, Vel Weights
                with dpg.group(horizontal=True):
                    # Column 1: Free Energy Components
                    with dpg.group():
                        dpg.add_text("FREE ENERGY", color=(255, 100, 255))
                        with dpg.group(horizontal=True):
                            dpg.add_text("F_like:", color=(200, 200, 200))
                            dpg.add_text("0.000", tag="f_likelihood_text", color=(255, 200, 100))
                        with dpg.group(horizontal=True):
                            dpg.add_text("F_prior:", color=(200, 200, 200))
                            dpg.add_text("0.000", tag="f_prior_text", color=(100, 200, 255))
                        with dpg.group(horizontal=True):
                            dpg.add_text("VFE:", color=(200, 200, 200))
                            dpg.add_text("0.000", tag="vfe_text", color=(255, 255, 100))

                    dpg.add_spacer(width=20)

                    # Column 2: Legend
                    with dpg.group():
                        dpg.add_text("LEGEND", color=(255, 255, 0))
                        dpg.add_text("● Estimated", color=(100, 150, 255))
                        dpg.add_text("● Ground Truth", color=(100, 255, 100))
                        dpg.add_text("· LIDAR points", color=(255, 200, 100))

                    dpg.add_spacer(width=20)

                    # Column 3: Plan Control
                    with dpg.group():
                        dpg.add_text("PLAN", color=(255, 255, 0))
                        dpg.add_combo(self.plan_names, tag="plan_combo",
                                      default_value=self.plan_names[0],
                                      callback=self._on_plan_combo_changed,
                                      width=150)
                        dpg.add_spacer(height=2)
                        dpg.add_button(label="Start Plan", tag="plan_button",
                                       callback=self._on_plan_button_clicked,
                                       width=100)

                    dpg.add_spacer(width=20)

                    # Column 4: CPU/Performance Stats
                    with dpg.group():
                        dpg.add_text("CPU STATS", color=(255, 150, 50))
                        with dpg.group(horizontal=True):
                            dpg.add_text("Subsample:", color=(200, 200, 200))
                            dpg.add_text("1x", tag="subsample_text", color=(150, 255, 150))
                        with dpg.group(horizontal=True):
                            dpg.add_text("LIDAR pts:", color=(200, 200, 200))
                            dpg.add_text("0", tag="lidar_pts_text")
                        with dpg.group(horizontal=True):
                            dpg.add_text("Optimizer:", color=(200, 200, 200))
                            dpg.add_text("0 iters", tag="optimizer_iters_text")

                    dpg.add_spacer(width=20)

                    # Column 5: More CPU Stats
                    with dpg.group():
                        dpg.add_text("PERF", color=(255, 150, 50))
                        with dpg.group(horizontal=True):
                            dpg.add_text("Compute:", color=(200, 200, 200))
                            dpg.add_text("0.0 ms", tag="compute_time_text")
                        with dpg.group(horizontal=True):
                            dpg.add_text("CPU:", color=(200, 200, 200))
                            dpg.add_text("0.0%", tag="cpu_percent_text")
                        with dpg.group(horizontal=True):
                            dpg.add_text("Speed:", color=(200, 200, 200))
                            dpg.add_text("100%", tag="speed_factor_text", color=(100, 255, 100))
                        with dpg.group(horizontal=True):
                            dpg.add_text("Vel Cal:", color=(200, 200, 200))
                            dpg.add_text("1.000", tag="velocity_scale_text", color=(100, 200, 255))

                    dpg.add_spacer(width=20)

                    # Column 6: Velocity-Adaptive Weights
                    with dpg.group():
                        dpg.add_text("VEL WEIGHTS", color=(180, 100, 255))
                        with dpg.group(horizontal=True):
                            dpg.add_text("w_x:", color=(200, 200, 200))
                            dpg.add_text("1.00", tag="vel_weight_x_text", color=(150, 200, 255))
                        with dpg.group(horizontal=True):
                            dpg.add_text("w_y:", color=(200, 200, 200))
                            dpg.add_text("1.00", tag="vel_weight_y_text", color=(150, 200, 255))
                        with dpg.group(horizontal=True):
                            dpg.add_text("w_θ:", color=(200, 200, 200))
                            dpg.add_text("1.00", tag="vel_weight_theta_text", color=(150, 200, 255))

                    dpg.add_spacer(width=20)

                    # Column 7: Commanded Velocities
                    with dpg.group():
                        dpg.add_text("CMD VEL", color=(100, 255, 200))
                        with dpg.group(horizontal=True):
                            dpg.add_text("adv_x:", color=(200, 200, 200))
                            dpg.add_text("0", tag="cmd_adv_x_text", color=(150, 255, 200))
                        with dpg.group(horizontal=True):
                            dpg.add_text("adv_y:", color=(200, 200, 200))
                            dpg.add_text("0", tag="cmd_adv_y_text", color=(150, 255, 200))
                        with dpg.group(horizontal=True):
                            dpg.add_text("rot:", color=(200, 200, 200))
                            dpg.add_text("0.00", tag="cmd_rot_text", color=(150, 255, 200))

        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Main render loop
        while dpg.is_dearpygui_running() and self.is_running:
            self._render_frame()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    def _on_plan_button_clicked(self, sender, app_data):
        """Handle Start/Stop plan button click"""
        self.plan_running = not self.plan_running
        label = "Stop Plan" if self.plan_running else "Start Plan"
        dpg.set_item_label("plan_button", label)
        if self._on_plan_toggle:
            self._on_plan_toggle(self.plan_running)

    def _on_plan_combo_changed(self, sender, app_data):
        """Handle plan selection from combo"""
        try:
            plan_index = self.plan_names.index(app_data)
            if self._on_plan_selected:
                self._on_plan_selected(plan_index)
            print(f"[Viewer] Plan selected: {app_data} (index {plan_index})")
        except ValueError:
            pass

    def _world_to_screen(self, x: float, y: float,
                         room_width: float, room_height: float) -> Tuple[float, float]:
        """
        Convert world coordinates (meters) to screen coordinates (pixels).

        Room is centered in the drawing area with the coordinate system:
        - X axis pointing right (room width direction)
        - Y axis pointing up (room length direction)

        Args:
            x: World x coordinate (meters)
            y: World y coordinate (meters)
            room_width: Room width in meters
            room_height: Room height/length in meters

        Returns:
            (screen_x, screen_y) in pixels
        """
        # Calculate scale to fit room with margin
        total_width = room_width + 2 * self.margin
        total_height = room_height + 2 * self.margin

        scale_x = self.draw_width / total_width
        scale_y = self.draw_height / total_height
        scale = min(scale_x, scale_y)

        # Center of drawing area
        center_x = self.draw_width / 2
        center_y = self.draw_height / 2

        # Transform: world origin (0,0) maps to screen center
        # Y is flipped because screen Y increases downward
        screen_x = center_x + x * scale
        screen_y = center_y - y * scale  # Flip Y axis

        return screen_x, screen_y

    def _render_frame(self):
        """Render one frame of the visualization"""
        with self.data_lock:
            data = self.data

        # Clear canvas
        dpg.delete_item("canvas", children_only=True)

        room_w = data.room.width
        room_l = data.room.length

        # Draw room
        self._draw_room(room_w, room_l)

        # Draw coordinate axes
        self._draw_axes(room_w, room_l)

        # Draw LIDAR points (transformed to world frame using estimated pose)
        if self.show_lidar and data.lidar_points is not None and len(data.lidar_points) > 0:
            self._draw_lidar_points(data.lidar_points, data.estimated_pose, room_w, room_l)

        # Draw ground truth robot (green) - circle
        self._draw_robot(data.ground_truth_pose, room_w, room_l,
                        color=(100, 255, 100, 255), label="GT")

        # Draw estimated robot (blue) - ellipse from covariance
        self._draw_robot(data.estimated_pose, room_w, room_l,
                        color=(100, 150, 255, 255), label="Est",
                        covariance=data.pose_covariance)

        # Update stats panel
        self._update_stats(data)

        # Update DSR graph viewer if present
        if self.dsr_viewer:
            self.dsr_viewer.update()

    def _draw_room(self, width: float, height: float):
        """Draw room boundaries"""
        # Room corners (centered at origin)
        corners = [
            (-width/2, -height/2),  # Bottom-left
            (width/2, -height/2),   # Bottom-right
            (width/2, height/2),    # Top-right
            (-width/2, height/2),   # Top-left
        ]

        # Convert to screen coordinates
        screen_corners = [self._world_to_screen(x, y, width, height) for x, y in corners]

        # Draw room walls
        wall_color = (200, 200, 200, 255)
        wall_thickness = 3

        for i in range(4):
            x1, y1 = screen_corners[i]
            x2, y2 = screen_corners[(i + 1) % 4]
            dpg.draw_line((x1, y1), (x2, y2), color=wall_color,
                         thickness=wall_thickness, parent="canvas")

        # Draw room fill (very transparent)
        dpg.draw_quad(screen_corners[0], screen_corners[1],
                     screen_corners[2], screen_corners[3],
                     color=(50, 50, 80, 50), fill=(50, 50, 80, 30),
                     parent="canvas")

    def _draw_axes(self, room_width: float, room_height: float):
        """Draw coordinate axes at room center"""
        origin = self._world_to_screen(0, 0, room_width, room_height)

        # X axis (red, pointing right)
        x_end = self._world_to_screen(0.5, 0, room_width, room_height)
        dpg.draw_line(origin, x_end, color=(255, 100, 100, 200),
                     thickness=2, parent="canvas")
        dpg.draw_text((x_end[0] + 5, x_end[1] - 8), "X",
                     color=(255, 100, 100, 200), parent="canvas")

        # Y axis (green, pointing up)
        y_end = self._world_to_screen(0, 0.5, room_width, room_height)
        dpg.draw_line(origin, y_end, color=(100, 255, 100, 200),
                     thickness=2, parent="canvas")
        dpg.draw_text((y_end[0] + 5, y_end[1]), "Y",
                     color=(100, 255, 100, 200), parent="canvas")

    def _draw_robot(self, pose: RobotState, room_width: float, room_height: float,
                   color: Tuple[int, int, int, int], label: str = "",
                   covariance: np.ndarray = None):
        """
        Draw a robot as an ellipse (from covariance) or circle with a direction arrow.

        Args:
            pose: Robot pose (x, y, theta)
            room_width: Room width for coordinate transform
            room_height: Room height for coordinate transform
            color: RGBA color tuple
            label: Optional label to draw near robot
            covariance: Optional 2x2 covariance matrix for uncertainty ellipse
        """
        # Robot center in screen coords
        cx, cy = self._world_to_screen(pose.x, pose.y, room_width, room_height)

        # Calculate scale for robot size
        total_width = room_width + 2 * self.margin
        scale = min(self.draw_width / total_width,
                   self.draw_height / (room_height + 2 * self.margin))

        arrow_length_px = self.robot_arrow_length * scale

        # Base radius for GT robot (small fixed circle)
        gt_radius_px = 8  # Small fixed radius for GT

        # Draw robot body: ellipse from covariance or default circle
        if covariance is not None and covariance.shape == (2, 2):
            # Compute ellipse parameters from covariance matrix
            # Use diagonal elements for axis lengths (variance in x and y)
            # Align ellipse with robot orientation (theta)
            try:
                # Get variances from diagonal (ignore off-diagonal correlations for simplicity)
                # Or use eigenvalues for proper ellipse axes
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)

                # Sort by eigenvalue (largest first)
                order = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[order]

                # Semi-axes lengths (2-sigma = 95% confidence)
                sigma_scale = 2.0  # 2-sigma ellipse
                axis1 = np.sqrt(max(eigenvalues[0], 1e-6)) * sigma_scale * scale
                axis2 = np.sqrt(max(eigenvalues[1], 1e-6)) * sigma_scale * scale

                # Minimum radius matches GT robot size, max is larger
                axis1 = np.clip(axis1, gt_radius_px, 150)
                axis2 = np.clip(axis2, gt_radius_px, 150)

                # Use robot orientation (theta) for ellipse alignment
                # This aligns the ellipse axes with the robot's forward/lateral directions
                angle = pose.theta

                # Draw ellipse as polygon (DearPyGui doesn't have native ellipse)
                num_points = 32
                ellipse_points = []
                for i in range(num_points):
                    t = 2 * np.pi * i / num_points
                    # Point on unit ellipse (axis1 = forward, axis2 = lateral)
                    px = axis2 * np.cos(t)  # lateral axis
                    py = axis1 * np.sin(t)  # forward axis (larger if more uncertainty forward)
                    # Rotate by robot angle (theta)
                    # Note: screen Y is inverted, so we use sin/cos appropriately
                    rx = px * np.cos(angle) - py * np.sin(angle)
                    ry = px * np.sin(angle) + py * np.cos(angle)
                    ellipse_points.append((cx + rx, cy + ry))

                # Close the ellipse
                ellipse_points.append(ellipse_points[0])

                # Draw ellipse outline
                dpg.draw_polygon(ellipse_points, color=color, thickness=2, parent="canvas")

                # Draw filled ellipse with transparency
                fill_color = (color[0], color[1], color[2], 50)
                dpg.draw_polygon(ellipse_points, color=fill_color, fill=fill_color, parent="canvas")

            except Exception as e:
                # Fallback to small circle
                dpg.draw_circle((cx, cy), gt_radius_px, color=color,
                               thickness=2, parent="canvas")
        else:
            # GT robot: draw small fixed circle
            dpg.draw_circle((cx, cy), gt_radius_px, color=color,
                           thickness=2, parent="canvas")

        # Draw direction arrow
        # Arrow points in direction of theta
        arrow_dx = np.sin(pose.theta) * arrow_length_px
        arrow_dy = np.cos(pose.theta) * arrow_length_px

        arrow_end = (cx + arrow_dx, cy + arrow_dy)
        dpg.draw_arrow((cx, cy), arrow_end, color=color,
                      thickness=2, size=8, parent="canvas")

        # Draw label
        if label:
            dpg.draw_text((cx + 20, cy - 8), label,
                         color=color, size=12, parent="canvas")

    def _draw_lidar_points(self, points: np.ndarray, robot_pose: RobotState,
                          room_width: float, room_height: float):
        """
        Draw LIDAR points transformed to world frame.

        Args:
            points: [N, 2] LIDAR points in robot frame
            robot_pose: Current robot pose for transformation
            room_width: Room width for coordinate transform
            room_height: Room height for coordinate transform
        """
        if len(points) == 0:
            return

        # Transform points from robot frame to world frame
        cos_t = np.cos(robot_pose.theta)
        sin_t = np.sin(robot_pose.theta)

        # Rotation matrix
        R = np.array([[cos_t, -sin_t],
                     [sin_t, cos_t]])

        # Transform: p_world = R @ p_robot + [x, y]
        world_points = (R @ points.T).T + np.array([robot_pose.x, robot_pose.y])

        # Draw points
        point_color = (255, 200, 100, 180)

        # Subsample if too many points
        step = max(1, len(world_points) // 200)

        for i in range(0, len(world_points), step):
            px, py = world_points[i]
            sx, sy = self._world_to_screen(px, py, room_width, room_height)
            dpg.draw_circle((sx, sy), 2, color=point_color,
                           fill=point_color, parent="canvas")

    def _update_stats(self, data: ViewerData):
        """Update the statistics panel"""
        # Phase
        phase_color = (0, 255, 255) if data.phase == "init" else (0, 255, 0)
        dpg.set_value("phase_text", data.phase.upper())
        dpg.configure_item("phase_text", color=phase_color)

        # Step
        dpg.set_value("step_text", str(data.step))

        # Room size
        dpg.set_value("room_size_text",
                     f"Est: {data.room.width:.2f} x {data.room.length:.2f} m")
        dpg.set_value("room_gt_size_text",
                     f"GT:  {data.gt_room_width:.2f} x {data.gt_room_length:.2f} m")

        # Motion model (innovation and prior weight)
        dpg.set_value("innov_x_text", f"dx: {data.innovation_x*100:.1f} cm")
        dpg.set_value("innov_y_text", f"dy: {data.innovation_y*100:.1f} cm")
        dpg.set_value("innov_theta_text", f"dθ: {data.innovation_theta:.1f}°")
        pp_color = (100, 255, 100) if data.prior_precision > 0.5 else (255, 255, 100) if data.prior_precision > 0.1 else (255, 150, 100)
        dpg.set_value("prior_precision_text", f"π: {data.prior_precision:.3f}")
        dpg.configure_item("prior_precision_text", color=pp_color)

        # Estimated pose
        dpg.set_value("est_x_text", f"X: {data.estimated_pose.x:.3f} m")
        dpg.set_value("est_y_text", f"Y: {data.estimated_pose.y:.3f} m")
        dpg.set_value("est_theta_text",
                     f"θ: {np.degrees(data.estimated_pose.theta):.1f}°")

        # Ground truth pose
        dpg.set_value("gt_x_text", f"X: {data.ground_truth_pose.x:.3f} m")
        dpg.set_value("gt_y_text", f"Y: {data.ground_truth_pose.y:.3f} m")
        dpg.set_value("gt_theta_text",
                     f"θ: {np.degrees(data.ground_truth_pose.theta):.1f}°")

        # Errors (vs GT)
        err_x_cm = data.error_x * 100
        err_y_cm = data.error_y * 100
        err_x_color = (100, 255, 100) if abs(err_x_cm) < 5 else (255, 255, 100) if abs(err_x_cm) < 10 else (255, 100, 100)
        err_y_color = (100, 255, 100) if abs(err_y_cm) < 5 else (255, 255, 100) if abs(err_y_cm) < 10 else (255, 100, 100)
        err_theta_color = (100, 255, 100) if abs(data.angle_error) < 5 else (255, 255, 100) if abs(data.angle_error) < 15 else (255, 100, 100)

        dpg.set_value("err_x_text", f"X err: {err_x_cm:.1f} cm")
        dpg.configure_item("err_x_text", color=err_x_color)
        dpg.set_value("err_y_text", f"Y err: {err_y_cm:.1f} cm")
        dpg.configure_item("err_y_text", color=err_y_color)
        dpg.set_value("err_theta_text", f"θ err: {data.angle_error:.1f}°")
        dpg.configure_item("err_theta_text", color=err_theta_color)

        pose_err_color = (100, 255, 100) if data.pose_error < 0.05 else (255, 255, 100) if data.pose_error < 0.1 else (255, 100, 100)
        dpg.set_value("pose_error_text", f"{data.pose_error*100:.1f} cm")
        dpg.configure_item("pose_error_text", color=pose_err_color)

        sdf_err_color = (100, 255, 100) if data.sdf_error < 0.05 else (255, 255, 100) if data.sdf_error < 0.1 else (255, 100, 100)
        dpg.set_value("sdf_error_text", f"{data.sdf_error:.3f} m")
        dpg.configure_item("sdf_error_text", color=sdf_err_color)

        # Pose Covariance (standard deviations)
        try:
            if data.pose_covariance is not None:
                sigma_x = np.sqrt(max(0, data.pose_covariance[0, 0]))
                sigma_y = np.sqrt(max(0, data.pose_covariance[1, 1]))
                sigma_pos = np.sqrt(sigma_x**2 + sigma_y**2)

                # Color: green if low (<5cm), yellow if medium (<15cm), orange if high
                cov_x_color = (100, 255, 100) if sigma_x < 0.05 else (255, 255, 100) if sigma_x < 0.15 else (255, 150, 100)
                cov_y_color = (100, 255, 100) if sigma_y < 0.05 else (255, 255, 100) if sigma_y < 0.15 else (255, 150, 100)
                cov_pos_color = (100, 255, 100) if sigma_pos < 0.07 else (255, 255, 100) if sigma_pos < 0.2 else (255, 150, 100)

                dpg.set_value("pose_cov_x_text", f"σx: {sigma_x:.3f} m")
                dpg.configure_item("pose_cov_x_text", color=cov_x_color)
                dpg.set_value("pose_cov_y_text", f"σy: {sigma_y:.3f} m")
                dpg.configure_item("pose_cov_y_text", color=cov_y_color)
                dpg.set_value("pose_cov_total_text", f"{sigma_pos:.3f} m")
                dpg.configure_item("pose_cov_total_text", color=cov_pos_color)

            if data.pose_covariance_full is not None:
                sigma_theta = np.sqrt(max(0, data.pose_covariance_full[2, 2]))
                sigma_theta_deg = np.degrees(sigma_theta)
                cov_theta_color = (100, 255, 100) if sigma_theta_deg < 2 else (255, 255, 100) if sigma_theta_deg < 5 else (255, 150, 100)
                dpg.set_value("pose_cov_theta_text", f"σθ: {sigma_theta_deg:.1f}°")
                dpg.configure_item("pose_cov_theta_text", color=cov_theta_color)
        except Exception:
            pass  # Ignore errors in covariance display

        # CPU Stats
        # Subsample factor with color coding (higher = more efficient)
        subsample_color = (100, 255, 100) if data.lidar_subsample_factor >= 4 else (255, 255, 100) if data.lidar_subsample_factor >= 2 else (200, 200, 200)
        dpg.set_value("subsample_text", f"{data.lidar_subsample_factor}x")
        dpg.configure_item("subsample_text", color=subsample_color)

        # LIDAR points count
        lidar_pts = len(data.lidar_points) if data.lidar_points is not None else 0
        dpg.set_value("lidar_pts_text", str(lidar_pts))

        # Optimizer iterations
        dpg.set_value("optimizer_iters_text", f"{data.optimizer_iterations} iters")

        # Compute time with color coding
        compute_color = (100, 255, 100) if data.compute_time_ms < 20 else (255, 255, 100) if data.compute_time_ms < 50 else (255, 100, 100)
        dpg.set_value("compute_time_text", f"{data.compute_time_ms:.1f} ms")
        dpg.configure_item("compute_time_text", color=compute_color)

        # CPU percentage with color coding (green < 50%, yellow < 100%, red >= 100%)
        cpu_color = (100, 255, 100) if data.cpu_percent < 50 else (255, 255, 100) if data.cpu_percent < 100 else (255, 100, 100)
        dpg.set_value("cpu_percent_text", f"{data.cpu_percent:.1f}%")
        dpg.configure_item("cpu_percent_text", color=cpu_color)

        # Speed factor (uncertainty-based modulation)
        speed_pct = data.speed_factor * 100
        speed_color = (100, 255, 100) if data.speed_factor > 0.8 else (255, 255, 100) if data.speed_factor > 0.5 else (255, 150, 100)
        dpg.set_value("speed_factor_text", f"{speed_pct:.0f}%")
        dpg.configure_item("speed_factor_text", color=speed_color)

        # Velocity calibration factor
        # Green if close to 1.0 (0.95-1.05), yellow if moderate (0.8-1.2), orange if significant
        vel_scale = data.velocity_scale
        if 0.95 <= vel_scale <= 1.05:
            vel_scale_color = (100, 255, 100)  # Good calibration
        elif 0.8 <= vel_scale <= 1.2:
            vel_scale_color = (255, 255, 100)  # Moderate correction
        else:
            vel_scale_color = (255, 150, 100)  # Significant correction needed
        dpg.set_value("velocity_scale_text", f"{vel_scale:.3f}")
        dpg.configure_item("velocity_scale_text", color=vel_scale_color)

        # Velocity-adaptive weights
        if data.velocity_weights is not None and len(data.velocity_weights) >= 3:
            w_x, w_y, w_theta = data.velocity_weights[:3]

            # Color coding: green if boosted (>1.2), yellow if normal (0.8-1.2), orange if reduced (<0.8)
            def weight_color(w):
                if w > 1.2:
                    return (100, 255, 100)  # Boosted - green
                elif w < 0.8:
                    return (255, 150, 100)  # Reduced - orange
                else:
                    return (150, 200, 255)  # Normal - light blue

            dpg.set_value("vel_weight_x_text", f"{w_x:.2f}")
            dpg.configure_item("vel_weight_x_text", color=weight_color(w_x))
            dpg.set_value("vel_weight_y_text", f"{w_y:.2f}")
            dpg.configure_item("vel_weight_y_text", color=weight_color(w_y))
            dpg.set_value("vel_weight_theta_text", f"{w_theta:.2f}")
            dpg.configure_item("vel_weight_theta_text", color=weight_color(w_theta))

        # Free Energy components
        # F_likelihood (accuracy term) - lower is better
        f_like_color = (100, 255, 100) if data.f_likelihood < 0.01 else (255, 255, 100) if data.f_likelihood < 0.05 else (255, 150, 100)
        dpg.set_value("f_likelihood_text", f"{data.f_likelihood:.4f}")
        dpg.configure_item("f_likelihood_text", color=f_like_color)

        # F_prior (complexity term) - lower means motion model is accurate
        f_prior_color = (100, 255, 100) if data.f_prior < 0.1 else (255, 255, 100) if data.f_prior < 0.5 else (255, 150, 100)
        dpg.set_value("f_prior_text", f"{data.f_prior:.4f}")
        dpg.configure_item("f_prior_text", color=f_prior_color)

        # VFE (total Free Energy after optimization) - lower is better
        vfe_color = (100, 255, 100) if data.vfe < 0.05 else (255, 255, 100) if data.vfe < 0.1 else (255, 150, 100)
        dpg.set_value("vfe_text", f"{data.vfe:.4f}")
        dpg.configure_item("vfe_text", color=vfe_color)

        # Commanded velocities
        # adv_x: green if moving, gray if zero
        adv_x_color = (100, 255, 200) if abs(data.cmd_adv_x) > 1 else (150, 150, 150)
        dpg.set_value("cmd_adv_x_text", f"{data.cmd_adv_x:.0f}")
        dpg.configure_item("cmd_adv_x_text", color=adv_x_color)

        adv_y_color = (100, 255, 200) if abs(data.cmd_adv_y) > 1 else (150, 150, 150)
        dpg.set_value("cmd_adv_y_text", f"{data.cmd_adv_y:.0f}")
        dpg.configure_item("cmd_adv_y_text", color=adv_y_color)

        # rot: positive=left (green), negative=right (cyan), zero=gray
        if abs(data.cmd_rot) < 0.01:
            rot_color = (150, 150, 150)
        elif data.cmd_rot > 0:
            rot_color = (100, 255, 100)  # Turning left - green
        else:
            rot_color = (100, 200, 255)  # Turning right - cyan
        dpg.set_value("cmd_rot_text", f"{data.cmd_rot:.2f}")
        dpg.configure_item("cmd_rot_text", color=rot_color)


class RoomSubject:
    """
    Subject class for the Observer pattern.

    Manages a list of observers and notifies them when data changes.
    This class should be used by SpecificWorker to notify the viewer.
    """

    def __init__(self):
        self._observers: List[RoomObserver] = []

    def attach(self, observer: RoomObserver):
        """Attach an observer"""
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: RoomObserver):
        """Detach an observer"""
        if observer in self._observers:
            self._observers.remove(observer)

    def notify(self, data: ViewerData):
        """Notify all observers with new data"""
        for observer in self._observers:
            try:
                observer.on_update(data)
            except Exception as e:
                print(f"[RoomSubject] Error notifying observer: {e}")


# Convenience function to create viewer data from estimator state
def create_viewer_data(room_estimator,
                       robot_pose_gt: List[float],
                       lidar_points: np.ndarray = None,
                       step: int = 0,
                       innovation: np.ndarray = None,
                       prior_precision: float = 0.0,
                       lidar_subsample_factor: int = 1,
                       optimizer_iterations: int = 0,
                       compute_time_ms: float = 0.0,
                       cpu_percent: float = 0.0,
                       velocity_weights: np.ndarray = None,
                       f_likelihood: float = 0.0,
                       f_prior: float = 0.0,
                       vfe: float = 0.0,
                       speed_factor: float = 1.0,
                       velocity_scale: float = 1.0,
                       cmd_vel: tuple = (0.0, 0.0, 0.0)) -> ViewerData:
    """
    Create ViewerData from room estimator and ground truth.

    Args:
        room_estimator: RoomPoseEstimatorV2 instance
        robot_pose_gt: Ground truth pose [x, y, theta]
        lidar_points: Optional LIDAR points array
        step: Current step number
        innovation: Optional [dx, dy, dtheta] prediction error
        prior_precision: Adaptive prior precision
        lidar_subsample_factor: Current LIDAR subsampling factor
        optimizer_iterations: Number of iterations used in optimizer
        compute_time_ms: Total compute time in milliseconds
        cpu_percent: CPU usage percentage
        velocity_weights: Optional [w_x, w_y, w_theta] velocity-adaptive weights
        f_likelihood: F_likelihood term (SDF error = accuracy)
        f_prior: F_prior term (motion model deviation = complexity)
        vfe: Variational Free Energy after optimization
        speed_factor: Uncertainty-based speed modulation factor
        velocity_scale: Learned velocity calibration factor (k)

    Returns:
        ViewerData instance ready for the viewer
    """
    data = ViewerData()

    # Ground truth
    data.ground_truth_pose = RobotState(
        x=robot_pose_gt[0],
        y=robot_pose_gt[1],
        theta=robot_pose_gt[2]
    )

    # Ground truth room size
    data.gt_room_width = room_estimator.true_width
    data.gt_room_length = room_estimator.true_height

    # Estimated pose from belief
    if room_estimator.belief is not None:
        data.estimated_pose = RobotState(
            x=room_estimator.belief.x,
            y=room_estimator.belief.y,
            theta=room_estimator.belief.theta
        )
        data.room = RoomState(
            width=room_estimator.belief.width,
            length=room_estimator.belief.length
        )
        # Get pose covariance (2x2 submatrix for x,y)
        if hasattr(room_estimator.belief, 'pose_cov'):
            data.pose_covariance = room_estimator.belief.pose_cov[:2, :2].copy()
            data.pose_covariance_full = room_estimator.belief.pose_cov[:3, :3].copy()
    else:
        data.estimated_pose = RobotState()
        data.room = RoomState(
            width=room_estimator.true_width,
            length=room_estimator.true_height
        )
        # Default high covariance when no belief yet
        data.pose_covariance = np.eye(2) * 1.0  # 1m² variance
        data.pose_covariance_full = np.eye(3) * 1.0

    # LIDAR points
    data.lidar_points = lidar_points

    # Phase and errors
    data.phase = room_estimator.phase
    data.step = step

    # Compute pose error (vs GT)
    data.error_x = data.estimated_pose.x - data.ground_truth_pose.x
    data.error_y = data.estimated_pose.y - data.ground_truth_pose.y
    data.pose_error = np.sqrt(data.error_x**2 + data.error_y**2)

    # Compute angle error in degrees (vs GT)
    angle_diff = data.estimated_pose.theta - data.ground_truth_pose.theta
    data.angle_error = np.degrees(np.arctan2(np.sin(angle_diff), np.cos(angle_diff)))

    # Get latest SDF error from stats
    if len(room_estimator.stats['sdf_error_history']) > 0:
        data.sdf_error = room_estimator.stats['sdf_error_history'][-1]

    # Innovation (prediction error)
    if innovation is not None:
        data.innovation_x = innovation[0]
        data.innovation_y = innovation[1]
        data.innovation_theta = np.degrees(innovation[2])
    data.prior_precision = prior_precision

    # CPU/Performance stats
    data.lidar_subsample_factor = lidar_subsample_factor
    data.optimizer_iterations = optimizer_iterations
    data.compute_time_ms = compute_time_ms
    data.cpu_percent = cpu_percent

    # Velocity-adaptive weights
    if velocity_weights is not None:
        data.velocity_weights = velocity_weights

    # Free Energy components
    data.f_likelihood = f_likelihood
    data.f_prior = f_prior
    data.vfe = vfe

    # Speed modulation factor
    data.speed_factor = speed_factor

    # Velocity calibration factor
    data.velocity_scale = velocity_scale

    # Commanded velocities
    data.cmd_adv_x = cmd_vel[0]  # Forward velocity (mm/s)
    data.cmd_adv_y = cmd_vel[1]  # Lateral velocity (mm/s)
    data.cmd_rot = cmd_vel[2]    # Rotation velocity (rad/s)

    return data
