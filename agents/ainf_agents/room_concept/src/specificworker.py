#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2026 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

from PySide6.QtCore import QTimer
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import traceback
import numpy as np
import time
import psutil
import os
from src.concept_room import RoomPoseEstimatorV2
from src.room_viewer import RoomViewerDPG, RoomSubject, ViewerData, create_viewer_data
from src.dsr_graph_viewer import DSRGraphViewerDPG

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

from pydsr import *
import torch



class SpecificWorker(GenericWorker):
    """
    Active Inference Room and Pose Estimator.

    This component implements variational inference for joint room geometry and robot
    pose estimation using the Active Inference framework. The system operates in two
    phases following the theoretical development in main.tex:

    MATHEMATICAL FOUNDATION (see main.tex Section 5.2):
    ====================================================

    Hidden State (Eq. in Sec. 5.2.1):
        s_loc = (x, y, θ, W, L)ᵀ

        where (x, y, θ) is robot pose in room frame, and (W, L) are room dimensions.

    Variational Free Energy (Sec. 2.2, Eq. 5):
        F[q, o] = D_KL[q(s) || p(s|o)] - ln p(o)

        Minimizing F makes q(s) approximate the true posterior p(s|o).

    Free Energy Decomposition (Sec. 2.2.1, Eq. 11):
        F = D_KL[q(s) || p(s)]  -  E_q[ln p(o|s)]
            \_____________/        \____________/
              Complexity              Accuracy

    TWO-PHASE OPERATION:
    ====================

    Phase 1 - Initialization (Sec. 5.2.3):
        Robot static, estimate full state s = (x, y, θ, W, L) from LIDAR.
        Uses Laplace approximation: q(s) = N(μ, Σ)

    Phase 2 - Tracking (Sec. 5.2.4):
        Room fixed, update pose using motion model + LIDAR correction.
        Prediction: s_pred = f(s_prev, u)    [motion model]
        Correction: minimize F = F_likelihood + π_prior · F_prior

    COORDINATE CONVENTION:
    ======================
    - Room frame: origin at center, X+ right, Y+ forward
    - Robot frame: X+ right, Y+ forward (direction facing)
    - Walls at x = ±W/2, y = ±L/2

    References:
        - main.tex Section 2: Fundamental Equations of Active Inference
        - main.tex Section 5.2: Perception (I): Variational Inference for Room and Pose
        - ACTIVE_INFERENCE_MATH.md: Implementation-specific mathematical details
    """

    def __init__(self, proxy_map, configData, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map, configData)
        self.Period = configData["Period"]["Compute"]

        try:
            signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.update_node_att)
            signals.connect(self.g, signals.UPDATE_NODE, self.update_node)
            signals.connect(self.g, signals.DELETE_NODE, self.delete_node)
            signals.connect(self.g, signals.UPDATE_EDGE, self.update_edge)
            signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.update_edge_att)
            signals.connect(self.g, signals.DELETE_EDGE, self.delete_edge)
        except RuntimeError as e:
            print(e)
    
        if startup_check:
            self.startup_check()
        else:
            # Initialize total step counter
            self.total_steps = 0

            self.robot_pose_gt = [0.0, 0.0, 0.0]  # Ground truth robot pose [x, y, theta]
            self.last_cmd = [0.0, 0.0, 0.0]  # Last command issued to robot (advx, advz, rot)

            # Create room estimator - room is 6m x 4m
            self.room_estimator = RoomPoseEstimatorV2(
                room_width=6.0,   # width in x direction (lateral) - 6m
                room_height=4.0,  # length in y direction (forward) - 4m
                use_known_room_dimensions=False  # Estimate room dimensions
            )

            # Statistics tracking
            self.pose_errors = []
            self.sdf_errors = []
            self.stats_printed = False

            # DSR graph state
            self.room_node_inserted = False  # True once room node is in the graph
            self.room_node_id = 100          # ID for the room node

            # Create room viewer with observer pattern
            self.viewer_subject = RoomSubject()

            # Create DSR graph viewer (not started independently - integrated into room viewer)
            # Height matches room canvas: window_height(900) - stats_height(220) - margin(50) = 630
            self.dsr_viewer = DSRGraphViewerDPG(
                g=self.g,
                window_width=300,    # Width of DSR panel
                window_height=630,   # Same height as room canvas
                update_period_ms=500,
                canvas_tag="dsr_canvas"
            )
            print("[SpecificWorker] DSR graph viewer created")

            # Create room viewer with integrated DSR viewer
            self.room_viewer = RoomViewerDPG(
                window_width=900,   # Wider to accommodate DSR panel + stats
                window_height=900,
                margin=0.5,
                show_lidar=True,
                dsr_viewer=self.dsr_viewer
            )
            self.viewer_subject.attach(self.room_viewer)
            self.plan_running = False
            self.selected_plan = 0  # 0 = Basic, 1 = Circle

            # Initialize trajectory state variables
            self._trajectory_initialized = False
            self._trajectory = []
            self._current_action_idx = 0
            self._action_step = 0
            self._action_total_steps = 0
            self._trajectory_complete = False
            self._waiting_for_tracking = False

            # Circle Center plan state variables
            self._circle_radius = 0.9          # Target orbit radius (meters)
            self._orbit_target_angle = 0.0     # Target angle for orbit completion
            self._orbit_start_angle = 0.0      # Starting angle in orbit
            self._goto_phase = 0               # 0=turn_to_target, 1=move_to_circle, 2=turn_tangent

            # Uncertainty-based speed modulation parameters
            # Speed is reduced when pose uncertainty is high (precision-weighted action)
            self._speed_modulation_enabled = True
            self._min_speed_factor = 0.3       # Minimum speed factor (30% of commanded)
            self._uncertainty_scale = 3.0      # Sensitivity to uncertainty (lower = less aggressive)
            self._uncertainty_threshold = 0.05 # Below this, no reduction (meters)
            self._current_speed_factor = 1.0   # Current speed modulation factor
            self._speed_factor_smoothing = 0.2 # EMA smoothing factor (lower = smoother)

            # Adaptive LIDAR subsampling (passed to proxy call)
            # 1 = no subsampling, 2 = one of two, 4 = one of four, etc.
            self._lidar_subsample_factor = 4  # Default value
            self._adaptive_lidar_enabled = True  # Enable adaptive subsampling
            self._last_vfe = 0.0  # Last VFE for adaptive decimation

            # CPU monitoring
            self._process = psutil.Process(os.getpid())
            self._cpu_percent = 0.0  # Current CPU usage percentage
            self._num_cores = psutil.cpu_count()  # Number of CPU cores

            # Timing statistics
            self._timing_stats = {
                'lidar': [],
                'estimator': [],
                'explore': [],
                'graph': [],
                'total': [],
                'iterations': [],
                'lidar_points': [],
                'subsample_factor': [],
                'cpu_percent': []
            }

            self.room_viewer.set_plan_toggle_callback(self._on_plan_toggle)
            self.room_viewer.set_plan_selected_callback(self._on_plan_selected)
            self.room_viewer.start()
            print("[SpecificWorker] Room viewer started with integrated DSR panel (plan stopped)")

            self.print_graph()

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""
        if hasattr(self, 'room_viewer'):
            self.room_viewer.stop()
        # dsr_viewer is integrated into room_viewer, no separate cleanup needed


    def _on_plan_toggle(self, running: bool):
        """Callback from viewer Start/Stop button"""
        self.plan_running = running
        if running:
            # Reset trajectory when starting
            self._trajectory_initialized = False
            print(f"[SpecificWorker] Plan STARTED (using plan {self.selected_plan})")
        else:
            print("[SpecificWorker] Plan STOPPED")
            self.omnirobot_proxy.setSpeedBase(0, 0, 0)

    def _on_plan_selected(self, plan_index: int):
        """Callback from viewer plan dropdown"""
        self.selected_plan = plan_index
        # Reset trajectory so it uses the new plan on next start
        self._trajectory_initialized = False
        plan_names = ["Basic (Static)", "Circle (0.75m + 360°)", "Ocho (Figure-8)", "Circle Center (r=1.5m)"]
        plan_name = plan_names[plan_index] if plan_index < len(plan_names) else f"Unknown ({plan_index})"
        print(f"[SpecificWorker] Plan changed to: {plan_name}")

    @QtCore.Slot()
    def compute(self):
        """
        Main perception-action loop implementing Active Inference.

        This method realizes the core Active Inference cycle described in main.tex
        Section 3.2 (Perception as free energy minimisation) and Section 5:

        PERCEPTION-ACTION CYCLE:
        ========================

        1. OBSERVE: Acquire LIDAR observations o_t
           - Upper LIDAR points detect room walls
           - Points in robot frame: o = {p_j}_{j=1}^M

        2. INFER (Perception): Update beliefs by minimizing Variational Free Energy
           - F[q, o] = E_q[ln q(s) - ln p(o,s)]    [main.tex Eq. 5]
           - Uses previous velocity command for motion model prediction

        3. ACT: Execute exploration trajectory
           - Actions affect future observations o(a)
           - Cannot change current F, but shapes future F

        4. UPDATE GRAPH: Propagate beliefs to DSR knowledge graph
           - Room node with dimensions (W, L)
           - RT edge with pose (x, y, θ) and covariance Σ

        TIMING MODEL:
        =============
        - At time t: LIDAR reflects state after command u_{t-1}
        - Command u_t issued now will affect state at t+1
        - This temporal ordering is critical for correct state estimation

        References:
            main.tex Section 3.2: Perception as free energy minimisation
            main.tex Section 5.2: Room and Pose Estimation
        """

        if not self.plan_running:
            return True

        t_start = time.perf_counter()

        # Get lidar data (comes in mm, convert to meters)
        # LIDAR reflects robot position at time t
        t0 = time.perf_counter()
        lidar_points = self.get_lidar_points()
        t_lidar = time.perf_counter() - t0

        # Update room estimator using command from PREVIOUS step
        # The previous command was executed during interval [t-dt, t]
        # So LIDAR at time t reflects the result of that command
        t0 = time.perf_counter()
        self.update_room_estimator(self.last_cmd, lidar_points)
        t_estimator = time.perf_counter() - t0

        # Execute exploration movement and get the command for NEXT interval
        # This command will be executed during interval [t, t+dt]
        t0 = time.perf_counter()
        self.last_cmd = self.explore_room()  # Returns (advx, advz, rot)
        t_explore = time.perf_counter() - t0

        # Keep DSR graph in sync: insert room when converged, update RT edge each step
        t0 = time.perf_counter()
        self.update_graph_from_estimator()
        t_graph = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start

        # Store timing stats
        self._timing_stats['lidar'].append(t_lidar)
        self._timing_stats['estimator'].append(t_estimator)
        self._timing_stats['explore'].append(t_explore)
        self._timing_stats['graph'].append(t_graph)
        self._timing_stats['total'].append(t_total)
        self._timing_stats['subsample_factor'].append(self._lidar_subsample_factor)
        self._timing_stats['lidar_points'].append(len(lidar_points))

        # Measure CPU usage (percentage across all cores)
        self._cpu_percent = self._process.cpu_percent()
        self._timing_stats['cpu_percent'].append(self._cpu_percent)

        # Print timing every 50 steps
        if self.total_steps > 0 and self.total_steps % 50 == 0:
            self._print_timing_stats()

        self.total_steps += 1
        return True

    def _print_timing_stats(self):
        """Print timing statistics"""
        n = len(self._timing_stats['total'])
        if n == 0:
            return

        # Calculate averages in milliseconds
        avg_lidar = np.mean(self._timing_stats['lidar']) * 1000
        avg_estimator = np.mean(self._timing_stats['estimator']) * 1000
        avg_explore = np.mean(self._timing_stats['explore']) * 1000
        avg_graph = np.mean(self._timing_stats['graph']) * 1000
        avg_total = np.mean(self._timing_stats['total']) * 1000

        # Calculate percentages
        if avg_total > 0:
            pct_lidar = (avg_lidar / avg_total) * 100
            pct_estimator = (avg_estimator / avg_total) * 100
            pct_explore = (avg_explore / avg_total) * 100
            pct_graph = (avg_graph / avg_total) * 100
        else:
            pct_lidar = pct_estimator = pct_explore = pct_graph = 0

        print(f"\n⏱️  TIMING STATS (avg over {n} steps, Period={self.Period}ms):")
        print(f"   LIDAR:     {avg_lidar:6.2f}ms ({pct_lidar:5.1f}%)")
        print(f"   Estimator: {avg_estimator:6.2f}ms ({pct_estimator:5.1f}%)")
        print(f"   Explore:   {avg_explore:6.2f}ms ({pct_explore:5.1f}%)")
        print(f"   Graph:     {avg_graph:6.2f}ms ({pct_graph:5.1f}%)")
        print(f"   TOTAL:     {avg_total:6.2f}ms")

        # Show adaptive CPU stats if available
        if len(self._timing_stats['iterations']) > 0:
            avg_iters = np.mean(self._timing_stats['iterations'])
            max_iters = self.room_estimator.max_iterations_tracking
            print(f"   Optimizer: {avg_iters:.1f}/{max_iters} iterations avg (early stop enabled)")

        if len(self._timing_stats['lidar_points']) > 0:
            avg_pts = np.mean(self._timing_stats['lidar_points'])
            avg_subsample = np.mean(self._timing_stats['subsample_factor'])
            print(f"   LIDAR pts: {avg_pts:.0f} points avg (subsample factor: {avg_subsample:.1f})")

        # Show frame skip stats
        if hasattr(self.room_estimator, 'stats') and 'frames_skipped' in self.room_estimator.stats:
            skipped = self.room_estimator.stats['frames_skipped']
            if skipped > 0:
                print(f"   Frames skipped: {skipped} (when stable & static)")

        # Show prediction early exit stats
        if hasattr(self.room_estimator, 'stats') and 'prediction_early_exits' in self.room_estimator.stats:
            early_exits = self.room_estimator.stats['prediction_early_exits']
            if early_exits > 0:
                print(f"   Prediction early exits: {early_exits} (optimizer skipped)")

        # Show CPU usage
        if len(self._timing_stats['cpu_percent']) > 0:
            avg_cpu = np.mean(self._timing_stats['cpu_percent'])
            max_cpu = np.max(self._timing_stats['cpu_percent'])
            # cpu_percent() returns percentage per core, so 100% = 1 core fully used
            cores_used = avg_cpu / 100.0
            print(f"   CPU: {avg_cpu:.1f}% avg, {max_cpu:.1f}% max ({cores_used:.2f} cores of {self._num_cores})")

        # Clear stats for next batch
        for key in self._timing_stats:
            self._timing_stats[key] = []

    def update_room_estimator(self, last_cmd, lidar_points):
        """
        Update room and pose beliefs by minimizing Variational Free Energy.

        This method implements the core perception update of Active Inference,
        approximating the posterior p(s|o) by minimizing:

            F[q, o] = D_KL[q(s) || p(s)] - E_q[ln p(o|s)]
                      \_______________/   \_____________/
                        Complexity          Accuracy

        PHASE 1 - INITIALIZATION (main.tex Sec. 5.2.3):
        ===============================================
        Estimate full state s = (x, y, θ, W, L) from accumulated LIDAR points.

        Likelihood (SDF-based):
            p(o|s) ∝ exp(-1/(2σ²) · Σᵢ SDF(pᵢ)²)

        where SDF measures distance of transformed LIDAR points to room walls.

        PHASE 2 - TRACKING (main.tex Sec. 5.2.4):
        =========================================
        Room dimensions fixed, update pose s = (x, y, θ).

        Prediction step (motion model):
            s_pred = f(s_prev, u) = s_prev + [R(θ)·v·Δt, ω·Δt]ᵀ

        Correction step (minimize Free Energy):
            F = F_likelihood + π_prior · F_prior

        where:
            F_likelihood = (1/N) Σᵢ SDF(pᵢ)²           [LIDAR fit]
            F_prior = ½(s - s_pred)ᵀ Σ_pred⁻¹ (s - s_pred)  [motion model]
            π_prior = adaptive precision (see ACTIVE_INFERENCE_MATH.md Sec. 7)

        Args:
            last_cmd: Last velocity command (advx, advz, rot) in mm/s and rad/s
            lidar_points: LIDAR wall points in robot frame [N, 2] in meters

        Returns:
            Updates self.room_estimator.belief with posterior (μ, Σ)

        References:
            main.tex Sec. 5.2.2: Variational Free Energy Objective
            main.tex Sec. 5.2.3: Approximation 1 - Laplace Approximation
            ACTIVE_INFERENCE_MATH.md Sec. 3: Two-Phase Estimation
        """
        if len(lidar_points) > 0:
            # Convert last command to velocity in m/s for the estimator
            velocity = np.array([last_cmd[0] / 1000.0, last_cmd[1] / 1000.0])  # [vx, vy] in m/s
            angular_velocity = -last_cmd[2]  # rad/s (negated to match coordinate transform)

            # Time step in seconds
            dt = self.Period / 1000.0

            # Update room estimator
            result = self.room_estimator.update(
                robot_pose_gt=self.robot_pose_gt,
                robot_velocity=velocity,
                angular_velocity=angular_velocity,
                dt=dt,
                lidar_points=lidar_points
            )

            # Track iterations and LIDAR points used for CPU stats
            if 'iterations_used' in result:
                self._timing_stats['iterations'].append(result['iterations_used'])
            if 'lidar_points_used' in result:
                self._timing_stats['lidar_points'].append(result['lidar_points_used'])

            # Print status every 20 steps
            if self.total_steps % 20 == 0:
                phase = result.get('phase', result.get('status', 'unknown'))
                sdf_err = result.get('sdf_error', 0)
                uncertainty = result.get('belief_uncertainty', 0)

                # Get estimated pose from belief
                est_x = self.room_estimator.belief.x if self.room_estimator.belief else 0
                est_y = self.room_estimator.belief.y if self.room_estimator.belief else 0
                est_theta = self.room_estimator.belief.theta if self.room_estimator.belief else 0

                # Compute error against ground truth (for evaluation only)
                gt_x, gt_y, gt_theta = self.robot_pose_gt
                pose_error = np.sqrt((est_x - gt_x)**2 + (est_y - gt_y)**2)

                # Compute angle error (handling wrapping)
                angle_diff = est_theta - gt_theta
                angle_error = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

                # Track errors for statistics
                self.pose_errors.append(pose_error)
                self.sdf_errors.append(sdf_err)

            # Update viewer with current data
            # Extract innovation and prior precision from result if available
            innovation = result.get('innovation', None)
            prior_precision = result.get('prior_precision', 0.0)
            optimizer_iterations = result.get('iterations_used', 0)
            velocity_weights = result.get('velocity_weights', None)

            # Free Energy components
            f_likelihood = result.get('f_likelihood', 0.0)
            f_prior = result.get('f_prior', 0.0)
            vfe = result.get('vfe', 0.0)

            # Store VFE for adaptive LIDAR decimation
            self._last_vfe = vfe

            # Get last compute time from timing stats
            last_compute_time = self._timing_stats['total'][-1] * 1000 if len(self._timing_stats['total']) > 0 else 0.0

            viewer_data = create_viewer_data(
                room_estimator=self.room_estimator,
                robot_pose_gt=self.robot_pose_gt,
                lidar_points=lidar_points,
                step=self.total_steps,
                innovation=innovation,
                prior_precision=prior_precision,
                lidar_subsample_factor=self._lidar_subsample_factor,
                optimizer_iterations=optimizer_iterations,
                compute_time_ms=last_compute_time,
                cpu_percent=self._cpu_percent,
                velocity_weights=velocity_weights,
                f_likelihood=f_likelihood,
                f_prior=f_prior,
                vfe=vfe,
                speed_factor=self._current_speed_factor,
                cmd_vel=tuple(self.last_cmd)
            )
            self.viewer_subject.notify(viewer_data)

    def print_statistics_summary(self):
        """Print a summary of statistics at the end of the run"""
        if self.stats_printed:
            return
        self.stats_printed = True

        print("\n" + "="*60)
        print("           ACTIVE INFERENCE ROOM ESTIMATOR - SUMMARY")
        print("="*60)

        # Plan executed
        plan_names = ["Basic (Static)", "Circle (0.75m + 360°)", "Ocho (Figure-8)", "Circle Center (r=1.5m)"]
        plan_name = plan_names[self.selected_plan] if self.selected_plan < len(plan_names) else f"Unknown ({self.selected_plan})"
        print(f"\n🎯 PLAN EXECUTED: {plan_name}")
        if self._trajectory_initialized and len(self._trajectory) > 0:
            completed_actions = min(self._current_action_idx + 1, len(self._trajectory))
            print(f"   Actions completed: {completed_actions}/{len(self._trajectory)}")
            if self._trajectory_complete:
                print(f"   Status: ✓ Complete")
            else:
                current_action = self._get_current_action_description()
                print(f"   Status: In progress - {current_action}")

        # Room estimation
        if self.room_estimator.belief:
            est_w = self.room_estimator.belief.width
            est_l = self.room_estimator.belief.length
            true_w = self.room_estimator.true_width
            true_l = self.room_estimator.true_height
            room_err = np.sqrt((est_w - true_w)**2 + (est_l - true_l)**2)

            print(f"\n📐 ROOM ESTIMATION:")
            print(f"   True room:      {true_w:.2f} x {true_l:.2f} m")
            print(f"   Estimated room: {est_w:.2f} x {est_l:.2f} m")
            print(f"   Room error:     {room_err:.3f} m")

        # Pose errors
        if len(self.pose_errors) > 0:
            pose_arr = np.array(self.pose_errors)
            print(f"\n📍 POSE ERROR (vs Ground Truth):")
            print(f"   Mean:    {np.mean(pose_arr):.3f} m")
            print(f"   Std:     {np.std(pose_arr):.3f} m")
            print(f"   Min:     {np.min(pose_arr):.3f} m")
            print(f"   Max:     {np.max(pose_arr):.3f} m")
            print(f"   Final:   {pose_arr[-1]:.3f} m")

        # SDF errors
        if len(self.sdf_errors) > 0:
            sdf_arr = np.array(self.sdf_errors)
            print(f"\n📊 SDF ERROR (LIDAR fit):")
            print(f"   Mean:    {np.mean(sdf_arr):.3f} m")
            print(f"   Std:     {np.std(sdf_arr):.3f} m")
            print(f"   Min:     {np.min(sdf_arr):.3f} m")
            print(f"   Max:     {np.max(sdf_arr):.3f} m")
            print(f"   Final:   {sdf_arr[-1]:.3f} m")

        # Final pose comparison
        if self.room_estimator.belief:
            gt_x, gt_y, gt_theta = self.robot_pose_gt
            est_x = self.room_estimator.belief.x
            est_y = self.room_estimator.belief.y
            est_theta = self.room_estimator.belief.theta

            print(f"\n🤖 FINAL POSE:")
            print(f"   Ground Truth: x={gt_x:.3f}m, y={gt_y:.3f}m, θ={gt_theta:.3f}rad")
            print(f"   Estimated:    x={est_x:.3f}m, y={est_y:.3f}m, θ={est_theta:.3f}rad")
            print(f"   Position err: {np.sqrt((est_x-gt_x)**2 + (est_y-gt_y)**2):.3f} m")

            # Angle error (handle wrapping)
            angle_err = abs(est_theta - gt_theta)
            if angle_err > np.pi:
                angle_err = 2*np.pi - angle_err
            print(f"   Angle error:  {np.degrees(angle_err):.1f}°")

        print(f"\n⏱️  Total steps: {self.total_steps}")
        print("="*60 + "\n")

    def get_lidar_points(self) -> np.ndarray:
        """
        Acquire LIDAR observations for room perception.

        Returns wall points detected by upper LIDAR, which constitute the
        observations o = {pⱼ}ⱼ₌₁ᴹ used in the generative model (main.tex Sec. 5.2.1).

        OBSERVATION MODEL:
        ==================
        Each LIDAR point pⱼ in robot frame is transformed to room frame:

            pⱼ^room = R(θ) · pⱼ^robot + [x, y]ᵀ

        where R(θ) is the 2D rotation matrix:

            R(θ) = [cos(θ)  -sin(θ)]
                   [sin(θ)   cos(θ)]

        The likelihood p(o|s) is based on the SDF (Signed Distance Function):

            p(o|s) ∝ exp(-1/(2σ²) · Σⱼ SDF(pⱼ^room)²)

        ADAPTIVE SUBSAMPLING:
        =====================
        Uses stability-based subsampling to reduce CPU usage when system is stable:
        - SDF error < τ: high subsampling (fewer points, faster)
        - SDF error > 4τ: no subsampling (all points, more accurate)

        Coordinate convention:
            Robot frame: x+ = right, y+ = forward

        Returns:
            np.ndarray: [N, 2] array of LIDAR points in robot frame (meters)

        References:
            main.tex Sec. 5.2.1: Hidden State and Observations
            main.tex Sec. 5.4.1: LIDAR Observations in the Room Frame
        """
        try:
            # Adaptive LIDAR subsampling based on estimation quality
            # When quality degrades (high VFE, SDF, or covariance), reduce subsampling
            # to get more points and improve estimation
            if self._adaptive_lidar_enabled and hasattr(self, 'room_estimator'):
                estimator = self.room_estimator

                # Get quality metrics
                last_sdf = getattr(estimator, '_last_sdf_error', float('inf'))
                stability_threshold = getattr(estimator, 'stability_threshold', 0.05)

                # Get covariance if available
                pose_uncertainty = 0.0
                if estimator.belief is not None:
                    pose_cov = estimator.belief.pose_cov
                    pose_uncertainty = np.trace(pose_cov[:2, :2])  # x, y uncertainty

                # Get last VFE if available (stored from previous update)
                last_vfe = getattr(self, '_last_vfe', 0.0)

                # Thresholds for quality assessment
                sdf_good = stability_threshold          # 0.05
                sdf_medium = stability_threshold * 1.5  # 0.075
                sdf_bad = stability_threshold * 2.5     # 0.125

                cov_good = 0.02    # Low uncertainty
                cov_medium = 0.05  # Medium uncertainty
                cov_bad = 0.1      # High uncertainty

                vfe_good = 0.1     # Low VFE
                vfe_medium = 0.3   # Medium VFE
                vfe_bad = 0.5      # High VFE

                # Compute quality score (0 = bad, 1 = good) for each metric
                sdf_score = 1.0 if last_sdf < sdf_good else (0.5 if last_sdf < sdf_medium else (0.25 if last_sdf < sdf_bad else 0.0))
                cov_score = 1.0 if pose_uncertainty < cov_good else (0.5 if pose_uncertainty < cov_medium else (0.25 if pose_uncertainty < cov_bad else 0.0))
                vfe_score = 1.0 if last_vfe < vfe_good else (0.5 if last_vfe < vfe_medium else (0.25 if last_vfe < vfe_bad else 0.0))

                # Combined quality: minimum of all scores (worst metric dominates)
                quality = min(sdf_score, cov_score, vfe_score)

                # Map quality to subsampling factor (responsive to degradation)
                if quality >= 0.75:
                    # Excellent quality - can use high subsampling
                    target_factor = 4
                elif quality >= 0.5:
                    # Good quality - moderate subsampling
                    target_factor = 3
                elif quality >= 0.25:
                    # Degraded quality - minimal subsampling
                    target_factor = 2
                else:
                    # Poor quality - use all points
                    target_factor = 1

                # Immediate response to quality degradation (no smoothing when reducing)
                # But smooth when increasing subsampling to avoid oscillation
                if target_factor < self._lidar_subsample_factor:
                    # Quality degraded - immediately reduce subsampling
                    self._lidar_subsample_factor = target_factor
                elif target_factor > self._lidar_subsample_factor:
                    # Quality improved - gradually increase subsampling
                    # Only increase if quality has been stable
                    self._lidar_subsample_factor = min(
                        self._lidar_subsample_factor + 1,
                        target_factor
                    )

            # Get 2D LIDAR data with adaptive subsampling
            # decimationDegreeFactor: 1 = no subsampling, 2 = one of two, etc.
            helios = self.lidar3d_proxy.getLidarDataWithThreshold2d(
                "helios", 8000, self._lidar_subsample_factor
            )

            # OPTIMIZED: Vectorized LIDAR processing
            n_points = len(helios.points)
            if n_points > 0:
                # Extract all points as numpy array (single pass)
                xyz = np.array([(p.x, p.y, p.z) for p in helios.points], dtype=np.float32)

                # Vectorized filtering (z > 1000mm) and convert to meters
                mask = xyz[:, 2] > 1000
                if np.any(mask):
                    # Extract x, y and convert mm -> m in one step
                    lidar_points = xyz[mask, :2] * 0.001  # Multiply faster than divide
                else:
                    lidar_points = np.empty((0, 2), dtype=np.float32)
            else:
                lidar_points = np.empty((0, 2), dtype=np.float32)


        except Ice.Exception as e:
            print(f"Error reading lidar: {e}")
            lidar_points = np.array([])

        return lidar_points

    def explore_room(self):
        """
        Execute exploration policy to gather informative observations.

        In Active Inference, action selection minimizes Expected Free Energy G(π)
        over future trajectories (main.tex Sec. 4). For room estimation, we use
        predefined exploration trajectories that maximize information gain.

        EXPECTED FREE ENERGY (main.tex Sec. 4.1):
        =========================================
        G(π) = E_q(s̃,õ|π)[F[q, õ]]

        Decomposes into (main.tex Sec. 4.2):
            G(π) = Epistemic Value + Pragmatic Value
                 = Information Gain + Goal Achievement

        EXPLORATION RATIONALE:
        ======================
        Movement generates diverse viewpoints, reducing uncertainty about:
        - Room dimensions (W, L) during initialization
        - Robot pose (x, y, θ) during tracking

        Circular/figure-8 trajectories are particularly effective because they:
        1. Observe all walls from multiple angles
        2. Create large pose changes that disambiguate orientation
        3. Return to start, allowing drift assessment

        AVAILABLE PLANS:
        ================
        - Plan 0 (Basic): Static detection, wait for room convergence
        - Plan 1 (Circle): Forward 0.75m, then 360° arc
        - Plan 2 (Ocho): Figure-8 pattern, 1m loops

        Returns:
            tuple: (advx, advz, rot) velocity command in mm/s and rad/s

        References:
            main.tex Sec. 4: From Free Energy to Action Selection
            main.tex Sec. 4.3: Unifying Planning and Reactive Control
        """
        # Initialize trajectory on first call or when reset
        if not self._trajectory_initialized:
            self._init_trajectory()

        return self._execute_trajectory()

    def _compute_speed_factor(self) -> float:
        """
        Compute speed modulation factor based on pose uncertainty.

        This implements precision-weighted action from Active Inference:
        - High precision (low uncertainty) → full speed
        - Low precision (high uncertainty) → reduced speed

        The factor is computed from the positional covariance:
            σ_pos = sqrt(σ²ₓ + σ²ᵧ)
            factor = min_factor + (1 - min_factor) × exp(-scale × σ_pos)

        Returns:
            float: Speed factor in [min_speed_factor, 1.0]

        References:
            ACTIVE_INFERENCE_MATH.md Sec. 7: Velocity-Adaptive Precision Weighting
        """
        if not self._speed_modulation_enabled:
            return 1.0

        if self.room_estimator.belief is None:
            return self._min_speed_factor  # Maximum caution when no estimate

        # Get pose covariance
        pose_cov = self.room_estimator.belief.pose_cov  # 3x3 matrix [x, y, theta]

        # Compute positional uncertainty (standard deviation in meters)
        sigma_x = np.sqrt(max(0, pose_cov[0, 0]))
        sigma_y = np.sqrt(max(0, pose_cov[1, 1]))
        sigma_pos = np.sqrt(sigma_x**2 + sigma_y**2)

        # Below threshold, no reduction
        if sigma_pos < self._uncertainty_threshold:
            target_factor = 1.0
        else:
            # Exponential decay: factor → min_factor as uncertainty → ∞
            excess_uncertainty = sigma_pos - self._uncertainty_threshold
            target_factor = self._min_speed_factor + \
                           (1.0 - self._min_speed_factor) * \
                           np.exp(-self._uncertainty_scale * excess_uncertainty)

        # Smooth the factor with EMA to avoid jerky motion
        self._current_speed_factor = (1 - self._speed_factor_smoothing) * self._current_speed_factor + \
                                     self._speed_factor_smoothing * target_factor

        return self._current_speed_factor

    def _apply_speed_modulation(self, cmd: tuple) -> tuple:
        """
        Apply uncertainty-based speed modulation to velocity command.

        Args:
            cmd: (advx, advz, rot) raw velocity command

        Returns:
            (advx, advz, rot) modulated velocity command
        """
        factor = self._compute_speed_factor()

        advx, advz, rot = cmd
        return (advx * factor, advz * factor, rot * factor)

    def _get_current_action_description(self) -> str:
        """Get a human-readable description of the current action being executed"""
        if not self._trajectory_initialized:
            return "Not started"
        if self._trajectory_complete:
            return "Complete"
        if self._current_action_idx >= len(self._trajectory):
            return "Complete"

        action_type, value = self._trajectory[self._current_action_idx]
        progress = f"({self._action_step}/{self._action_total_steps})" if self._action_total_steps > 0 else ""

        if action_type == 'forward':
            return f"Forward {value:.2f}m {progress}"
        elif action_type == 'backward':
            return f"Backward {value:.2f}m {progress}"
        elif action_type == 'turn':
            direction = "left" if value > 0 else "right"
            return f"Turn {abs(value):.0f}° {direction} {progress}"
        elif action_type == 'arc':
            return f"Arc {value:.1f}s {progress}"
        elif action_type == 'arc_right':
            return f"Arc RIGHT {value:.1f}s {progress}"
        elif action_type == 'arc_left':
            return f"Arc LEFT {value:.1f}s {progress}"
        elif action_type == 'wait':
            return f"Wait {value:.1f}s {progress}"
        elif action_type == 'hold_for_tracking':
            return "Waiting for tracking..."
        elif action_type == 'statistics':
            return "Statistics"
        elif action_type == 'goto_circle_start':
            # Show distance to circle
            if self.room_estimator.belief is not None:
                est_x = self.room_estimator.belief.x
                est_y = self.room_estimator.belief.y
                dist = np.sqrt(est_x**2 + est_y**2)
                remaining = abs(value - dist)
                return f"Goto circle r={value:.1f}m (dist={remaining:.2f}m)"
            return f"Moving to circle (r={value:.1f}m) {progress}"
        elif action_type == 'orbit_center':
            # Show accumulated progress if available
            if hasattr(self, '_orbit_accumulated'):
                done_deg = np.degrees(self._orbit_accumulated)
                return f"Orbiting {done_deg:.0f}°/{value:.0f}°"
            return f"Orbiting center ({value:.0f}°) {progress}"
        else:
            return f"{action_type} {progress}"

    def _init_trajectory(self):
        """Initialize the trajectory system based on selected plan

        Plan 0 - Basic (Static): Wait for room detection without moving
        Plan 1 - Circle: Advance 0.75m, then do a complete circle
        """
        self._trajectory_initialized = True

        # Movement parameters
        self.advance_speed = 200.0    # mm/s
        self.rotation_speed = 0.15    # rad/s - LOW SPEED for better detection
        self.period_sec = self.Period / 1000.0

        if self.selected_plan == 0:
            # Plan 0: Basic - Wait without moving for room detection
            self._trajectory = [
                ('wait', 10.0),          # Wait 10 seconds for room detection
                ('hold_for_tracking', 0),  # Wait until tracking mode, then hold
            ]
            print("[Explorer] Using Plan 0: Basic (Static detection)")

        elif self.selected_plan == 1:
            # Plan 1: Circle - Advance 0.75m, then complete circle
            # This plan moves immediately without waiting for tracking
            # At rot=0.3 rad/s, 360° (2π rad) takes about 21 seconds
            self.rotation_speed = 0.3  # Faster rotation for circle
            self._trajectory = [
                ('wait', 1.0),           # Initial stabilization
                ('forward', 0.75),       # Advance 0.75m from center
                ('wait', 0.5),
                ('arc', 21.0),           # Circular arc for ~360° (full circle)
                ('wait', 1.0),
                ('statistics', 0),       # Print statistics summary
            ]
            print("[Explorer] Using Plan 1: Circle (0.75m + 360°)")

        elif self.selected_plan == 2:
            # Plan 2: Figure-8 (Ocho) - Two circles forming a figure 8
            # Each loop spans 1 meter on each side (2m total width)
            # Start from center, go right loop, return to center, go left loop
            self.rotation_speed = 0.3   # rad/s for smooth curves
            self.advance_speed = 150.0  # Slightly slower for precision

            # Figure 8 geometry:
            # - Each loop has radius ~0.5m (diameter 1m)
            # - Right loop: turn right (negative rotation) while advancing
            # - Left loop: turn left (positive rotation) while advancing
            # Time for 360° at 0.3 rad/s ≈ 21 seconds per loop
            self._trajectory = [
                ('wait', 1.0),              # Initial stabilization
                ('forward', 0.25),          # Move slightly forward to offset
                ('wait', 0.5),
                # Right loop (clockwise when viewed from above)
                ('arc_right', 21.0),        # Full circle turning right
                ('wait', 0.5),
                # Left loop (counter-clockwise)
                ('arc_left', 21.0),         # Full circle turning left
                ('wait', 0.5),
                ('backward', 0.25),         # Return to start
                ('wait', 1.0),
                ('statistics', 0),          # Print statistics summary
            ]
            print("[Explorer] Using Plan 2: Ocho (Figure-8, 1m per side)")

        elif self.selected_plan == 3:
            # Plan 3: Circle around room center
            # Phase 0: Turn to face center
            # Phase 1: Move to circle radius (1.5m)
            # Phase 2: Turn 90° to be tangent
            # Phase 3: Complete 360° orbit
            self.rotation_speed = 0.3    # rad/s for turning
            self.advance_speed = 150.0   # mm/s
            self._goto_phase = 0         # Reset phase

            self._trajectory = [
                ('wait', 1.0),                    # Initial stabilization
                ('goto_circle_start', 1.5),       # All 4 phases in one action
                ('wait', 1.0),
                ('statistics', 0),
            ]
            print("[Explorer] Using Plan 3: Circle around center (r=1.5m, 360°)")

        else:
            # Default to basic plan
            self._trajectory = [
                ('wait', 10.0),
                ('hold_for_tracking', 0),
            ]
            print(f"[Explorer] Unknown plan {self.selected_plan}, using Basic")

        self._current_action_idx = 0
        self._action_step = 0
        self._action_total_steps = 0
        self._trajectory_complete = False
        self._waiting_for_tracking = False

        self._print_trajectory()

    def _print_trajectory(self):
        """Print the trajectory plan"""
        print("[Explorer] Trajectory plan:")
        total_distance = 0
        total_rotation = 0
        for i, action in enumerate(self._trajectory):
            action_type, value = action
            if action_type == 'forward':
                print(f"  {i+1}. Forward {value:.2f}m")
                total_distance += value
            elif action_type == 'backward':
                print(f"  {i+1}. Backward {value:.2f}m")
                total_distance += value
            elif action_type == 'turn':
                direction = "left" if value > 0 else "right"
                print(f"  {i+1}. Turn {abs(value):.0f}° {direction}")
                total_rotation += abs(value)
            elif action_type == 'arc':
                # Estimate rotation during arc (value is duration in seconds)
                arc_rotation = np.degrees(self.rotation_speed * value)
                print(f"  {i+1}. Arc {value:.1f}s (~{arc_rotation:.0f}°)")
                total_rotation += arc_rotation
            elif action_type == 'arc_right':
                arc_rotation = np.degrees(self.rotation_speed * value)
                print(f"  {i+1}. Arc RIGHT {value:.1f}s (~{arc_rotation:.0f}° clockwise)")
                total_rotation += arc_rotation
            elif action_type == 'arc_left':
                arc_rotation = np.degrees(self.rotation_speed * value)
                print(f"  {i+1}. Arc LEFT {value:.1f}s (~{arc_rotation:.0f}° counter-clockwise)")
                total_rotation += arc_rotation
            elif action_type == 'wait':
                print(f"  {i+1}. Wait {value:.1f}s")
            elif action_type == 'hold_for_tracking':
                print(f"  {i+1}. HOLD - Wait for tracking mode")
            elif action_type == 'statistics':
                print(f"  {i+1}. Print statistics summary")
            elif action_type == 'goto_circle_start':
                print(f"  {i+1}. Go to circle start (radius={value:.2f}m)")
            elif action_type == 'orbit_center':
                print(f"  {i+1}. Orbit around center ({value:.0f}°)")
                total_rotation += value
        print(f"  Total: {total_distance:.2f}m distance, {total_rotation:.0f}° rotation")

    def _execute_trajectory(self):
        """Execute the current action in the trajectory"""
        cmd = (0.0, 0.0, 0.0)

        if self._trajectory_complete:
            self.omnirobot_proxy.setSpeedBase(0, 0, 0)
            return cmd

        if self._current_action_idx >= len(self._trajectory):
            # Trajectory complete
            if not self._trajectory_complete:
                self._trajectory_complete = True
                self.omnirobot_proxy.setSpeedBase(0, 0, 0)
                print(f"[Explorer] Trajectory complete - robot stopped")
                self.print_statistics_summary()
            return cmd

        # Get current action
        action_type, value = self._trajectory[self._current_action_idx]

        # Handle special action: hold_for_tracking
        if action_type == 'hold_for_tracking':
            self.omnirobot_proxy.setSpeedBase(0, 0, 0)

            # Check if estimator has transitioned to tracking mode
            if self.room_estimator.phase == 'tracking':
                if not self._waiting_for_tracking:
                    print(f"[Explorer] Room detection complete! Entering TRACKING mode (robot holding position)")
                    self._waiting_for_tracking = True
                # Stay here - don't advance to next action, just hold position
                # The robot will remain stationary in tracking mode
            else:
                if self._action_step == 0:
                    print(f"[Explorer] Waiting for room detection to converge...")
                self._action_step += 1
            return cmd

        # Calculate steps needed for this action (on first step)
        if self._action_step == 0:
            self._action_total_steps = self._calculate_action_steps(action_type, value)

        # Execute action
        if action_type == 'forward':
            cmd = (0, self.advance_speed, 0)
        elif action_type == 'backward':
            cmd = (0, -self.advance_speed, 0)
        elif action_type == 'turn':
            # Positive angle = left (positive rotation), negative = right
            rot_dir = 1 if value > 0 else -1
            cmd = (0, 0, rot_dir * self.rotation_speed)
        elif action_type == 'arc':
            # Circular arc: forward + rotation simultaneously (default left/CCW)
            cmd = (0, self.advance_speed, self.rotation_speed)
        elif action_type == 'arc_right':
            # Circular arc turning right (clockwise): forward + negative rotation
            cmd = (0, self.advance_speed, -self.rotation_speed)
        elif action_type == 'arc_left':
            # Circular arc turning left (counter-clockwise): forward + positive rotation
            cmd = (0, self.advance_speed, self.rotation_speed)
        elif action_type == 'wait':
            cmd = (0, 0, 0)
        elif action_type == 'statistics':
            # Print statistics and move to next action immediately
            self.print_statistics_summary()
            self._current_action_idx += 1
            self._action_step = 0
            return cmd

        elif action_type == 'goto_circle_start':
            # Move from current position to a point on the circle around room center
            # Uses estimated pose to calculate path (modulation applied inside)
            cmd = self._execute_goto_circle_start(value)  # value = radius
            return cmd

        elif action_type == 'orbit_center':
            # Orbit around the room center (modulation applied inside)
            cmd = self._execute_orbit_center(value)  # value = degrees
            return cmd

        # Apply uncertainty-based speed modulation
        cmd = self._apply_speed_modulation(cmd)

        self.omnirobot_proxy.setSpeedBase(*cmd)
        self._action_step += 1

        # Check if action complete
        if self._action_step >= self._action_total_steps:
            self._current_action_idx += 1
            self._action_step = 0

        return cmd

    def _calculate_action_steps(self, action_type, value):
        """Calculate number of steps needed for an action"""
        if action_type in ('forward', 'backward'):
            # value is distance in meters
            time_needed = abs(value) / (self.advance_speed / 1000.0)
            return int(time_needed / self.period_sec)
        elif action_type == 'turn':
            # value is angle in degrees
            angle_rad = abs(value) * np.pi / 180.0
            time_needed = angle_rad / self.rotation_speed
            return int(time_needed / self.period_sec)
        elif action_type in ('arc', 'arc_right', 'arc_left'):
            # value is duration in seconds
            return int(value / self.period_sec)
        elif action_type == 'wait':
            # value is time in seconds
            return int(value / self.period_sec)
        elif action_type == 'statistics':
            # Instant action - no steps needed
            return 0
        elif action_type in ('goto_circle_start', 'orbit_center'):
            # Dynamic actions - steps calculated during execution
            return 1  # Placeholder, actual control is continuous
        return 0

    def _execute_goto_circle_start(self, radius: float) -> tuple:
        """
        Move robot to circle around room center:
        Phase 0: Turn to face center
        Phase 1: Move forward/backward until at circle radius
        Phase 2: Turn 90° to be tangent
        Phase 3: Complete the arc (orbit around center)
        """
        if self.room_estimator.belief is None:
            return (0, 0, 0)

        # Get current estimated pose (robot in room frame)
        est_x = self.room_estimator.belief.x
        est_y = self.room_estimator.belief.y
        est_theta = self.room_estimator.belief.theta

        # Distance from center
        dist_to_center = np.sqrt(est_x**2 + est_y**2)

        # Center is at (0,0) in room frame
        # Transform center to robot frame:
        dx = -est_x
        dy = -est_y
        cos_t = np.cos(-est_theta)
        sin_t = np.sin(-est_theta)
        center_robot_x = dx * cos_t - dy * sin_t
        center_robot_y = dx * sin_t + dy * cos_t
        center_in_robot = (center_robot_x, center_robot_y)

        # Angle to center in robot frame (0 = straight ahead, positive = left)
        angle_to_center = np.arctan2(center_robot_x, center_robot_y)

        # Thresholds
        angle_threshold = 0.20  # ~11 degrees
        dist_threshold = 0.15   # 15 cm

        # Initialize phase
        if not hasattr(self, '_goto_phase'):
            self._goto_phase = 0

        # PHASE 0: Turn to face center
        if self._goto_phase == 0:
            if abs(angle_to_center) > angle_threshold:
                rot = np.clip(angle_to_center * 1.5, -self.rotation_speed, self.rotation_speed)
                cmd = (0, 0, rot)
                cmd = self._apply_speed_modulation(cmd)
                self.omnirobot_proxy.setSpeedBase(*cmd)
                self._action_step += 1
                return cmd
            else:
                print(f"[Circle] Phase 0: Aligned to center")
                self._goto_phase = 1

        # PHASE 1: Move forward/backward to reach circle
        if self._goto_phase == 1:
            dist_to_circle = radius - dist_to_center

            if abs(dist_to_circle) > dist_threshold:
                if dist_to_circle > 0:
                    speed = -self.advance_speed  # backward (away from center)
                else:
                    speed = self.advance_speed   # forward (toward center)

                cmd = (0, speed, 0)
                cmd = self._apply_speed_modulation(cmd)
                self.omnirobot_proxy.setSpeedBase(*cmd)
                self._action_step += 1
                return cmd
            else:
                print(f"[Circle] Phase 1: Reached circle at r={dist_to_center:.2f}m")
                self._goto_phase = 2

        # PHASE 2: Turn 90° left to be tangent (for CCW orbit)
        if self._goto_phase == 2:
            # We want the center to be at -90° (on our right) for CCW orbit
            # So we need to turn until angle_to_center reaches -90°
            target_angle = -np.pi / 2  # -90°

            # The rotation we need is simply to make the center move from
            # its current angle to -90°. If center is at +30°, we turn right (positive rot)
            # to make it appear at -90°. We need to turn by: current - target = 30 - (-90) = 120°
            # But we want the shortest path.

            # How much do we need to turn? If we rotate by 'r', the center angle changes by '-r'
            # New angle = angle_to_center - r
            # We want: angle_to_center - r = -90°
            # So: r = angle_to_center - (-90°) = angle_to_center + 90°

            rotation_needed = angle_to_center - target_angle
            # Normalize to [-pi, pi]
            while rotation_needed > np.pi:
                rotation_needed -= 2 * np.pi
            while rotation_needed < -np.pi:
                rotation_needed += 2 * np.pi

            if abs(rotation_needed) > angle_threshold:
                # Proportional control - rotate in the direction of rotation_needed
                rot = np.clip(rotation_needed * 1.5, -self.rotation_speed, self.rotation_speed)

                cmd = (0, 0, rot)
                cmd = self._apply_speed_modulation(cmd)
                self.omnirobot_proxy.setSpeedBase(*cmd)
                self._action_step += 1
                return cmd
            else:
                # Tangent achieved - move to phase 3
                print(f"[Circle] Phase 2: Tangent aligned, starting orbit")
                self._goto_phase = 3
                # Initialize orbit tracking - use robot position angle in room frame
                self._orbit_accumulated = 0.0
                self._orbit_last_angle = np.arctan2(est_y, est_x)  # Robot's angle around room center

        # PHASE 3: Orbit around center (360°) with proportional control on radius and tangent
        if self._goto_phase == 3:
            target_radius = radius
            target_arc = 2 * np.pi  # 360°

            # Current angle to center (from robot perspective)
            current_angle_to_center = angle_to_center

            # Calculate how far we've orbited using robot position in ROOM frame
            # This measures the polar angle of the robot around the room center
            robot_polar_angle = np.arctan2(est_y, est_x)

            # Update accumulated angle
            if hasattr(self, '_orbit_last_angle'):
                delta_angle = robot_polar_angle - self._orbit_last_angle
                # Handle wrap-around
                while delta_angle > np.pi:
                    delta_angle -= 2 * np.pi
                while delta_angle < -np.pi:
                    delta_angle += 2 * np.pi
                self._orbit_accumulated += delta_angle
            self._orbit_last_angle = robot_polar_angle

            # Check if orbit complete
            if abs(self._orbit_accumulated) >= target_arc:
                print(f"[Circle] Phase 3: Orbit complete ({np.degrees(self._orbit_accumulated):.0f}°)")
                self.omnirobot_proxy.setSpeedBase(0, 0, 0)
                self._goto_phase = 0
                self._current_action_idx += 1
                self._action_step = 0
                return (0, 0, 0)

            # Proportional controllers:
            # 1. Radius error: we want dist_to_center == target_radius
            radius_error = dist_to_center - target_radius

            # 2. Tangent error: we want center at -90° (on our right)
            target_tangent_angle = -np.pi / 2
            tangent_error = current_angle_to_center - target_tangent_angle
            while tangent_error > np.pi:
                tangent_error -= 2 * np.pi
            while tangent_error < -np.pi:
                tangent_error += 2 * np.pi

            # Control gains
            Kp_radius = 0.3   # Lateral correction gain
            Kp_tangent = 1.0  # Rotation correction gain

            # Base forward speed (CCW orbit means moving forward while center is on the right)
            base_speed = self.advance_speed  # mm/s

            # Lateral correction: if too far (radius_error > 0), move left (toward center)
            lateral_correction = -Kp_radius * radius_error * 1000  # Convert to mm/s

            # Rotation correction: maintain tangent angle
            rotation_correction = Kp_tangent * tangent_error

            # Combine commands
            advx = np.clip(lateral_correction, -100, 100)  # Side speed in mm/s
            advz = base_speed  # Forward speed in mm/s
            rot = np.clip(rotation_correction, -self.rotation_speed, self.rotation_speed)


            cmd = (advx, advz, rot)
            cmd = self._apply_speed_modulation(cmd)
            self.omnirobot_proxy.setSpeedBase(*cmd)
            self._action_step += 1
            return cmd

        return (0, 0, 0)


    ##########################################################################################
    def startup_check(self):
        """Test interface types and quit (used for testing only)"""
        test = ifaces.RoboCompLidar3D.TPoint()
        test = ifaces.RoboCompLidar3D.TDataImage()
        test = ifaces.RoboCompLidar3D.TData()
        test = ifaces.RoboCompLidar3D.TDataCategory()
        test = ifaces.RoboCompLidar3D.TColorCloudData()
        test = ifaces.RoboCompOmniRobot.TMechParams()
        QTimer.singleShot(200, QApplication.instance().quit)


    ######################
    # From the RoboCompLidar3D you can call this methods:
    # RoboCompLidar3D.TColorCloudData self.lidar3d_proxy.getColorCloudData()
    # RoboCompLidar3D.TData self.lidar3d_proxy.getLidarData(str name, float start, float len, int decimationDegreeFactor)
    # RoboCompLidar3D.TDataImage self.lidar3d_proxy.getLidarDataArrayProyectedInImage(str name)
    # RoboCompLidar3D.TDataCategory self.lidar3d_proxy.getLidarDataByCategory(TCategories categories, long timestamp)
    # RoboCompLidar3D.TData self.lidar3d_proxy.getLidarDataProyectedInImage(str name)
    # RoboCompLidar3D.TData self.lidar3d_proxy.getLidarDataWithThreshold2d(str name, float distance, int decimationDegreeFactor)

    ######################
    # From the RoboCompLidar3D you can use this types:
    # ifaces.RoboCompLidar3D.TPoint
    # ifaces.RoboCompLidar3D.TDataImage
    # ifaces.RoboCompLidar3D.TData
    # ifaces.RoboCompLidar3D.TDataCategory
    # ifaces.RoboCompLidar3D.TColorCloudData

    ######################
    # From the RoboCompOmniRobot you can call this methods:
    # RoboCompOmniRobot.void self.omnirobot_proxy.correctOdometer(int x, int z, float alpha)
    # RoboCompOmniRobot.void self.omnirobot_proxy.getBasePose(int x, int z, float alpha)
    # RoboCompOmniRobot.void self.omnirobot_proxy.getBaseState(RoboCompGenericBase.TBaseState state)
    # RoboCompOmniRobot.void self.omnirobot_proxy.resetOdometer()
    # RoboCompOmniRobot.void self.omnirobot_proxy.setOdometer(RoboCompGenericBase.TBaseState state)
    # RoboCompOmniRobot.void self.omnirobot_proxy.setOdometerPose(int x, int z, float alpha)
    # RoboCompOmniRobot.void self.omnirobot_proxy.setSpeedBase(float advx, float advz, float rot)
    # RoboCompOmniRobot.void self.omnirobot_proxy.stopBase()

    ######################
    # From the RoboCompOmniRobot you can use this types:
    # ifaces.RoboCompOmniRobot.TMechParams


    #
    # SUBSCRIPTION to newFullPose method from FullPoseEstimationPub interface
    #
    def FullPoseEstimationPub_newFullPose(self, pose):
        """
        Receive ground truth pose from simulator (for evaluation only).

        IMPORTANT: Ground truth is used ONLY for:
        1. Performance evaluation (computing pose error vs estimate)
        2. Initial approximation during INIT phase (weak prior)

        The Active Inference estimator does NOT use ground truth during
        TRACKING phase - it relies solely on:
        - Motion model prediction: s_pred = f(s_prev, u)
        - LIDAR observations: o = {pⱼ}

        This separation is critical for demonstrating that the system can
        operate without external localization (GPS, motion capture, etc.).

        Coordinate transform:
            Simulator frame → Room frame by negating (x, y, θ)

        Args:
            pose: FullPose message with (x, y, rz) from simulator
        """
        normalized_theta = np.arctan2(np.sin(-pose.rz), np.cos(-pose.rz))
        x_transformed = -pose.x
        y_transformed = -pose.y


        self.robot_pose_gt = [x_transformed, y_transformed, normalized_theta]


    # =============== DSR GRAPH MANAGEMENT  ================
    # =====================================================
    #
    # The DSR (Deep State Representation) graph stores the agent's beliefs
    # about the world in a distributed knowledge graph. This implements the
    # "generative model" structure described in main.tex Section 5.1.
    #
    # Graph structure for room estimation:
    #   room (node) ──[RT edge]──> robot (node)
    #
    # The RT edge encodes the robot's pose belief q(s) = N(μ, Σ):
    #   - rt_translation: μ_{x,y} (position mean)
    #   - rt_rotation_euler_xyz: μ_θ (orientation mean)
    #   - rt_se2_covariance: Σ (3x3 pose covariance matrix)
    #
    # =====================================================

    def insert_room_node(self):
        """
        Insert room node into DSR graph when room estimation converges.

        This method is called once during the transition from INIT to TRACKING phase,
        when the room dimensions (W, L) have been estimated with sufficient confidence.

        The room node represents the inferred room geometry and serves as the
        reference frame for robot localization. The RT edge room→robot encodes
        the posterior belief q(s) = N(μ, Σ) over robot pose.

        Graph structure created:
            room ──[RT]──> robot

        Attributes stored:
            - room_width, room_length: Estimated room dimensions (mm)
            - RT translation: Robot position μ_{x,y} (mm)
            - RT rotation: Robot heading μ_θ (rad)
            - RT covariance: Pose uncertainty Σ (3x3 matrix)

        References:
            main.tex Sec. 5.1: State Space and Generative Model
            main.tex Fig. 5: Reference frames
        """
        belief = self.room_estimator.belief
        if belief is None:
            return

        # Create room node
        room_node = Node(agent_id=self.agent_id, type="room", name="room")
        room_node.attrs["color"] = Attribute("GreenYellow", self.agent_id)
        room_node.attrs["level"] = Attribute(0, self.agent_id)
        room_node.attrs["pos_x"] = Attribute(float(0), self.agent_id)
        room_node.attrs["pos_y"] = Attribute(float(0), self.agent_id)
        room_node.attrs["room_width"] = Attribute(float(belief.width * 1000), self.agent_id)   # mm
        room_node.attrs["room_length"] = Attribute(float(belief.length * 1000), self.agent_id)  # mm

        new_id = self.g.insert_node(room_node)
        if new_id is not None:
            self.room_node_id = new_id
            self.room_node_inserted = True
            print(f"[DSR] Room node inserted with id={new_id}, "
                  f"size={belief.width:.2f}x{belief.length:.2f}m")

            # Update robot node: set parent to room
            robot_node = self.g.get_node("robot")
            if robot_node:
                robot_node.attrs["parent"] = Attribute(int(new_id), self.agent_id)
                robot_node.attrs["level"] = Attribute(1, self.agent_id)
                self.g.update_node(robot_node)

            # Create RT edge room → robot with current estimated pose
            self._update_rt_edge(belief)
        else:
            print("[DSR] ERROR: Failed to insert room node")

    def _update_rt_edge(self, belief):
        """
        Update RT edge with current pose belief q(s) = N(μ, Σ).

        The RT (Rigid Transform) edge encodes the spatial relationship between
        room and robot frames. In Active Inference terms, this represents the
        current posterior belief over robot pose after free energy minimization.

        BELIEF REPRESENTATION:
        ======================
        The Laplace approximation (main.tex Sec. 5.2.3) yields a Gaussian posterior:

            q(s) = N(s | μ, Σ)

        where:
            μ = [x, y, θ]ᵀ       - posterior mean (MAP estimate)
            Σ = H⁻¹ · σ²         - posterior covariance (from Hessian)

        The covariance Σ encodes uncertainty and is computed via:

            Σ_post = σ² · H⁻¹

        where H is the Hessian of the Free Energy at the optimum and σ² is
        the residual variance (SDF error).

        Args:
            belief: RoomBelief object containing pose mean and covariance

        Edge attributes:
            - rt_translation: [x, y, 0] in mm
            - rt_rotation_euler_xyz: [0, 0, θ] in radians
            - rt_se2_covariance: flattened 3x3 covariance matrix

        References:
            main.tex Sec. 5.2.3: Laplace Approximation
            ACTIVE_INFERENCE_MATH.md Sec. 4: Posterior Covariance
        """
        robot_node = self.g.get_node("robot")
        if robot_node is None:
            return

        # Translation: belief pose is in meters, DSR uses mm
        translation = [float(belief.x * 1000),
                       float(belief.y * 1000),
                       0.0]
        rotation = [0.0, 0.0, float(belief.theta)]

        # SE(2) covariance [x, y, θ] as flattened 3x3 → 9 floats
        cov = belief.pose_cov.flatten().tolist()

        rt_edge = Edge(robot_node.id, self.room_node_id, "RT", self.agent_id)
        rt_edge.attrs["rt_translation"] = Attribute(translation, self.agent_id)
        rt_edge.attrs["rt_rotation_euler_xyz"] = Attribute(rotation, self.agent_id)
        rt_edge.attrs["rt_se2_covariance"] = Attribute(cov, self.agent_id)

        self.g.insert_or_assign_edge(rt_edge)

    def update_graph_from_estimator(self):
        """
        Synchronize DSR graph with current estimator beliefs.

        This method propagates the posterior belief q(s) from the Active Inference
        estimator to the DSR knowledge graph, making the robot's internal model
        available to other agents and components in the system.

        TEMPORAL DYNAMICS:
        ==================
        The graph update follows the perception-action cycle (main.tex Sec. 3.2):

        1. INIT phase: No graph updates (beliefs not yet reliable)
        2. INIT→TRACKING transition: Insert room node with estimated (W, L)
        3. TRACKING phase: Update RT edge with current pose belief (x, y, θ, Σ)

        This ensures the graph always reflects the agent's best current estimate
        of the world state, enabling other components to:
        - Plan paths using the room geometry
        - Reason about robot position uncertainty
        - Detect and track obstacles relative to the room frame

        References:
            main.tex Sec. 5.1: Robot and Environment State Spaces
        """
        belief = self.room_estimator.belief
        if belief is None:
            return

        if self.room_estimator.phase == 'tracking' and not self.room_node_inserted:
            # Room just converged — insert room node + first RT edge
            self.insert_room_node()
        elif self.room_node_inserted:
            # Update RT edge with latest pose
            self._update_rt_edge(belief)

    def print_graph(self):
        """Read and print the entire DSR graph: all nodes with their attributes,
        and for each node all its edges with their attributes."""
        nodes = self.g.get_nodes()
        if not nodes:
            console.print("[bold red]Graph is empty[/bold red]")
            return

        console.print(f"\n[bold cyan]===== DSR Graph ({len(nodes)} nodes) =====[/bold cyan]")

        for node in nodes:
            # Print node header
            console.print(f"\n[bold yellow]Node[/bold yellow] id={node.id}  name='{node.name}'  type='{node.type}'")

            # Print node attributes
            if node.attrs:
                for attr_name, attr in node.attrs.items():
                    console.print(f"  [green]attr[/green] {attr_name} = {attr.value}")

            # Print edges from this node
            if node.edges:
                for edge_key, edge in node.edges.items():
                    console.print(f"  [magenta]edge[/magenta] ({edge.origin}) --[{edge.type}]--> ({edge.destination})")
                    # Print edge attributes
                    if edge.attrs:
                        for attr_name, attr in edge.attrs.items():
                            console.print(f"    [dim]edge attr[/dim] {attr_name} = {attr.value}")

        console.print(f"\n[bold cyan]===== End of Graph =====[/bold cyan]\n")

    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        pass

    def update_node(self, id: int, type: str):
        pass

    def delete_node(self, id: int):
        pass

    def update_edge(self, fr: int, to: int, type: str):
        pass

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        pass

    def delete_edge(self, fr: int, to: int, type: str):
        pass
