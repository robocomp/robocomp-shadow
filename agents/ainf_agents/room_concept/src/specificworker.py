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
            # console.print("signals connected")
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
            self.dsr_viewer = DSRGraphViewerDPG(
                g=self.g,
                window_width=380,    # Width of DSR panel
                window_height=560,   # Height adjusted for margins
                update_period_ms=500,
                canvas_tag="dsr_canvas"
            )
            print("[SpecificWorker] DSR graph viewer created")

            # Create room viewer with integrated DSR viewer
            self.room_viewer = RoomViewerDPG(
                window_width=1400,   # Wider to accommodate DSR panel + stats
                window_height=800,
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

            # Adaptive LIDAR subsampling (passed to proxy call)
            # 1 = no subsampling, 2 = one of two, 4 = one of four, etc.
            self._lidar_subsample_factor = 4  # Default value
            self._adaptive_lidar_enabled = True  # Enable adaptive subsampling

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
        plan_names = ["Basic (Static)", "Circle (0.75m + 360°)", "Ocho (Figure-8)"]
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

                print(f"[Step {self.total_steps}] Phase: {phase}, SDF: {sdf_err:.3f}, "
                      f"Pose_err: {pose_error:.3f}m, Angle_err: {np.degrees(abs(angle_error)):.1f}°, Uncertainty: {uncertainty:.3f}")

                # Show current action being executed
                current_action = self._get_current_action_description()
                print(f"  Action: {current_action}")

                # Show innovation (prediction error) and prior precision if available
                if phase == 'tracking' and 'innovation' in result:
                    innov = result['innovation']
                    pp = result.get('prior_precision', 0.0)
                    print(f"  Innovation: dx={innov[0]*100:.1f}cm, dy={innov[1]*100:.1f}cm, dθ={np.degrees(innov[2]):.1f}°, Precision: {pp:.3f}")

                # Ground truth (from simulator)
                print(f"  Ground truth: x={gt_x:.3f}m, y={gt_y:.3f}m, theta={gt_theta:.3f}rad ({np.degrees(gt_theta):.1f}°)")

                # Estimated pose (from Active Inference)
                print(f"  Estimated:    x={est_x:.3f}m, y={est_y:.3f}m, theta={est_theta:.3f}rad ({np.degrees(est_theta):.1f}°)")

                # Room estimation
                if self.room_estimator.belief:
                    print(f"  Room: {self.room_estimator.belief.width:.2f} x "
                          f"{self.room_estimator.belief.length:.2f}m (true: {self.room_estimator.true_width:.1f} x {self.room_estimator.true_height:.1f}m)")

            # Update viewer with current data
            # Extract innovation and prior precision from result if available
            innovation = result.get('innovation', None)
            prior_precision = result.get('prior_precision', 0.0)
            optimizer_iterations = result.get('iterations_used', 0)
            velocity_weights = result.get('velocity_weights', None)

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
                velocity_weights=velocity_weights
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
        plan_names = ["Basic (Static)", "Circle (0.75m + 360°)", "Ocho (Figure-8)"]
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
            # Determine adaptive subsampling factor
            if self._adaptive_lidar_enabled and hasattr(self, 'room_estimator'):
                last_sdf = getattr(self.room_estimator, '_last_sdf_error', float('inf'))
                stability_threshold = getattr(self.room_estimator, 'stability_threshold', 0.02)

                if last_sdf < stability_threshold:
                    # Very stable - use high subsampling (fewer points)
                    self._lidar_subsample_factor = 8
                elif last_sdf < stability_threshold * 2:
                    # Moderately stable
                    self._lidar_subsample_factor = 4
                elif last_sdf < stability_threshold * 4:
                    # Slightly unstable
                    self._lidar_subsample_factor = 2
                else:
                    # Unstable - use all points
                    self._lidar_subsample_factor = 1

            # Get 2D LIDAR data with adaptive subsampling
            # decimationDegreeFactor: 1 = no subsampling, 2 = one of two, etc.
            helios = self.lidar3d_proxy.getLidarDataWithThreshold2d(
                "helios", 8000, self._lidar_subsample_factor
            )
            # LIDAR points in robot frame: p.x = right, p.y = forward
            lidar_points = np.array([[p.x / 1000.0, p.y / 1000.0] for p in helios.points if p.z > 1000])


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
        return 0

    ##########################################################################################
    def startup_check(self):
        print(f"Testing RoboCompLidar3D.TPoint from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TPoint()
        print(f"Testing RoboCompLidar3D.TDataImage from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TDataImage()
        print(f"Testing RoboCompLidar3D.TData from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TData()
        print(f"Testing RoboCompLidar3D.TDataCategory from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TDataCategory()
        print(f"Testing RoboCompLidar3D.TColorCloudData from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TColorCloudData()
        print(f"Testing RoboCompOmniRobot.TMechParams from ifaces.RoboCompOmniRobot")
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
