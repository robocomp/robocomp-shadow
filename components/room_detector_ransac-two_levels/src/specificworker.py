#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2025 by YOUR NAME HERE
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

import os, sys, typing
import time
from copy import deepcopy

import torch

sys.path.insert(0, os.path.dirname(__file__))
import compat_typing_self

if sys.version_info < (3, 11):
    has_self = hasattr(typing, "Self")
    print(f"[compat] Python {sys.version.split()[0]} | typing.Self present: {has_self}")
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import open3d as o3d
from plane_detector import PlaneDetector
from room_particle_filter import RoomParticleFilter, Particle
import numpy as np
from collections import deque
import subprocess, tempfile, json
from regional_loss import RegionalizedRectLoss
from open3d_hotzones import build_hot_patches_heatmap
from params import AppParams

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, configData, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map, configData)
        # one place to configure everything
        self.params = AppParams()

        self.Period = configData["Period"]["Compute"]
        self.hide()
        if startup_check:
            self.startup_check()
        else:

            # --- Plane detector (from params) ---
            pp = self.params.plane
            self.plane_detector = PlaneDetector(
                voxel_size=pp.voxel_size,
                angle_tolerance_deg=pp.angle_tolerance_deg,
                ransac_threshold=pp.ransac_threshold,
                ransac_n=pp.ransac_n,
                ransac_iterations=pp.ransac_iterations,
                min_plane_points=pp.min_plane_points,
                nms_normal_dot_threshold=pp.nms_normal_dot_threshold,
                nms_distance_threshold=pp.nms_distance_threshold,
                plane_thickness=pp.plane_thickness
            )  # :contentReference[oaicite:0]{index=0}

            # --- Initial hypothesis -> Particle ---
            H = self.params.hypothesis
            ground_truth_particle = Particle(
                x=H.x, y=H.y, theta=H.theta,
                length=H.length, width=H.width,
                height=H.height, z_center=H.z_center, weight=H.weight
            )  # :contentReference[oaicite:1]{index=1}

            # --- Particle Filter (constructed + tuned from params) ---
            pfp = self.params.pf
            self.particle_filter = RoomParticleFilter(
                num_particles=pfp.num_particles,
                initial_hypothesis=ground_truth_particle,
                device=pfp.device,
                use_gradient_refinement=pfp.use_gradient_refinement,
                adaptive_particles=pfp.adaptive_particles,
                min_particles=pfp.min_particles,
                max_particles=pfp.max_particles,
                elite_count=pfp.elite_count,
                trans_noise=pfp.trans_noise,
                rot_noise=pfp.rot_noise,
                trans_noise_stationary=pfp.trans_noise_stationary,
                rot_noise_stationary=pfp.rot_noise_stationary,
                ess_frac=pfp.ess_frac,
                lr=pfp.lr,
                num_steps=pfp.num_steps,
                top_n=pfp.top_n,
                pose_lambda=pfp.pose_lambda,
                size_lambda=pfp.size_lambda,
            )

            self.current_best_particle = None
            self.current_best_smoothed_particle = None
            self.room_box_height = H.height

            # --- Open3D Visualizer from params ---
            V = self.params.viz
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(width=V.window_size[0], height=V.window_size[1])
            self.view_control = self.vis.get_view_control()
            self.view_control.set_front(list(V.view_front))
            self.view_control.set_lookat(list(V.lookat))
            self.view_control.set_up(list(V.up))
            self.view_control.set_zoom(V.zoom)
            self.camera_parameters = self.view_control.convert_to_pinhole_camera_parameters()

            # colors as numpy arrays
            import numpy as np
            self.h_color = np.array(V.h_color, dtype=np.float64)
            self.v_colors = [np.array(c, dtype=np.float64) for c in V.v_colors]
            self.o_color = np.array(V.o_color, dtype=np.float64)

            # robot state
            self.robot_velocity = [0.0, 0.0, 0.0]
            self.best_loss = 1.0

            from collections import deque
            self.cmd_buffer = deque(maxlen=300)
            now = time.monotonic()
            self.cmd_buffer.append((now, 0.0, 0.0, 0.0))

            # PF tick timing
            self.last_tick_time = time.monotonic()
            self.last_pred_time = time.monotonic()
            self.cycle_time = 0.1

            # joystick smoothing
            self.vel_alpha = self.params.timing.vel_alpha

            # Load Shadow mesh
            self.shadow_mesh = o3d.io.read_triangle_mesh("src/meshes/shadow.obj", print_progress=True)
            self.shadow_mesh.paint_uniform_color([1, 0, 1])
            self.vis.add_geometry(self.shadow_mesh)

            # plotter
            import tempfile, subprocess, json, os, sys
            self.plot_history_path = os.path.join(tempfile.gettempdir(), "pf_history.json")
            plotter_py = os.path.join(os.path.dirname(__file__), "plotter.py")
            self._plotter_proc = subprocess.Popen([sys.executable, "-u", plotter_py, self.plot_history_path],
                                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                                  start_new_session=True)  # :contentReference[oaicite:4]{index=4}

            # --- Regionalized loss (segmentation) from params ---
            RL = self.params.regional
            if not hasattr(self, "seg_loss"):
                self.seg_loss = RegionalizedRectLoss(
                    num_segments_per_side=RL.num_segments_per_side,
                    band_outside=RL.band_outside,
                    band_inside=RL.band_inside,
                    huber_delta=RL.huber_delta,
                    device=RL.device,
                    absence_alpha=RL.absence_alpha,
                    absence_curve_k=RL.absence_curve_k
                )  # :contentReference[oaicite:5]{index=5}

            # Period from params (keep configData if you prefer)
            self.Period = self.params.timing.period_ms
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""
        try:
            if hasattr(self, "_plotter_proc") and self._plotter_proc and self._plotter_proc.poll() is None:
                self._plotter_proc.terminate()
        except Exception:
            pass

    @QtCore.Slot()
    def compute(self):

        # Measure cycle time
        now = time.monotonic()
        t0, t1 = self.last_pred_time, now
        self.last_pred_time = now
        cycle = now - getattr(self, "last_tick_time", now)
        self.cycle_time = 0.9 * getattr(self, "cycle_time", cycle) + 0.1 * cycle

        # read lidar data
        pcd = o3d.geometry.PointCloud()
        if not (res := self.read_lidar_data()):
            return False
        pcd.points = res

        # Adaptive plane detection based on convergence
        if self.best_loss < 0.001:  # Fully converged → Maximum efficiency
            h_planes, validated_v, unmatched_v, o_planes, outliers = \
                self.plane_detector.detect_with_bayesian_filtering(
                    pcd,
                    self.current_best_particle,
                    wall_band_width=0.15,  # Points within 15cm of wall
                    refine_alpha=0.8
                )
        elif self.best_loss < 0.01:  # Converging → Balanced
            h_planes, validated_v, unmatched_v, o_planes, outliers = \
                self.plane_detector.detect_with_prior_seeded(
                    pcd, self.current_best_particle
                )
        else:  # Exploring → Flexible
            h_planes, validated_v, unmatched_v, o_planes, outliers = \
                self.plane_detector.detect_with_prior(
                    pcd, self.current_best_particle
                )

        # if self.best_loss < self.params.pf.min_loss_for_locking_ransac:  # Converged. Use seeded RANSAC for rock-solid stability
        #       h_planes, validated_v, unmatched_v, o_planes, outliers = \
        #           self.plane_detector.detect_with_prior_seeded(pcd, RoomParticleFilter.particle_to_pose(self.current_best_particle)) #dependency inversion
        # else:  # Still exploring.  # Use normal validation for flexibility
        #      h_planes, validated_v, unmatched_v, o_planes, outliers = \
        #          self.plane_detector.detect_with_prior(pcd, RoomParticleFilter.particle_to_pose(self.current_best_particle))

        # if no unmatched vertical planes and outliers are few, skip this frame
        # if (not unmatched_v) and (len(outliers) < self.params.plane.min_plane_points):
        #     console.log("[yellow]Skipping PF update: no new vertical planes detected.[/yellow]")
        #     self.visualize_results(self.current_best_particle, pcd, h_planes,
        #                            validated_v + unmatched_v, o_planes, outliers,
        #                            torch.from_numpy(self._extract_wall_points(pcd, validated_v)[:, :2]).float().to('cuda'))
        #     self.send_data_to_plotter()
        #     self.last_tick_time = now
        #     return True


        # Use weighted extraction ===
        wall_points_np = self._extract_wall_points_weighted(
                                pcd, validated_v, unmatched_v,
                                validated_weight=0.8,  # 80% from stabilized planes
                                unmatched_weight=0.2  # 20% from novel/obstacle planes
                            )
        wall_points_torch = torch.from_numpy(wall_points_np[:, :2]).float().to(self.particle_filter.device)

        # Estimate latency and integrate commands
        tau = min(0.20, max(0.0, self.cycle_time))   # cap at 200ms?
        dx, dy, dtheta = self.integrate_cmds_with_latency(t0, t1, self.params.timing.latency_gain * tau)
        odometry_delta = (dx, dy, dtheta)

        # Run particle filter step with pre-computed wall points
        self.best_loss, self.current_best_particle, smoothed_particle = self.particle_filter.step(odometry_delta, wall_points_torch, self.cycle_time)

        # Visualize results (combine for visualization)
        v_planes = validated_v + unmatched_v
        self.visualize_results(smoothed_particle, pcd, h_planes, v_planes, o_planes, outliers, wall_points_torch)

        # Send data to plotter
        self.send_data_to_plotter()

        # move robot under model contraints
        self.move_robot()

        self.last_tick_time = now
        return True


    ############################# MY CODE HERE #############################
    def move_robot(self):
        '''
            move robot under model constraints
            if loss is high, reduce speed to avoid going out of bounds
        '''

        loss = self.best_loss
        speed_factor = max(0.0, min(1.0, 1.0 - loss * 100.0))  # linearly reduce speed for loss in [0.0, 0.1]
        self.robot_velocity[0] *= speed_factor
        self.robot_velocity[1] *= speed_factor
        self.robot_velocity[2] *= speed_factor
        # print velocity with two decimals before and after speed factor correction
        console.log(f"[green]Sending speed command: vx={self.robot_velocity[0]:.2f} m/s, vy={self.robot_velocity[1]:.2f} m/s, omega={self.robot_velocity[2]:.2f} rad/s (speed factor: {speed_factor:.2f})[/green]")

        # send command to robot (convert m/s to mm/s)
        try:
            self.omnirobot_proxy.setSpeedBase(self.robot_velocity[0] * 1000.0, self.robot_velocity[1] * 1000.0, self.robot_velocity[2])
        except Exception as e:
            console.log(f"[red]Error sending speed command: {e}[/red]")

    def _extract_wall_points(self, pcd, v_planes):
        """Extract wall inlier points from vertical planes (with fallback to all points)."""
        if not v_planes:
            return np.asarray(pcd.points)

        all_indices = []
        for _, indices in v_planes:
            all_indices.extend(indices)

        if not all_indices:
            return np.asarray(pcd.points)

        points = np.asarray(pcd.points)
        return points[all_indices]

    def read_lidar_data(self):
        lidar_data = self.lidar3d_proxy.getLidarDataWithThreshold2d("helios", 10000, 3)
        if not lidar_data.points or len(lidar_data.points) == 0:
            console.log("[yellow]No lidar data received.[/yellow]")
            return False

        # Convert to open3d format filtering out those with z lower than 1.5m
        points_list = [[p.x / 1000.0, p.y / 1000.0, p.z / 1000.0] for p in lidar_data.points if p.z >= 1500.0]
        return o3d.utility.Vector3dVector(points_list)

    def send_data_to_plotter(self):
        N = 5
        self._tick_counter = getattr(self, "_tick_counter", 0) + 1
        if self._tick_counter % N == 0:
            try:
                hist = self.particle_filter.get_history()
                N = 600  # last N points
                payload = {
                    "tick": hist.get("tick", [])[-N:],
                    "loss_best": hist.get("loss_best", [])[-N:],
                    "num_features": hist.get("num_features", [])[-N:],
                    "ess": hist.get("ess", [])[-N:],
                    "births": hist.get("births", [])[-N:],
                    "deaths": hist.get("deaths", [])[-N:],
                    "n_particles": hist.get("n_particles", [])[-N:],
                    "ess_pct": hist.get("ess_pct", [])[-N:],
                    "weight_entropy": hist.get("weight_entropy", [])[-N:],
                    "x_std": hist.get("x_std", [])[-N:],
                    "y_std": hist.get("y_std", [])[-N:],
                    "theta_std": hist.get("theta_std", [])[-N:],
                    "period": hist.get("period", [])[-N:],
                }
                tmp = self.plot_history_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(payload, f)
                os.replace(tmp, self.plot_history_path)  # atomic swap
            except Exception:
                pass

    def visualize_results(self, particle, pcd, h_planes, v_planes, o_planes, outliers, wall_points_torch):
        """Create and display all visualization geometries (room-centric frame)"""
        if particle is None:
            return

        geometries_to_draw = []

        # Get transform to room-centric frame
        #p = self.particle_filter.mean_weighted_particle()
        p = particle

        # Create inverse transform (world -> room frame)
        # Room is at origin, so we move everything by -particle pose
        c, s = np.cos(-p.theta), np.sin(-p.theta)
        R_inv = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        translation = np.array([-p.x, -p.y, 0])

        # Transform point cloud to room frame
        pcd_room = o3d.geometry.PointCloud(pcd)
        pcd_room.translate(translation)
        pcd_room.rotate(R_inv, center=(0, 0, 0))

        # Create plane visualizations (in room frame)
        all_planes = [(h_planes, self.h_color), (v_planes, self.v_colors), (o_planes, self.o_color)]
        for i, (plane_list, color_map) in enumerate(all_planes):
            for j, (model, indices) in enumerate(plane_list):
                inlier_pcd = pcd_room.select_by_index(indices)
                if len(inlier_pcd.points) < 3:
                    continue

                obb = inlier_pcd.get_oriented_bounding_box()
                new_extent = np.array(obb.extent)
                new_extent[np.argmin(new_extent)] = self.plane_detector.plane_thickness
                obb.extent = new_extent

                if i == 1:  # Vertical planes - cycle through colors
                    color = np.asarray(color_map[j % len(color_map)], dtype=np.float64)
                else:
                    color = np.asarray(color_map, dtype=np.float64)
                obb.color = color
                geometries_to_draw.append(obb)

        # Create room box visualization (at origin in room frame)
        center = [0, 0, p.z_center]
        R = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, 0])
        extent = [p.length, p.width, p.height]
        room_box = o3d.geometry.OrientedBoundingBox(center, R, extent)
        room_box_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(room_box)
        room_box_lines.paint_uniform_color(np.array([1.0, 0.0, 1.0], dtype=np.float64))
        geometries_to_draw.append(room_box_lines)

        # Transform robot mesh to room frame
        shadow_mesh_transformed = o3d.geometry.TriangleMesh(self.shadow_mesh)
        shadow_mesh_transformed.translate(translation)
        shadow_mesh_transformed.rotate(R_inv, center=(0, 0, 0))
        geometries_to_draw.append(shadow_mesh_transformed)

        # floor in room frame
        floor_mesh = o3d.geometry.TriangleMesh.create_box(width=p.length,
                                                          height=p.width,
                                                          depth=self.params.viz.floor_thickness)
        floor_mesh.translate([-p.length / 2, -p.width / 2, 0.0])
        floor_mesh.paint_uniform_color([1.0, 0.86, 0.58])
        geometries_to_draw.append(floor_mesh)

        ### Draw hotzones on lines (in room frame):
        seg_list, heat = self.seg_loss.evaluate_segments(p, wall_points_torch[:, :2])
        HZ = self.params.hotzones
        hotzone_meshes = build_hot_patches_heatmap(
            seg_list, p,
            threshold=None,
            topk=HZ.topk,
            min_norm=HZ.min_norm,
            min_points=0,
            min_support_ratio=HZ.min_support_ratio,
            percentile_clip=HZ.percentile_clip,
            # color range overrides not used by default
            vmin_override=None,
            vmax_override=None,
            # geometry
            thickness=HZ.thickness,
            outward_eps=0.0,
            height=HZ.height,
            lift=HZ.lift,
            inside_frac=HZ.inside_frac,
            gap_in=HZ.gap_in,
            gap_out=HZ.gap_out,
            # snapping
            snap_mode=HZ.snap_mode,
            eps_in=HZ.eps_in,
            eps_out=HZ.eps_out,
        )

        for m in hotzone_meshes:
            geometries_to_draw.append(m)

        # --- room_map from PF (L, W, features) ---
        # room_map = self.particle_filter.get_map()
        # L, W = room_map.L, room_map.W
        #
        # # draw features (in room frame, no transforms)
        # for feat in room_map.features:
        #     fmesh, flines = self._make_feature_mesh(L, W, feat, z0=0.0, thickness=0.02)
        #     geometries_to_draw.append(fmesh)
        #     geometries_to_draw.append(flines)

        # Create outlier visualization (in room frame)
        # outlier_cloud = pcd_room.select_by_index(outliers)
        # outlier_cloud.paint_uniform_color(np.array([0.5, 0.5, 0.5], dtype=np.float64))
        # geometries_to_draw.append(outlier_cloud)

        # Update visualizer
        self.vis.clear_geometries()
        for geom in geometries_to_draw:
            self.vis.add_geometry(geom)

        if self.camera_parameters is not None:
            self.view_control.convert_from_pinhole_camera_parameters(self.camera_parameters)

        self.vis.poll_events()
        self.camera_parameters = self.view_control.convert_to_pinhole_camera_parameters()
        self.vis.update_renderer()

    def _extract_wall_points_weighted(self, pcd, validated_planes, unmatched_planes,
                                      validated_weight=0.8, unmatched_weight=0.2):
        """
        Extract wall points with weighted sampling from validated and unmatched planes.

        Args:
            pcd: Point cloud
            validated_planes: List of (model, indices) for validated (stabilized) planes
            unmatched_planes: List of (model, indices) for unmatched (novel) planes
            validated_weight: Proportion of points from validated planes (0-1)
            unmatched_weight: Proportion of points from unmatched planes (0-1)

        Returns:
            numpy array of wall points (N, 3)
        """
        import numpy as np

        # Extract points from validated planes
        validated_points = self._extract_wall_points(pcd, validated_planes)

        # Extract points from unmatched planes
        unmatched_points = self._extract_wall_points(pcd, unmatched_planes)

        # Handle empty cases
        if len(validated_points) == 0 and len(unmatched_points) == 0:
            # Fallback to all points if no planes detected
            return np.asarray(pcd.points)

        if len(validated_points) == 0:
            # Only unmatched available
            return unmatched_points

        if len(unmatched_points) == 0:
            # Only validated available
            return validated_points

        # Determine target sample sizes
        total_target = min(len(validated_points) + len(unmatched_points), 10000)  # Cap at 10k points

        n_validated_target = int(validated_weight * total_target)
        n_unmatched_target = int(unmatched_weight * total_target)

        # Adjust if we don't have enough points
        n_validated = min(n_validated_target, len(validated_points))
        n_unmatched = min(n_unmatched_target, len(unmatched_points))

        # Sample from validated planes
        if n_validated > 0 and n_validated < len(validated_points):
            validated_indices = np.random.choice(len(validated_points), n_validated, replace=False)
            validated_sample = validated_points[validated_indices]
        else:
            validated_sample = validated_points

        # Sample from unmatched planes
        if n_unmatched > 0 and n_unmatched < len(unmatched_points):
            unmatched_indices = np.random.choice(len(unmatched_points), n_unmatched, replace=False)
            unmatched_sample = unmatched_points[unmatched_indices]
        else:
            unmatched_sample = unmatched_points

        # Combine
        wall_points = np.vstack([validated_sample, unmatched_sample])

        return wall_points

    def record_velocity_command(self, vx: float, vy: float, omega: float) -> None:
        """
        Call this whenever a new joystick command arrives.
        It updates self.robot_velocity and appends to the ring buffer with timestamp.
        """
        now = time.monotonic()
        # Optional low-pass
        vx = self.vel_alpha * vx + (1 - self.vel_alpha) * self.robot_velocity[0]
        vy = self.vel_alpha * vy + (1 - self.vel_alpha) * self.robot_velocity[1]
        omega = self.vel_alpha * omega + (1 - self.vel_alpha) * self.robot_velocity[2]
        self.robot_velocity[:] = [vx, vy, omega]
        self.cmd_buffer.append((now, vx, vy, omega))

    def integrate_cmds_with_latency(self, t0, t1, tau):
        start = t0 - tau
        end = t1 - tau

        # If the window is tiny or inverted, bail early
        if end - start <= 1e-6:
            return 0.0, 0.0, 0.0

        buf = list(self.cmd_buffer)
        if not buf:
            return 0.0, 0.0, 0.0

        # Ensure coverage until 'end' with last known command (stick held steady)
        t_last, vx_last, vy_last, om_last = buf[-1]
        if t_last < end:
            buf.append((end, vx_last, vy_last, om_last))

        dx = dy = dtheta = 0.0
        for i in range(len(buf) - 1):
            ts, vx, vy, om = buf[i]
            te, _, _, _ = buf[i + 1]

            # overlap with [start, end]
            s = max(ts, start)
            e = min(te, end)
            dt = e - s
            if dt <= 0:
                continue

            dx += vx * dt
            dy += vy * dt
            dtheta += om * dt

        return dx, dy, dtheta

    # --- Feature geometry helpers (room frame @ origin) ---

    def _feature_box_params(self, L, W, feat):
        """Return (cx, cy, sx, sy) for a feature box centered at (cx,cy) with sizes sx,sy."""
        if feat.wall == "+x":  # right wall at x=+L/2
            cx = L / 2 + (feat.depth / 2 if feat.kind == "extrusion" else -feat.depth / 2)
            cy = -W / 2 + (feat.t0 + feat.t1) / 2.0
            sx = feat.depth
            sy = (feat.t1 - feat.t0)
        elif feat.wall == "-x":  # left wall at x=-L/2
            cx = -L / 2 - (feat.depth / 2 if feat.kind == "extrusion" else -feat.depth / 2)
            cy = -W / 2 + (feat.t0 + feat.t1) / 2.0
            sx = feat.depth
            sy = (feat.t1 - feat.t0)
        elif feat.wall == "+y":  # top wall at y=+W/2
            cx = -L / 2 + (feat.t0 + feat.t1) / 2.0
            cy = W / 2 + (feat.depth / 2 if feat.kind == "extrusion" else -feat.depth / 2)
            sx = (feat.t1 - feat.t0)
            sy = feat.depth
        else:  # "-y": bottom wall at y=-W/2
            cx = -L / 2 + (feat.t0 + feat.t1) / 2.0
            cy = -W / 2 - (feat.depth / 2 if feat.kind == "extrusion" else -feat.depth / 2)
            sx = (feat.t1 - feat.t0)
            sy = feat.depth
        return cx, cy, sx, sy

    def _make_feature_mesh(self, L, W, feat, z0=0.0, thickness=0.02):
        """Open3D mesh + (optional) outline for a feature (in room frame)."""
        import open3d as o3d
        cx, cy, sx, sy = self._feature_box_params(L, W, feat)
        # Open3D boxes are anchored at a corner: shift to center
        mesh = o3d.geometry.TriangleMesh.create_box(width=sx, height=sy, depth=thickness)
        mesh.translate([cx - sx / 2.0, cy - sy / 2.0, z0])
        color = (0.10, 0.80, 0.10) if feat.kind == "extrusion" else (0.10, 0.80, 0.80)
        mesh.paint_uniform_color(color)

        # Thin outline for clarity (optional)
        # Four top corners (z=z0+thickness)
        x0, x1 = cx - sx / 2.0, cx + sx / 2.0
        y0, y1 = cy - sy / 2.0, cy + sy / 2.0
        z = z0 + thickness
        pts = [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]
        lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines)
        )
        ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
        return mesh, ls

    #######################################################################################

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

    # =============== Subscription Methods ================
    # SUBSCRIPTION to sendData method from JoystickAdapter interface

    def JoystickAdapter_sendData(self, data):
        # start from previous command so missing axes don't zero things
        vx, vy, omega = self.robot_velocity
        for a in data.axes:
            if a.name == "advance":
                vy = a.value / 1000.0  # m/s
            elif a.name == "side":
                vx = a.value / 1000.0  # m/s
            elif a.name == "rotate":
                omega = a.value  # rad/s
        self.record_velocity_command(vx, vy, omega)