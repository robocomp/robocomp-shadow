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


# import typing
# try:
#     # Prefer typing_extensions.Self if present
#     from typing_extensions import Self as _Self
#     # Alias it into typing so uses like `from typing import Self` see a valid symbol
#     setattr(typing, "Self", _Self)
# except Exception:
#     pass

import os, sys, typing
import time
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

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, configData, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map, configData)
        self.Period = configData["Period"]["Compute"]
        self.hide()
        if startup_check:
            self.startup_check()
        else:

            # Create a PlaneDetector
            self.plane_detector = PlaneDetector(
                voxel_size=0.05,
                angle_tolerance_deg=10.0,
                ransac_threshold=0.01,
                min_plane_points=100,
                nms_normal_dot_threshold=0.99,
                nms_distance_threshold=0.1,
                plane_thickness=0.01
            )

            # Initialize the ground truth particle (for reference)
            # NOTE: z_center should be set to the vertical center of your room's walls
            # Default is 1.25m (half of 2.5m height). Adjust if walls are detected higher/lower.
            ground_truth_particle = (Particle
                (
                x=0.0,
                y=0.0,
                theta=0.0,  # 0 degrees
                length=5.0,  # 5 meters long
                width=10.0,  # 4 meters wide
                height=2.5,  # 2.5 meters high
                weight=1.0,
            ))

            # Initialize the particle filter
            self.particle_filter = RoomParticleFilter(num_particles=1,
                                                      initial_hypothesis=ground_truth_particle,
                                                      device="cuda",
                                                      use_gradient_refinement=True,
                                                      adaptive_particles=False,
                                                      min_particles=20,
                                                      max_particles=300)

            self.current_best_particle = None
            self.room_box_height = 2.5

            # Create a visualizer object
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(width=600, height=600)
            self.view_control = self.vis.get_view_control()
            self.view_control.set_front([5, -5, 20])
            self.view_control.set_lookat([0, 0, 1.5])
            self.view_control.set_up([0, 0, 1])
            self.view_control.set_zoom(0.01)
            # # To store the camera state between frames
            self.camera_parameters = self.view_control.convert_to_pinhole_camera_parameters()
            # Use numpy arrays for colors to avoid Open3D warnings
            self.h_color = np.array([0.0, 0.0, 1.0])
            self.v_colors = [
                np.array([1.0, 0.0, 0.0], dtype=np.float64),  # Red
                np.array([0.0, 1.0, 0.0], dtype=np.float64),  # Green
                np.array([1.0, 1.0, 0.0], dtype=np.float64),  # Yellow
                np.array([0.0, 1.0, 1.0], dtype=np.float64),  # Cyan
                np.array([1.0, 0.0, 1.0], dtype=np.float64)  # Magenta
            ]
            self.o_color = np.array([1.0, 0.0, 1.0], dtype=np.float64)

            # robot
            self.robot_velocity = [0.0, 0.0, 0.0]  # [vx, vy, omega] in m/s and rad/s

            # Command ring buffer: store (timestamp, vx, vy, omega)
            self.cmd_buffer = deque(maxlen=300)  # ~3s at 100 Hz; adjust as needed
            now = time.monotonic()
            self.cmd_buffer.append((now, 0.0, 0.0, 0.0))

            # PF tick timing
            self.last_tick_time = time.monotonic()
            self.last_pred_time = time.monotonic()  # last time we integrated odometry
            self.cycle_time = 0.1  # initial guess

            # Optional: smoothing factor if you want to low-pass joystick (1.0 = no smoothing)
            self.vel_alpha = 1.0

            # Load the Shadow .obj mesh
            self.shadow_mesh = o3d.io.read_triangle_mesh("src/meshes/shadow.obj", print_progress=True)
            self.shadow_mesh.paint_uniform_color([1, 0, 1])
            self.vis.add_geometry(self.shadow_mesh)

            # where to write the history JSON
            self.plot_history_path = os.path.join(tempfile.gettempdir(), "pf_history.json")
            # print(f"[SpecificWorker] Plot history path: {self.plot_history_path}")

            # start the external plotter subprocess
            plotter_py = os.path.join(os.path.dirname(__file__), "plotter.py")
            self._plotter_proc = subprocess.Popen([sys.executable, "-u", plotter_py, self.plot_history_path],
                                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                                  start_new_session=True)

            self.Period = 100
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
        # print(f"[SpecificWorker] Cycle time: {self.cycle_time*1000:.1f} ms")

        # read lidar data
        pcd = o3d.geometry.PointCloud()
        if not (res := self.read_lidar_data()):
            return False
        pcd.points = res

        # Detect planes ONCE per iteration
        h_planes, v_planes, o_planes, outliers = self.plane_detector.detect(pcd)

        # Extract wall points ONCE (with fallback to all points)
        wall_points_np = self._extract_wall_points(pcd, v_planes)
        wall_points_torch = torch.from_numpy(wall_points_np[:, :2]).float().to('cuda')

        # Estimate latency tau (s) and integrate commands over [t0-tau, t1-tau]
        tau = min(0.20, max(0.0, 1.0 * self.cycle_time))  # e.g., 1× cycle, ≤200 ms
        dx, dy, dtheta = self.integrate_cmds_with_latency(t0, t1, 1.1 * self.cycle_time)
        odometry_delta = (dx, dy, dtheta)

        # Run particle filter step with pre-computed wall points
        self.particle_filter.step(odometry_delta, wall_points_torch)
        self.current_best_particle = self.particle_filter.best_particle()

        # Visualize results using the already-detected planes
        self.visualize_results(pcd, h_planes, v_planes, o_planes, outliers)

        # Send data to plotter
        self.send_data_to_plotter()

        self.last_tick_time = now
        return True

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

    ############################# MY CODE HERE #############################
    def read_lidar_data(self):
        lidar_data = self.lidar3d_proxy.getLidarDataWithThreshold2d("helios", 10000, 3)
        if not lidar_data.points or len(lidar_data.points) == 0:
            console.log("[yellow]No lidar data received.[/yellow]")
            return False

        # Convert to open3d format
        points_list = [[p.x / 1000.0, p.y / 1000.0, p.z / 1000.0] for p in lidar_data.points]
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
                    "weight_entropy": hist.get("weight_entropy", [])[-N:]
                }
                tmp = self.plot_history_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(payload, f)
                os.replace(tmp, self.plot_history_path)  # atomic swap
            except Exception:
                pass

    def visualize_results(self, pcd, h_planes, v_planes, o_planes, outliers):
        """Create and display all visualization geometries (room-centric frame)"""
        if self.current_best_particle is None:
            return

        geometries_to_draw = []

        # Get transform to room-centric frame
        p = self.current_best_particle

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
                                                          depth=0.02)
        floor_mesh.translate([-p.length / 2, -p.width / 2, 0.0])
        floor_mesh.paint_uniform_color([1.0, 0.86, 0.58])
        geometries_to_draw.append(floor_mesh)

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
        outlier_cloud = pcd_room.select_by_index(outliers)
        outlier_cloud.paint_uniform_color(np.array([0.5, 0.5, 0.5], dtype=np.float64))
        geometries_to_draw.append(outlier_cloud)

        # Update visualizer
        self.vis.clear_geometries()
        for geom in geometries_to_draw:
            self.vis.add_geometry(geom)

        if self.camera_parameters is not None:
            self.view_control.convert_from_pinhole_camera_parameters(self.camera_parameters)

        self.vis.poll_events()
        self.camera_parameters = self.view_control.convert_to_pinhole_camera_parameters()
        self.vis.update_renderer()

    def record_velocity_command(self, vx: float, vy: float, omega: float) -> None:
        """
        Call this whenever a new joystick command arrives.
        It updates self.robot_velocity and appends to the ring buffer with timestamp.
        """
        now = time.monotonic()
        # Optional low-pass
        # vx = self.vel_alpha * vx + (1 - self.vel_alpha) * self.robot_velocity[0]
        # vy = self.vel_alpha * vy + (1 - self.vel_alpha) * self.robot_velocity[1]
        # omega = self.vel_alpha * omega + (1 - self.vel_alpha) * self.robot_velocity[2]
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