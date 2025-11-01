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

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, configData, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map, configData)
        self.Period = configData["Period"]["Compute"]
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
                    height=2.5, # 2.5 meters high
                    weight=1.0,
                ))

            # Initialize the particle filter
            self.particle_filter = (RoomParticleFilter
            (
                num_particles=10,
                initial_hypothesis=ground_truth_particle  # Can use RoomAssembler here for a 1-shot guess
            ))
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
                np.array([1.0, 0.0, 1.0], dtype=np.float64)   # Magenta
            ]
            self.o_color = np.array([1.0, 0.0, 1.0], dtype=np.float64)

            # Add a floor
            # self.floor = o3d.geometry.TriangleMesh.create_box(width=5, height=10, depth=0.1)
            # self.floor.translate([-2.5, -5, -0.1])  # Adjust position
            # self.floor.paint_uniform_color([1, 0.86, 0.58])  # Set color to light gray
            # self.vis.add_geometry(self.floor)

            # Load the Shadow .obj mesh
            self.shadow_mesh = o3d.io.read_triangle_mesh("src/meshes/shadow.obj", print_progress=True)
            self.shadow_mesh.paint_uniform_color([1, 0, 1])
            self.vis.add_geometry(self.shadow_mesh)

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""


    @QtCore.Slot()
    def compute(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = self.read_lidar_data()

        # Get odometry
        odometry_delta = (0.0, 0.0, 0.0)  # (dx, dy, dtheta)

        # Run particle filter step: Predict -> Detect -> Update -> Resample
        self.particle_filter.step(odometry_delta, self.plane_detector, pcd)
        self.current_best_particle =  self.particle_filter.best_particle()
        h_planes, v_planes, o_planes, outliers = self.plane_detector.detect(pcd)

        # Visualize results
        self.visualize_results(pcd, h_planes, v_planes, o_planes, outliers)

        return True

    ############################# MY CODE HERE #############################
    def read_lidar_data(self):
        lidar_data = self.lidar3d_proxy.getLidarDataWithThreshold2d("helios", 10000, 3)
        if not lidar_data.points:
            console.log("[yellow]No lidar data received.[/yellow]")
            return True

        # Convert to open3d format
        points_list = [[p.x / 1000.0, p.y / 1000.0, p.z / 1000.0] for p in lidar_data.points]
        return o3d.utility.Vector3dVector(points_list)

    def visualize_results(self, pcd, h_planes, v_planes, o_planes, outliers):
        """Create and display all visualization geometries (room-centric frame)"""
        geometries_to_draw = []
        
        # Get transform to room-centric frame
        if self.current_best_particle is not None:
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
        else:
            # No particle - show world frame
            pcd_room = pcd
            R_inv = np.eye(3)
            translation = np.array([0, 0, 0])
        
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
        if self.current_best_particle is not None:
            p = self.current_best_particle
            
            # Room at origin with no rotation
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
            
            # Transform floor to room frame
            floor_mesh = o3d.geometry.TriangleMesh.create_box(width=p.length,
                                                              height=p.width,
                                                              depth=0.02)
            # center the floor at (0,0) in the room frame (Open3D box origin is at the corner)
            floor_mesh.translate([-p.length / 2, -p.width / 2, 0.0])
            floor_mesh.paint_uniform_color([1.0, 0.86, 0.58])
            geometries_to_draw.append(floor_mesh)

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


