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
from PySide6.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import numpy as np
from typing import Optional, Tuple

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

from pydsr import *
from src.table_manager import TableManager

# Choose visualizer: '2d' for DearPyGui, '3d' for Open3D
VISUALIZER_MODE = '3d'

if VISUALIZER_MODE == '3d':
    from src.visualizer_3d import BoxConceptVisualizer3D as BoxConceptVisualizer
else:
    from src.visualizer import BoxConceptVisualizer


class SpecificWorker(GenericWorker):
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
            console.print("signals connected")
        except RuntimeError as e:
            print(e)

        # Initialize table manager
        self.table_manager = TableManager(self.g, self.g.get_agent_id())

        # Initialize visualizer
        self.visualizer = BoxConceptVisualizer()
        self.visualizer.start_async()

        if startup_check:
            self.startup_check()
        else:



            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""


    @QtCore.Slot()
    def compute(self):
        # Get lidar data
        lidar_points = self.get_lidar_points() # meters
        #print(len(lidar_points))

        # Check if room node exists in G
        room_dims = self.get_room_dimensions()
        if room_dims is None:
            return True

        # Get robot pose and covariance from shadow node
        robot_pose, robot_cov = self.get_robot_pose_and_cov()
        if robot_pose is None:
            return True

        # Call table manager to process the data
        detected_tables = self.table_manager.update(lidar_points, robot_pose, robot_cov, room_dims)

        # Debug: compare first belief against GT every N frames (set to 0 to disable)
        debug_every_n_frames = 100
        if debug_every_n_frames > 0 and len(detected_tables) > 0 and self.table_manager.frame_count % debug_every_n_frames == 0:
            TableManager.debug_belief_vs_gt(detected_tables[0].to_dict(),
                                            gt_cx=0.0, gt_cy=0.0,
                                            gt_w=1.0, gt_h=0.6,
                                            gt_table_height=0.75,
                                            gt_theta=0.0)

        # Update visualizer
        self.visualizer.update(
            room_dims=room_dims,
            robot_pose=robot_pose,
            lidar_points_raw=self.table_manager.viz_data['lidar_points_raw'],
            lidar_points_filtered=self.table_manager.viz_data['lidar_points_filtered'],
            clusters=self.table_manager.viz_data['clusters'],
            beliefs=self.table_manager.get_beliefs_as_dicts(),
            historical_points=self.table_manager.get_historical_points_for_viz()
        )

        if len(detected_tables) > 0:
            pass  # console.print(f"[cyan]Tracking {len(detected_tables)} tables")

        return True

    ##############################################################
    def get_room_dimensions(self) -> Optional[Tuple[float, float]]:
        """Get room dimensions from DSR graph.

        Returns:
            Tuple (width, length) in meters, or None if room node not found
        """
        room_nodes = self.g.get_nodes_by_type("room")
        if not room_nodes:
            console.print("[yellow]Room node not found in G")
            return None

        if room_nodes[0] and room_nodes[0].name == "room":   # Take the first room node
            room_node = room_nodes[0]
            try:
                # get all node attributes
                #all_attrs = {name: attr.value for name, attr in room_node.attrs.items()}
                #console.print(f"[cyan]Room attributes: {all_attrs}")

                width_mm = room_node.attrs["room_width"].value
                length_mm = room_node.attrs["room_length"].value
                width_m = width_mm / 1000.0
                depth_m = length_mm / 1000.0
                return (width_m, depth_m)
            except KeyError as e:
                console.print(f"[red]Room attributes not found: {e}")
                return None
        return None

    def get_robot_pose_and_cov(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get robot pose and covariance from the shadow node RT edge to room.

        Returns:
            Tuple (pose [x, y, theta], covariance 3x3 matrix) or (None, None) if not found
        """
        shadow_nodes = self.g.get_nodes_by_type("robot")
        room_nodes = self.g.get_nodes_by_type("room")

        if not shadow_nodes:
            console.print("[yellow]Shadow node not found in G")
            return None, None

        if not room_nodes:
            return None, None

        shadow_node = shadow_nodes[0]
        room_node = room_nodes[0]

        # Get RT edge from room to shadow (parent -> child)
        rt_edge = self.g.get_edge(room_node.id, shadow_node.id, "RT")
        if rt_edge is None:
            console.print("[yellow]RT edge from room to Shadow not found")
            return None, None

        try:
            translation = rt_edge.attrs["rt_translation"].value  # [x, y, z] in mm
            rotation = rt_edge.attrs["rt_rotation_euler_xyz"].value  # [rx, ry, rz] in rad

            # Convert to meters
            x_m = translation[0] / 1000.0
            y_m = translation[1] / 1000.0
            theta = rotation[2]  # rz is the heading angle

            pose = np.array([x_m, y_m, theta])

            # Get covariance if available
            if "rt_se2_covariance" in rt_edge.attrs:
                cov_flat = rt_edge.attrs["rt_se2_covariance"].value
                cov_matrix = np.array(cov_flat).reshape(3, 3)
            else:
                # Default covariance if not available
                cov_matrix = np.eye(3) * 0.01

            return pose, cov_matrix

        except KeyError as e:
            console.print(f"[red]RT edge attributes not found: {e}")
            return None, None

    def get_lidar_points(self) -> np.ndarray:
        """Get LIDAR points in 3D.

        Coordinate convention in robot frame: x+ = right, y+ = forward, z+ = up
        Returns [N, 3] array with (x, y, z) coordinates in meters.
        """
        try:
            # Get 3D LIDAR data
            bpearl = self.lidar3d_proxy.getLidarDataWithThreshold2d("bpearl", 8000, 1)
            helios = self.lidar3d1_proxy.getLidarDataWithThreshold2d("helios", 8000, 1)
            # LIDAR points in robot frame: p.x = right, p.y = forward, p.z = up
            bpearl_np = np.array([[p.x / 1000.0, p.y / 1000.0, p.z / 1000.0] for p in bpearl.points])
            helios_np = np.array([[p.x / 1000.0, p.y / 1000.0, p.z / 1000.0] for p in helios.points if p.z < 1000])
            lidar_points = np.vstack((bpearl_np, helios_np))

        except Ice.Exception as e:
            print(f"Error reading lidar: {e}")
            lidar_points = np.array([]).reshape(0, 3)

        return lidar_points


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


    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        console.print(f"UPDATE NODE ATT: {id} {attribute_names}", style='green')

    def update_node(self, id: int, type: str):
        console.print(f"UPDATE NODE: {id} {type}", style='green')

    def delete_node(self, id: int):
        console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):
        pass
        #console.print(f"UPDATE EDGE: {fr} to {type}", type, style='green')

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        #console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')
        pass

    def delete_edge(self, fr: int, to: int, type: str):
        console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
