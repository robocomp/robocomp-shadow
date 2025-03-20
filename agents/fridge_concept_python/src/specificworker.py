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

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import numpy as np
import time
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points
import matplotlib.pyplot as plt

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

from pydsr import *


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 100

        # YOU MUST SET AN UNIQUE ID FOR THIS AGENT IN YOUR DEPLOYMENT. "_CHANGE_THIS_ID_" for a valid unique integer
        self.agent_id = 600
        self.g = DSRGraph(0, "pythonAgent", self.agent_id)

        try:
            # signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.update_node_att)
            # signals.connect(self.g, signals.UPDATE_NODE, self.update_node)
            # signals.connect(self.g, signals.DELETE_NODE, self.delete_node)
            # signals.connect(self.g, signals.UPDATE_EDGE, self.update_edge)
            # signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.update_edge_att)
            # signals.connect(self.g, signals.DELETE_EDGE, self.delete_edge)
            console.print("signals connected")
        except RuntimeError as e:
            print(e)

        self.rt_api = rt_api(self.g)
        self.inner_api = inner_api(self.g)

        # Check if room exists
        room_node = None
        room_nodes = self.g.get_nodes_by_type("room")
        if room_nodes:
            room_node = room_nodes[0]
            room_width, room_depth, room_polygon = self.generate_room_polygon(room_node)

        self.lidar_data = np.array([[0.0, 0.0]])
        self.transformed_lidar_data = np.array([[0.0, 0.0]])

        # Set up the plot
        plt.ion()  # Turn on interactive mode
        self.fig, ax = plt.subplots()
        self.scatter = ax.scatter( self.lidar_data[:, 0],  self.lidar_data[:, 1], c='blue', label="LiDAR Points")
        if room_node:
            self.room_plot = ax.plot(*room_polygon.exterior.xy,  c='red', label="Room")
        ax.set_xlim(-5000, 5000)
        ax.set_ylim(-5000, 5000)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Real-Time Point Plotting (Single Thread)")
        plt.legend()

        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        # try:
        #	self.innermodel = InnerModel(params["InnerModelPath"])
        # except:
        #	traceback.print_exc()
        #	print("Error reading config params")
        return True


    @QtCore.Slot()
    def compute(self):
        # Check if room exists
        room_node = None
        room_nodes = self.g.get_nodes_by_type("room")
        if room_nodes:
            room_node = room_nodes[0]
            # console.print(f"Room node found: {room_node.id}", style='green')

        # Get lidar data
        timestamp, self.lidar_data = self.read_lidar_helios()
        # Get robot node and pose
        robot_node = self.g.get_node("Shadow")
        if not robot_node:
            console.print("Robot node not found", style='red')
            return
        robot_pose = self.read_robot_pose(room_node, robot_node)

        room_width, room_depth, room_polygon = self.generate_room_polygon(room_node)

        self.transformed_lidar_data = -(robot_pose @ self.lidar_data.T).T

        # Get residuals from lidar points
        residuals = self.project_points_to_polygon(self.transformed_lidar_data, room_polygon, 200)

        # Get fridges from graph (TODO after inserting first fridge in graph)

        # Update actionables

        # fridges = actionable_fridge.update()

        self.scatter.set_offsets(residuals)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def read_lidar_helios(self):
        try:
            lidar_data = self.lidar3d_proxy.getLidarData("helios", 0, 2 * np.pi, 2)
            if lidar_data is None:
                console.print("Lidar data is None")
                return -1, np.array([])
            p_filter = np.array([np.array([point.x, point.y, 1]) for point in lidar_data.points if point.z > 0.1 and point.distance2d > 200])
            return lidar_data.timestamp, p_filter
        except Ice.Exception as e:
            console.print_exception(e)
            console.log("Error reading lidar data")
            return -1, np.array([])

    def read_robot_pose(self, room_node, robot_node):
        robot_edge_rt = self.rt_api.get_edge_RT(room_node, robot_node.id)
        robot_tx, robot_ty, _ = robot_edge_rt.attrs['rt_translation'].value
        _, _, robot_rz = robot_edge_rt.attrs['rt_rotation_euler_xyz'].value

        # Rotation matrix
        rotation = np.array([
            [np.cos(robot_rz), -np.sin(robot_rz), 0],
            [np.sin(robot_rz), np.cos(robot_rz), 0],
            [0, 0, 1]
        ])

        # Translation matrix
        translation = np.array([
            [1, 0, robot_tx],
            [0, 1, robot_ty],
            [0, 0, 1]
        ])

        # Combined transformation matrix (translation * rotation)
        transformation_matrix = translation @ rotation
        return transformation_matrix

    def generate_room_polygon(self, room_node):
        room_width = room_node.attrs["width"].value
        room_depth = room_node.attrs["depth"].value

        room_width_half = room_width / 2
        room_depth_half = room_depth / 2

        # Construct a polygon representing the room
        room_polygon = Polygon([
            (room_width_half, room_depth_half),
            (room_width_half, -room_depth_half),
            (-room_width_half, -room_depth_half),
            (-room_width_half, room_depth_half)
        ])
        return room_width, room_depth, room_polygon


    def project_points_to_polygon(self, points: np.ndarray, polygon: Polygon, distance_threshold: float) -> np.ndarray:
        """
        Projects LiDAR points onto a polygon and removes points that are close to the polygon.

        :param points: Nx2 array of LiDAR points (each row is a point [x, y]).
        :param polygon: Shapely Polygon representing the room.
        :param distance_threshold: Points closer than this distance to the polygon will be removed.
        :return: Filtered Nx2 array of LiDAR points.
        """

        return np.array([point for point in points if polygon.exterior.distance(Point(point)) > distance_threshold])

    def startup_check(self):
        print(f"Testing RoboCompLidar3D.TPoint from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TPoint()
        print(f"Testing RoboCompLidar3D.TDataImage from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TDataImage()
        print(f"Testing RoboCompLidar3D.TData from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TData()
        QTimer.singleShot(200, QApplication.instance().quit)




    ######################
    # From the RoboCompLidar3D you can call this methods:
    # self.lidar3d_proxy.getLidarData(...)
    # self.lidar3d_proxy.getLidarDataArrayProyectedInImage(...)
    # self.lidar3d_proxy.getLidarDataProyectedInImage(...)
    # self.lidar3d_proxy.getLidarDataWithThreshold2d(...)

    ######################
    # From the RoboCompLidar3D you can use this types:
    # RoboCompLidar3D.TPoint
    # RoboCompLidar3D.TDataImage
    # RoboCompLidar3D.TData



    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        console.print(f"UPDATE NODE ATT: {id} {attribute_names}", style='green')

    def update_node(self, id: int, type: str):
        console.print(f"UPDATE NODE: {id} {type}", style='green')

    def delete_node(self, id: int):
        console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):

        console.print(f"UPDATE EDGE: {fr} to {type}", type, style='green')

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')

    def delete_edge(self, fr: int, to: int, type: str):
        console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
