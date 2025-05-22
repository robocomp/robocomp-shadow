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
import time
import sys
import matplotlib.pyplot as plt
from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from numpy.polynomial.polynomial import polyline
from rich.console import Console
from genericworker import *
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import queue
from collections import deque
import cv2
import open3d as o3d
import copy
import math
import itertools
import cupy as cp
from room_model import RoomModel
import interfaces as ifaces
import torch
from scipy.optimize import linear_sum_assignment
import random
from scipy.spatial import distance
from scipy.signal import find_peaks


sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)


from pydsr import *


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 1

        # YOU MUST SET AN UNIQUE ID FOR THIS AGENT IN YOUR DEPLOYMENT. "_CHANGE_THIS_ID_" for a valid unique integer
        self.agent_id = 404
        self.g = DSRGraph(0, "pythonAgent", self.agent_id)

        # --------------- ODOMETRY ----------------
        self.odometry_queue_len = 50
        self.odometry_queue = deque(maxlen=self.odometry_queue_len)
        self.odometry_node_id = 200


        # # --------------- PROCESSING --------------
        # self.voxel_size = 0.1
        # self.max_height = 3.0
        # self.min_height = 1.2
        # # ----- remove_outliers -----
        # self.nb_neighbors = 400
        # self.std_ratio = 1
        # # ----- get_hough_lines -----
        # self.rho_min = -8000.0
        # self.rho_max = 8000.0
        # self.rho_step = 80  # self.voxel_size * 1000
        # self.theta_min = -np.pi
        # self.theta_max = np.pi + np.deg2rad(1)
        # self.theta_step = np.deg2rad(0.5)
        # self.min_votes = 20
        #
        # self.lines_max = 50  # 150
        # self.line_threshold = 25  # 50
        #
        # # ----- filter_parallel_hough_lines
        # self.theta_thresh = np.pi / 4
        # self.rho_thresh = 350

        # --------------- PROCESSING --------------
        self.voxel_size = 0.05
        self.max_height = 3.0
        self.min_height = 1.0
        # ----- remove_outliers -----
        self.nb_neighbors = 400
        self.std_ratio = 1

        # ----- get_hough_lines -----
        self.rho_min = -7000.0
        self.rho_max = 7000.0
        self.rho_step = 100
        self.theta_min = -np.pi
        self.theta_max = np.pi + np.deg2rad(1)
        self.theta_max = 0
        self.theta_max = 0.0
        self.theta_step = np.deg2rad(1)
        self.min_votes = 20

        self.lines_max = 200  # 150
        self.line_threshold = 100  # 50

        # ----- filter_parallel_hough_lines
        self.theta_thresh = np.pi /8
        self.rho_thresh = 500

        # ------ get corners ------
        # self.votes = 5
        # self.CORNER_SUPPORT_THRESHOLD = 0.5
        self.CORNERS_PERP_ANGLE_THRESHOLD = np.deg2rad(10)
        # self.NMS_MIN_DIST_AMONG_CORNERS = 0.5
        self.SUPPORT_DISTANCE_THRESHOLD = self.voxel_size * 1.1
        self.SUPPORT_PERCENTAGE_THRESHOLD = 0.6
        self.MAX_WALL_HOLE_WIDTH = 1.2

        if startup_check:
            self.startup_check()

        self.rt_api = rt_api(self.g)
        self.inner_api = inner_api(self.g)

        # --------------- VISUALIZACION ----------------
        o3d.core.Device("CUDA:0")

        self.mesh_view = False
        self.open3d_view = True

        # ----------------
        self.read_deque = deque(maxlen=1)
        # Filter points categories self.categories_filter = [0, 1, 22, 8, 14] in labels
        self.categories_filter = [0, 1, 22, 8, 14, 114]
        self.accumulated_pcs = np.empty((0, 3), dtype=np.float32)
        self.last_pointcloud = None
        self.last_pointcloud_exists = False
        self.last_robot_pose_timestamp = 0.0
        self.act_segmented_pointcloud_timestamp = 0

        if self.open3d_view:
            self.w, self.scatter, self.lines, self.hough_lines, self.corners_plot = self.initialize_application()

        if self.mesh_view:
            ####### Initialize Open3D visualizer for room mesh #######
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name='Real-Time 3D Point Cloud', height=480, width=640)
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            self.vis.add_geometry(axes)

            # Add a floor
            floor = o3d.geometry.TriangleMesh.create_box(width=10, height=10, depth=0.1)
            floor.translate([-5, -5, -0.1])  # Adjust position
            floor.paint_uniform_color([0.8, 0.8, 0.8])  # Set color to light gray
            self.vis.add_geometry(floor)

            # Add pointclouds
            self.pcd_explained = o3d.geometry.PointCloud()
            self.pcd_unexplained = o3d.geometry.PointCloud()
            points = np.random.rand(3, 3)
            self.pcd_explained.points = o3d.utility.Vector3dVector(points)
            self.pcd_unexplained.points = o3d.utility.Vector3dVector(points)
            self.vis.add_geometry(self.pcd_explained)
            self.vis.add_geometry(self.pcd_unexplained)
            self.vis.poll_events()
            self.vis.update_renderer()

            # Set up the camera
            view_control = self.vis.get_view_control()
            view_control.set_zoom(35)  # Adjust zoom level

            # Add robot geometry
            # Load the Shadow .obj mesh
            self.shadow_mesh = o3d.io.read_triangle_mesh("../../etc/meshes/shadow.obj", print_progress=True)
            self.shadow_mesh.paint_uniform_color([1, 0, 1])
            self.vis.add_geometry(self.shadow_mesh)

            self.residuals_ant = torch.tensor(np.array([], dtype=np.float32), dtype=torch.float32,
                                              device="cuda")

        # Statemachine actual state
        self.state = "finding_room"

        # Room model variables
        self.room = None

        current_edges = self.g.get_edges_by_type("current")
        if len(current_edges) > 0:
            # Get room node
            room_node = self.g.get_node(current_edges[0].destination)
            room_timestamp = room_node.attrs["timestamp_alivetime"].value
            # Get room data from graph
            room_corners = self.g.get_nodes_by_type("corner")
            # Iterate over room_corners and remove those which name include "measured". Then, order by corner_id
            room_corners = sorted([corner for corner in room_corners if "measured" not in corner.name],
                                  key=lambda x: x.attrs["corner_id"].value)
            corner_poses = []
            for corner in room_corners:
                print("Corner")
                corner_pose = self.inner_api.get_translation_vector(room_node.name, corner.name, sys.maxsize)

                corner_poses.append([corner_pose[0]/ 1000, corner_pose[1] / 1000])
            corner_poses.append(corner_poses[0])
            corner_poses = np.array(corner_poses, dtype=np.float32)
            print("Room corners", corner_poses)
            self.room = RoomModel(corner_poses, room_timestamp, height=2.5,
                                  device="cuda")
            self.state = "update_room_data"

        self.setWindowTitle("2D view")
        self.setFixedSize(1000, 1000)
        self.scale = 100  # pixels per meter
        self.origin = QPointF(self.width() / 2, self.height() / 2)  # center of the widget
        self.lidar_points_2d = np.array([])
        self.target_points = deque()
        self.intention_active = False
        self.target_tolerance = np.array([200, 200, 0, 0, 0, 0.5], dtype=np.float32)

        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)

        #       ------------------- POSE-------------------------
        self.robot_pose = np.array([0.0, 0.0, 0.0])

        # DSR Signals
        try:
            signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.update_node_att)
            # signals.connect(self.g, signals.UPDATE_NODE, self.update_node)
            # signals.connect(self.g, signals.DELETE_NODE, self.delete_node)
            # signals.connect(self.g, signals.UPDATE_EDGE, self.update_edge)
            # signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.update_edge_att)
            # signals.connect(self.g, signals.DELETE_EDGE, self.delete_edge)
            console.print("signals connected")
        except RuntimeError as e:
            print(e)

    def __del__(self):

        # Kill scatter plot
        self.scatter.clear()
        # Kill QGLWidget
        self.ui.viewer_3D.clear()
        """Destructor"""

    def setParams(self, params):
        return True

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(Qt.blue)
        pen.setWidth(2)
        painter.setPen(pen)

        lidar_points_copy = copy.deepcopy(self.lidar_points_2d) / 1000

        for point in lidar_points_copy:
            screen_point = self.world_to_screen(point[0], point[1])
            painter.drawPoint(screen_point)

        # Draw target points
        pen.setColor(Qt.red)
        painter.setPen(pen)
        for point in self.target_points:
            point = self.world_to_screen(point[0], point[1])
            painter.drawEllipse(point, 5, 5)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x_world, y_world = self.screen_to_world(event.x(), event.y())
            self.target_points.append(np.array([x_world, y_world, 0], dtype=np.float32))
            print(f"Target set at ({x_world:.2f}, {y_world:.2f}) meters")

    @QtCore.Slot()
    def compute(self):
        # ROOM State Machine
        # 1. Existe Room?
        #   SI -> Verificación?
        #          SI -> Update Model. (Update Corners in DSR)
        #          NO -> Inicializar de nuevo.
        #          Con Error -> Reajustar Modelo.
        #   NO -> INICIALIZACIÓN (BUCLE)
        #           ALMACENAR DATOS y GENERAR MODELO <->  VERIFICAR?
        #               SI -> Modelo Verificado -> INICIALIZAR
        #               NO -> Seguir Acumulando

        if self.read_deque:

            # Variables globales
            #     State machine state self.state
            #     Existe room? self.exists_room
            #     room_model object self.room
            #     self.robot_pose
            #     self.odometry_queue
            match self.state:
                case "finding_room":
                    ################# Pointcloud processing #################
                    self.accumulated_pcs, mean_center = self.processing_input_data(accumulate=True)
                    # Get accumulated pointcloud mean center
                    if not mean_center.size == 0:
                        print("Mean center", mean_center)

                        room = self.g.get_node("room")
                        if room:
                            self.insert_RT_edge("room", "Shadow",
                                                np.array([self.robot_pose[0] * 1000, self.robot_pose[1] * 1000, 0],
                                                         dtype=np.float32),
                                                np.array([0, 0, self.robot_pose[2]], dtype=np.float32),
                                                self.act_segmented_pointcloud_timestamp)
                            self.check_stabilizing_affordance_node(np.array([mean_center[0] * 1000, mean_center[1] * 1000, 0],
                                                         dtype=np.float32))
                        else:
                            # Create room_pre node and intention
                            self.create_room_for_stabilize(mean_center)

                    # Get Hough lines
                    np_pcd_2D_points, hough_lines = self.get_room_lines_from_pointcloud(self.accumulated_pcs)
                    # Compute corners
                    polyline, corners_to_display, _, _, _ = self.get_corners(hough_lines, pc2d_points=np_pcd_2D_points)
                    print("Polyline", polyline, "Hough lines", hough_lines)
                    # Transform polyline to a np.array of 2d points and process it to remove intermediate corners
                    corner_poses = self.remove_false_corners(np.array([corner[1] for corner in polyline]), epsilon=0.5)
                    # Generate room model with PyTorch3D
                    self.room = RoomModel(corner_poses, self.act_segmented_pointcloud_timestamp, height=2.5,
                                          device="cuda")
                    # Check if room has characteristics required: the room is closed and the robot falls inside the room
                    if self.room.is_mesh_closed_by_corners(corner_poses) and self.room.is_point_inside_room(
                            corner_poses):
                        # Draw PyTorch3D mesh
                        if self.mesh_view:
                            lidar_points = torch.tensor(np.array(self.accumulated_pcs, dtype=np.float32),
                                                        dtype=torch.float32,
                                                        device="cuda")  # Convert to meters
                            residuals_filtered, explained_points, face_counts = self.room.remove_explained_points(
                                lidar_points, 0.2)
                            self.draw_room(self.room, residuals_filtered, face_counts)
                        self.state = "initialize_room"
                    # Draw pointcloud and room with Open3D
                    if self.open3d_view:
                        self.update_visualization(self.scatter, self.accumulated_pcs.copy(), self.lines, self.hough_lines,
                                                  hough_lines.copy(),
                                                  self.corners_plot, polyline, [])
                case "initialize_room":
                    room_node = self.g.get_node("room")
                    # Get robot node
                    robot_node = self.g.get_node("Shadow")
                    # Get affordance node stabilize
                    affordance_node = self.g.get_node("aff_stabilizing")
                    if affordance_node:
                        self.g.delete_edge(robot_node.id, room_node.id, "has_intention")
                        affordance_node.attrs["bt_state"] = Attribute("completed", self.agent_id)
                        self.g.update_node(affordance_node)
                        self.intention_active = False

                    self.send_room_to_dsr(self.room)
                    self.state = "update_room_data"

                case "update_room_data":

                    start = time.time()
                    current_pointcloud, target_point = self.processing_input_data(accumulate=False)
                    # Perform corner detection
                    np_pcd_2D_points, hough_lines = self.get_room_lines_from_pointcloud(current_pointcloud)
                    # Compute corners
                    _, detected_corners, _, _, _ = self.get_corners(hough_lines, pc2d_points=np_pcd_2D_points,
                                                                    corner_detection=True)
                    # Transform polyline to a np.array of 2d points and process it to remove intermediate corners
                    detected_corners = self.remove_false_corners(np.array([corner[1] * 1000 for corner in detected_corners], dtype=np.float32), epsilon=0.5)
                    # Calculate mahalanobis distance matrix and execute Hungarian algorithm to match nominal and measured corners
                    matched_corners = self.match_corners(self.room.get_corners() * 1000, detected_corners,
                                                         threshold=400)
                    print("###############################################################")
                    print("###############################################################")
                    print("Match corners time", time.time() - start)
                    # Update graph data if posible
                    print("@@@@@@@@@@@@@@@@@@@@@@@@······················ timestamps current pc", self.act_segmented_pointcloud_timestamp)
                    for i, matched_corner in enumerate(matched_corners):
                        measured_corner = self.g.get_node("corner_measured_" + str(i+1))
                        if not np.any(np.isnan(matched_corner)):
                            # print("Matched corner", matched_corner)
                            measured_corner.attrs['valid'] = Attribute(True, self.agent_id)
                            self.insert_RT_edge("Shadow", "corner_measured_" + str(i+1),
                                                np.array([matched_corner[0], matched_corner[1], 0], dtype=np.float32),
                                                np.array([0, 0, 0], dtype=np.float32), self.act_segmented_pointcloud_timestamp)
                        else:
                            measured_corner.attrs['valid'] = Attribute(False, self.agent_id)
                        self.g.update_node(measured_corner)
                    print("Update corners data time", time.time() - start)
                    # Check affordance node status
                    self.check_wandering_affordance_node()
                    print("Check Wandering time", time.time() - start)
                    if self.open3d_view:
                        self.update_visualization(self.scatter, current_pointcloud.copy(), self.lines, self.hough_lines,
                                                  hough_lines.copy(),
                                                  self.corners_plot, self.room.get_corners(), matched_corners * 0.001)
                    print("Update visualization time", time.time() - start)
                    print("###############################################################")
                    print("###############################################################")
            # Update visualization
            if self.mesh_view:
                self.vis.poll_events()
                self.vis.update_renderer()

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    def send_room_to_dsr(self, room):
        # Get robot node
        robot_node = self.g.get_node("Shadow")
        # Get room node
        room_node = self.g.get_node("room")
        # Get room corners
        room_corners = room.get_corners() * 1000
        # Get room dimensions TODO: think about it

        # Insert RT edge between "room" and "Shadow"
        for i in range(30): # TODO: Please, modify RT:API TO remove ñapa
            self.insert_RT_edge(room_node.name, robot_node.name,
                                np.array([self.robot_pose[0] * 1000, self.robot_pose[1] * 1000, 0], dtype=np.float32),
                                np.array([0.0, 0.0, self.robot_pose[2]], dtype=np.float32), self.act_segmented_pointcloud_timestamp)

        self.insert_room_structure_into_DSR(room_corners, room_node)

        self.insert_room_affordances(room_node)

        current_edge = Edge(room_node.id, room_node.id, "current", self.agent_id)
        self.g.insert_or_assign_edge(current_edge)

    def insert_room_node(self, root_node, room_name):
        room_node = Node(agent_id=self.agent_id, type="room", name=room_name)

        pos_x = 350.0
        pos_y = 250.0 * (1 % 3 - 1)
        if abs(pos_y) > 700:
            pos_y = 250.0 * (random.randint(0, 2) - 1)

        room_node.attrs['parent'] = Attribute(int(root_node.id), self.agent_id)
        room_node.attrs['color'] = Attribute("red", self.agent_id)
        room_node.attrs['pos_x'] = Attribute(float(pos_x), self.agent_id)
        room_node.attrs['pos_y'] = Attribute(float(pos_y), self.agent_id)
        room_node.attrs['obj_checked'] = Attribute(False, self.agent_id)
        room_node.attrs['level'] = Attribute(root_node.attrs["level"].value + 1, self.agent_id)
        room_node.attrs['room_id'] = Attribute(0, self.agent_id)
        room_node.attrs['timestamp_alivetime'] = Attribute(int(self.act_segmented_pointcloud_timestamp), self.agent_id)

        self.g.insert_node(room_node)
        return room_node

    def insert_RT_edge(self, origin_name, destination_name, translation, orientation, timestamp, verbose=False):

        origin = self.g.get_node(origin_name)
        destination = self.g.get_node(destination_name)
        self.rt_api.insert_or_assign_edge_RT(origin, destination.id, translation,
                                             orientation, np.uint64(timestamp))
        if verbose:
            rt_edge = self.rt_api.get_edge_RT(origin, destination.id)
            if rt_edge is not None:
                translation = rt_edge.attrs["rt_translation"].value
                rotation = rt_edge.attrs["rt_rotation_euler_xyz"].value
                timestamps = rt_edge.attrs["rt_timestamps"].value
                print("RT edge data from", origin_name, "to", destination_name, ":")
                print("Translation", translation)
                print("Rotation", rotation)
                print("Timestamps", timestamps)

    def insert_room_structure_into_DSR(self,
                             room_corners: np.ndarray,  # Shape (N, 2) for N corners
                             room_node: Node
                             ) -> None:
        """
        Process room corners to create walls and corner nodes using NumPy optimizations.

        Args:
            room_corners: NumPy array of shape (N, 2) containing (x, y) coordinates
            room_id: Identifier for the room
            room_node: Parent node for walls
            G: Graph object with get_node method
        """
        if len(room_corners) < 2:
            return

        # Create circularly shifted array for next corners
        next_corners = np.roll(room_corners, -1, axis=0)

        # Vectorized wall center calculations (all walls at once)
        wall_centers = ((room_corners + next_corners) * 0.5)[:-1]

        # Vectorized wall angles calculation
        wall_angles = self.calculate_wall_angles(room_corners)
        room_corners = room_corners[:-1]

        robot_node = self.g.get_node("Shadow")
        robot_pose = np.array([
            [self.robot_pose[2], -self.robot_pose[2], self.robot_pose[0] * 1000],
            [self.robot_pose[2], self.robot_pose[2], self.robot_pose[1] * 1000],
            [0, 0, 1]
        ], dtype=np.float32)

        # Process each wall-corner pair
        for i in range(len(room_corners)):
            # print(i)
            # Create wall with pre-computed values
            wall_node = self.create_wall(i+1, wall_centers[i], wall_angles[i], room_node)
            # print("Wall node level", wall_node.attrs["level"].value)
            # Given wall_centers[i] and wall_angles[i], generate a np.array with x,y, angle
            wall_pose = np.array([wall_centers[i][0], wall_centers[i][1]], dtype=np.float32)

            # Get wall node if exists
            corner_node = self.create_corner(i+1, room_corners[i], wall_node.name, room_node, wall_pose)
            print("Corner node level", corner_node.attrs["level"].value)

            corner_measured_node = self.create_corner_measured(i+1, room_corners[i], robot_node, robot_pose)

    def calculate_wall_angles(self, room_corners):
        # Circularly shifted array for next corners
        next_corners = np.roll(room_corners, -1, axis=0)

        # Calculate wall vectors (p2 - p1 for each wall)
        wall_vectors = next_corners - room_corners

        # Calculate normal vectors (rotated 90° counterclockwise for outward normal)
        normals = np.empty_like(wall_vectors)
        normals[:, 0] = -wall_vectors[:, 1]  # x component = -dy
        normals[:, 1] = wall_vectors[:, 0]  # y component = dx

        # Adjust normal direction to point away from origin (0,0)
        # We want the normal that's in the opposite direction from the wall's midpoint to origin
        midpoints = (room_corners + next_corners) * 0.5
        dot_products = np.sum(midpoints * normals, axis=1)

        # Flip normals that point towards origin
        normals[dot_products < 0] *= -1

        # Calculate angles in radians (-π to π)
        angles = np.arctan2(normals[:, 0], normals[:, 1])

        return -angles[:-1]  # Exclude last wall if it's closing the room

    def create_wall(self, wall_id, wall_center, wall_angle, room_node):
        room_id = room_node.id

        # Get parent node(room) level
        room_node_level = room_node.attrs["level"].value

        print("Creating wall", wall_id, wall_center, wall_angle, room_node_level)

        wall_node = Node(agent_id=self.agent_id, type="wall", name=f"wall_{room_node.attrs['room_id'].value}_{wall_id}")
        wall_node.attrs['pos_x'] = Attribute(float(wall_center[0] * 0.1), self.agent_id)
        wall_node.attrs['pos_y'] = Attribute(float(-wall_center[1] * 0.1), self.agent_id)
        wall_node.attrs['room_id'] = Attribute(room_node.attrs['room_id'].value, self.agent_id)
        wall_node.attrs['level'] = Attribute(room_node_level + 1, self.agent_id)
        wall_node.attrs['parent'] = Attribute(int(room_id), self.agent_id)
        # Insert node in graph
        self.g.insert_node(wall_node)

        # Create wall node RT edge
        for i in range(30):# TODO: Please, modify RT:API TO remove ñapa
            self.insert_RT_edge(room_node.id, wall_node.id, np.array([wall_center[0], wall_center[1], 0], dtype=np.float32),
                                np.array([0, 0, -wall_angle], dtype=np.float32), self.act_segmented_pointcloud_timestamp)
        return wall_node

    def create_corner(self, corner_id, corner_pose, wall_node_name, room_node, wall_pose):
        # Get parent node(room) level
        wall_node = self.g.get_node(wall_node_name)
        wall_node_level = wall_node.attrs["level"].value

        print("Creating corner", corner_id, corner_pose, wall_node_level)

        corner_node = Node(agent_id=self.agent_id, type="corner",
                           name=f"corner_{room_node.attrs['room_id'].value}_{corner_id}")
        corner_node.attrs['pos_x'] = Attribute(float(corner_pose[0] * 0.1), self.agent_id)
        corner_node.attrs['pos_y'] = Attribute(float(-corner_pose[1] * 0.1), self.agent_id)
        corner_node.attrs['room_id'] = Attribute(room_node.attrs['room_id'].value, self.agent_id)
        corner_node.attrs['corner_id'] = Attribute(corner_id, self.agent_id)
        corner_node.attrs['level'] = Attribute(wall_node_level + 1, self.agent_id)
        corner_node.attrs['parent'] = Attribute(int(wall_node.id), self.agent_id)
        # Insert node in graph
        self.g.insert_node(corner_node)

        corner_in_wall_pose = self.transform_point_to_wall_frame(wall_node_name, corner_pose)
        print("Corner in wall pose", corner_in_wall_pose)
        # Create wall node RT edge
        for i in range(30):# TODO: Please, modify RT:API TO remove ñapa
            self.insert_RT_edge(wall_node.id, corner_node.id,
                                np.array([corner_in_wall_pose[0], corner_in_wall_pose[1], 0], dtype=np.float32),
                                np.array([0, 0, 0], dtype=np.float32), self.act_segmented_pointcloud_timestamp)

        return corner_node

    def create_corner_measured(self, corner_id, corner_pose, robot_node, robot_pose):
        robot_node_level = robot_node.attrs["level"].value

        print("Creating corner measured", corner_id, corner_pose)

        corner_node = Node(agent_id=self.agent_id, type="corner", name=f"corner_measured_{corner_id}")
        corner_node.attrs['pos_x'] = Attribute(float(robot_node.attrs["pos_x"].value + 1000), self.agent_id)
        corner_node.attrs['pos_y'] = Attribute(float(corner_id * 50), self.agent_id)
        corner_node.attrs['level'] = Attribute(robot_node_level + 1, self.agent_id)
        corner_node.attrs['corner_id'] = Attribute(corner_id, self.agent_id)
        corner_node.attrs['parent'] = Attribute(int(robot_node.id), self.agent_id)
        corner_node.attrs['valid'] = Attribute(False, self.agent_id)
        # Insert node in graph
        self.g.insert_node(corner_node)
        # Transform nominal corner to robot frame

        corner_in_robot_frame = self.inner_api.transform("room", np.array([corner_pose[0], corner_pose[1], 0],
                                                                          dtype=np.float32), "Shadow")

        self.insert_RT_edge(robot_node.id, corner_node.id,
                            np.array([corner_in_robot_frame[0], corner_in_robot_frame[1], 0], dtype=np.float32),
                            np.array([0, 0, 0], dtype=np.float32), self.act_segmented_pointcloud_timestamp)

        return corner_node

    def transform_point_to_wall_frame(self, wall_name, corner_point):
        # Return and array which "y" is 0 and "x" is the norm between wall_pose and corner_point
        print("Transforming point to wall frame", wall_name, corner_point)
        return self.inner_api.transform(wall_name, np.array([corner_point[0], corner_point[1], 0], dtype=np.float32),
                                        "room")

    def insert_room_affordances(self, room_node):
        # Get room node level
        room_node_level = room_node.attrs["level"].value

        # Create affordance node
        affordance_node = Node(agent_id=self.agent_id, type="affordance", name="aff_wandering")
        affordance_node.attrs['pos_x'] = Attribute(float(room_node.attrs["pos_x"].value) + 25, self.agent_id)
        affordance_node.attrs['pos_y'] = Attribute(float(room_node.attrs["pos_y"].value) + 25, self.agent_id)
        affordance_node.attrs['level'] = Attribute(room_node_level + 1, self.agent_id)
        affordance_node.attrs['parent'] = Attribute(int(room_node.id), self.agent_id)
        affordance_node.attrs['active'] = Attribute(False, self.agent_id)
        affordance_node.attrs['bt_state'] = Attribute("waiting", self.agent_id)
        affordance_node.attrs['room_id'] = Attribute(int(0), self.agent_id)
        # Insert node in graph
        self.g.insert_node(affordance_node)

        has_edge = Edge(affordance_node.id, room_node.id, "has", self.agent_id)
        self.g.insert_or_assign_edge(has_edge)

    def plot_mesh_open3d(self, mesh):
        verts = mesh.verts_packed().cpu().numpy()
        faces = mesh.faces_packed().cpu().numpy()

        # Create Open3D TriangleMesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

        # Compute normals for better visualization
        o3d_mesh.compute_vertex_normals()

        # Visualize
        o3d.visualization.draw_geometries([o3d_mesh])

    def get_points_xy_mean(self, points):
        """
        Calcula el punto promedio (solo en x, y) de un array de arrays de puntos 3D.

        Args:
          puntos_3d: Un array de NumPy donde cada elemento es un array [x, y, z].

        Returns:
          Un array de NumPy [x_promedio, y_promedio].
        """
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("La entrada debe ser un array de NumPy 2D con 3 columnas (x, y, z).")

        # Seleccionamos solo las columnas x (índice 0) e y (índice 1)
        puntos_xy = points[:, :2]

        # Calculamos el promedio a lo largo del eje 0 (es decir, el promedio de cada columna)
        promedio_xy = np.mean(puntos_xy, axis=0)

        return promedio_xy

    def create_room_for_stabilize(self, mean_center):

        # Get robot node
        robot_node = self.g.get_node("Shadow")
        # Get root node
        root_node = self.g.get_node("root")
        # Create room_pre node and intention
        room_node = self.insert_room_node(root_node, "room")
        for i in range(30): # TODO: Please, modify RT:API TO remove ñapa
            self.insert_RT_edge(root_node.name, room_node.name, np.array([0.0, 0.0, 0.0], dtype=np.float32),
                                np.array([0.0, 0.0, 0.0], dtype=np.float32), self.act_segmented_pointcloud_timestamp)

        # Remove RT edge between "root" and "Shadow"
        self.g.delete_edge(root_node.id, robot_node.id, "RT")
        # Change robot node level
        robot_node.attrs["level"].value = room_node.attrs["level"].value + 1
        robot_node.attrs["parent"].value = room_node.id
        self.g.update_node(robot_node)

        inserted_room_node = self.g.get_node("room")

        for i in range(30):# TODO: Please, modify RT:API TO remove ñapa
            print("Pose rt", np.array([mean_center[0] * 1000, mean_center[1] * 1000, 0], dtype=np.float32))
            self.insert_RT_edge("room", "Shadow", np.array([self.robot_pose[0] * 1000, self.robot_pose[1] * 1000, 0], dtype=np.float32), np.array([0, 0, self.robot_pose[2]], dtype=np.float32), self.act_segmented_pointcloud_timestamp)

        # Create affordance node
        affordance_node = Node(agent_id=self.agent_id, type="affordance", name="aff_stabilizing")
        affordance_node.attrs['pos_x'] = Attribute(float(inserted_room_node.attrs["pos_x"].value) + 25, self.agent_id)
        affordance_node.attrs['pos_y'] = Attribute(float(inserted_room_node.attrs["pos_y"].value) + 25, self.agent_id)
        affordance_node.attrs['level'] = Attribute(inserted_room_node.attrs["level"].value + 1, self.agent_id)
        affordance_node.attrs['parent'] = Attribute(int(inserted_room_node.id), self.agent_id)
        affordance_node.attrs['active'] = Attribute(False, self.agent_id)
        affordance_node.attrs['bt_state'] = Attribute("waiting", self.agent_id)
        affordance_node.attrs['room_id'] = Attribute(int(0), self.agent_id)
        # Insert node in graph
        self.g.insert_node(affordance_node)

        has_edge = Edge(affordance_node.id, inserted_room_node.id, "has", self.agent_id)
        self.g.insert_or_assign_edge(has_edge)


    def processing_input_data(self, accumulate=False):

        segmented_pointcloud = self.read_deque.pop()
        self.act_segmented_pointcloud_timestamp = segmented_pointcloud.timestamp
        print("############################### ACT TIMESTAMP", self.act_segmented_pointcloud_timestamp)
        category = np.array(segmented_pointcloud.CategoriesArray).flatten()
        # Generate np.array from new_pc arrayX, arrayY, arrayZ
        new_pc = np.column_stack(
            [np.array(segmented_pointcloud.XArray), np.array(segmented_pointcloud.YArray),
             np.array(segmented_pointcloud.ZArray)]) / 1000.0
        if self.state == "update_room_data":
            self.lidar_points_2d = self.transform_lidar_points_to_room(new_pc)
            self.draw_lidar_qt()

        # PC filter Category & height
        height_mask = (new_pc[:, 2] < self.max_height) & (new_pc[:, 2] > self.min_height)
        category_mask = np.isin(category, self.categories_filter)
        new_pc = new_pc[category_mask & height_mask]

        if not accumulate:
            robot_stabilization_target = self.get_points_xy_mean(new_pc) if self.accumulated_pcs.shape[
                                                                                              0] > 3 else np.array([])
            return np.asarray(new_pc), robot_stabilization_target

        # Integrate pointcloud transformed to robot origin
        transformed_points = self.integrate_odometry_to_pointcloud(new_pc,
                                                                   segmented_pointcloud.timestamp)

        self.accumulated_pcs = np.vstack((self.accumulated_pcs, transformed_points))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.accumulated_pcs)

        # Voxelization (new_pc)
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # pcd = self.remove_outliers(pcd, self.nb_neighbors, self.std_ratio)
        # Get mean center for generating target to stabilize room # TODO: Maybe "get_points_xy_mean" should be adapted to a different target depending the stabilization target (e.g. wall, corner, etc)
        robot_stabilization_target = self.get_points_xy_mean(self.accumulated_pcs) if self.accumulated_pcs.shape[0] > 3 else np.array([])
        return np.asarray(pcd.points), robot_stabilization_target

    def update_robot_pose(self):
        queue_copy = self.get_odometry_simple(np.array(self.odometry_queue.copy()),
                                              self.last_robot_pose_timestamp,
                                              self.act_segmented_pointcloud_timestamp)
        if not queue_copy:
            print("No odometry values or invalid format")
            return np.eye(3)

        # Construir accum 4x4 desde self.robot_pose
        # get adv, side, angular from self.robot_pose
        x = self.robot_pose[0]
        y = self.robot_pose[1]
        angle = self.robot_pose[2]

        # Build 4x4 transformation matrix using adv, side and rot
        cos_rot = np.cos(angle)
        sin_rot = np.sin(angle)

        accum = np.array([
            [cos_rot.item(), -sin_rot.item(), 0, x.item()],
            [sin_rot.item(), cos_rot.item(), 0, y.item()],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        if self.open3d_view:
            self.update_robot_axes(accum)

        prev_time = queue_copy[0][3]
        for i in range(len(queue_copy)):
            curr_time = queue_copy[i][3]
            dt = curr_time - prev_time
            curr_speed = queue_copy[i][:3] * dt * 0.001

            # Construct the 3x3 transformation matrix for this step
            T = np.array([
                [np.cos(curr_speed[2]), -np.sin(curr_speed[2]), 0, curr_speed[1]],
                [np.sin(curr_speed[2]), np.cos(curr_speed[2]), 0, curr_speed[0]],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            # Accumulate the transformation
            accum = np.dot(accum, T)
            prev_time = curr_time

        # update robot pose, get from accum matrix
        self.robot_pose[0] = accum[0, 3]
        self.robot_pose[1] = accum[1, 3]
        self.robot_pose[2] = np.arctan2(accum[1, 0], accum[0, 0])

        self.last_robot_pose_timestamp = self.act_segmented_pointcloud_timestamp

        # Update robot pose in DSR
        # Insert RT edge between "room" and "Shadow"
        room_node = self.g.get_node("room")
        robot_node = self.g.get_node("Shadow")
        self.insert_RT_edge(room_node.name, robot_node.name,
                            np.array([self.robot_pose[0] * 1000, self.robot_pose[1] * 1000, 0],
                                     dtype=np.float32), np.array([0.0, 0.0, self.robot_pose[2]],
                                                                 dtype=np.float32), self.last_robot_pose_timestamp)

    def integrate_odometry_to_pointcloud(self, new_pc, timestamp):
        # # copy odometry_list to avoid modifying the original
        queue_copy = self.get_odometry_simple(np.array(self.odometry_queue.copy()), self.last_robot_pose_timestamp,
                                              timestamp)
        if not queue_copy:
            print("No odometry values or invalid format")
            return np.eye(3)

        # Construir accum 4x4 desde self.robot_pose
        # get adv, side, angular from self.robot_pose
        x = self.robot_pose[0]
        y = self.robot_pose[1]
        angle = self.robot_pose[2]

        # Build 4x4 transformation matrix using adv, side and rot
        cos_rot = np.cos(angle)
        sin_rot = np.sin(angle)

        accum = np.array([
            [cos_rot.item(), -sin_rot.item(), 0, x.item()],
            [sin_rot.item(), cos_rot.item(), 0, y.item()],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        if self.open3d_view:
            self.update_robot_axes(accum)

        prev_time = queue_copy[0][3]
        for i in range(len(queue_copy)):
            curr_time = queue_copy[i][3]
            dt = curr_time - prev_time
            curr_speed = queue_copy[i][:3] * dt * 0.001

            # Construct the 3x3 transformation matrix for this step
            T = np.array([
                [np.cos(curr_speed[2]), -np.sin(curr_speed[2]), 0, curr_speed[1]],
                [np.sin(curr_speed[2]), np.cos(curr_speed[2]), 0, curr_speed[0]],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            # Accumulate the transformation
            accum = np.dot(accum, T)
            prev_time = curr_time

        # update robot pose, get from accum matrix
        self.robot_pose[0] = accum[0, 3]
        self.robot_pose[1] = accum[1, 3]
        self.robot_pose[2] = np.arctan2(accum[1, 0], accum[0, 0])

        self.last_robot_pose_timestamp = timestamp

        # Transform every three first elements of the new_pc and build homogeneus matrix
        homogeneous_points = np.hstack([new_pc, np.ones((new_pc.shape[0], 1))])

        if self.last_pointcloud_exists:
            # Apply the transformation to the current point cloud
            transformed_points = np.dot(accum, homogeneous_points.T).T[:, :3] / np.dot(accum, homogeneous_points.T).T[:,
                                                                                3][:, None]

            self.last_pointcloud = transformed_points
        else:
            # total_points = np.vstack((self.accumulated_pcs, new_pc))
            self.last_pointcloud = new_pc

        self.last_pointcloud_exists = True
        # Stack a 1 as fourth element
        # return total_points # Return as numpy array if needed
        return transformed_points  # Return as numpy array if needed

    def get_odometry_simple(self, queue, last_ts, pc_ts):
        lower = min(float(last_ts), float(pc_ts))
        upper = max(float(last_ts), float(pc_ts))
        return [odom for odom in queue if lower <= float(odom[3]) <= upper]

    def remove_outliers(self, pcd, nb_neighbors=200, std_ratio=0.3):
        """
        Filtra outliers usando un criterio estadístico basado en la desviación estándar.

        :param pcd: Nube de puntos Open3D
        :param nb_neighbors: Número de vecinos a considerar por punto
        :param std_ratio: Umbral de desviación estándar
        :return: Nube de puntos filtrada
        """
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return cl

    def get_room_lines_from_pointcloud(self, pcd):
        """
        Obtains lines representing room walls from a 3D point cloud using the Hough transform.

        :param pcd: Open3D 3D point cloud
        :return: Tuple (2D_point_cloud POINTS, hough_lines)
        """

        # Outlier filtering code (disabled)
        # pcd_filtered = self.remove_outliers(pcd, self.nb_neighbors, self.std_ratio)
        # self.voxel_visualization = np.asarray(pcd_filtered.points)

        # Extract only X and Y coordinates (top-down projection)
        points_2D = pcd[:, :2]

        # Calculate lines using the Hough transform method implemented in get_hough_lines
        # which internally uses OpenCV
        # hough_lines = self.get_hough_lines(points_2D)
        hough_lines = self.get_hough_lines_v2(pcd)
        # Add Z=0 coordinates to maintain the three-dimensional format required by Open3D
        points_2D = np.hstack((points_2D, np.zeros((points_2D.shape[0], 1))))
        #
        return points_2D, hough_lines

    def get_hough_lines(self, points):
        """
        Calculates Hough lines from a 2D Point Cloud points using cv2.HoughLinesPointSet.

        :param points: np.asarray(pcd2D).
        :return: List of detected lines in format [(votes, rho, theta), ...] in meters.
        """

        # Ensure the point cloud is 2D (discard third coordinate if it exists)
        if points.shape[1] > 2:
            points = points[:, :2]

        # Convert points from meters to millimeters
        points_mm = points * 1000.0

        # Normalize points to float for OpenCV (required by HoughLinesPointSet)
        points_float = np.round(points_mm).astype(np.float32).reshape(-1, 1, 2)

        # Apply HoughLinesPointSet
        lines = cv2.HoughLinesPointSet(
            points_float, self.lines_max, self.line_threshold, self.rho_min, self.rho_max, self.rho_step,
            self.theta_min, self.theta_max, self.theta_step
        )

        if lines is None:
            print("No lines detected")
            return []

        # Votes filter
        filtered_voted_lines = [line for line in lines if line[0][0] >= self.min_votes]
        # Parallel filter
        filtered_lines = self.filter_parallel_hough_lines(filtered_voted_lines, self.theta_thresh, self.rho_thresh)

        if filtered_lines is not None:
            return [(votes, rho / 1000.0, theta) for votes, rho, theta in
                    filtered_lines[:, 0]]  # Convert rho from mm to meters
        else:
            return []

    def filter_parallel_hough_lines(self, lines, theta_thresh=np.pi / 2, rho_thresh=2000):
        """
        Filters Hough lines by removing parallel ones, grouping them by theta and rho,
        and returning the most voted line from each group.

        :param lines: List of lines in format [(votes, rho, theta)] (rho in meters).
                      Lines may be nested (e.g., arrays with shape (1, 3)).
        :param theta_thresh: Angular difference threshold in radians to consider lines as parallel.
        :param rho_thresh: Distance difference threshold to consider lines as redundant.
        :return: List with the most voted lines per group [(votes, rho, theta)].
        """

        groups = []  # Each group is a list of similar lines

        def normalize_angle(theta):
            """Normaliza un ángulo al rango [-pi, pi]."""
            return (theta + np.pi) % (2 * np.pi) - np.pi

        def angle_diff(theta1, theta2):
            """
            Calcula la diferencia angular mínima entre dos ángulos normalizados.
            Se utiliza para comparar direcciones teniendo en cuenta la circularidad.
            """
            diff = np.arctan2(np.sin(theta1 - theta2), np.cos(theta1 - theta2))
            return diff

        # Ordena las líneas por votos de forma descendente
        lines = sorted(lines, key=lambda x: x[0] if np.isscalar(x[0]) else x[0].item(0), reverse=True)

        for line in lines:
            # Asegurar que line sea un array plano (extraer contenido si está anidado)
            if isinstance(line, np.ndarray):
                line = line.flatten()
            elif isinstance(line, list) and len(line) == 1:
                line = line[0]

            if len(line) != 3:
                raise ValueError(f"Line does not have three elements: {line}")

            votes, rho, theta = line
            theta_norm = normalize_angle(theta)
            added = False

            # Verifica si la línea pertenece a algún grupo existente
            for group in groups:
                ref_votes, ref_rho, ref_theta = group[0]
                ref_theta_norm = normalize_angle(ref_theta)

                # Calcula la diferencia angular mínima
                diff = angle_diff(theta_norm, ref_theta_norm)
                # Considera líneas paralelas si la diferencia angular es menor que theta_thresh o si son casi opuestas
                parallel = abs(diff) < theta_thresh or abs(abs(diff) - np.pi) < theta_thresh

                # Ajusta rho para líneas con dirección opuesta (cuando corresponda)
                if abs(abs(diff) - np.pi) < theta_thresh:
                    rho_adjusted = -rho
                else:
                    rho_adjusted = rho

                # Comprueba que los valores de rho sean compatibles
                same_rho = abs(rho_adjusted - ref_rho) < rho_thresh

                if parallel and same_rho:
                    group.append((votes, rho, theta))
                    added = True
                    break

            # Si la línea no se agrupa con ninguna existente, crea un nuevo grupo
            if not added:
                groups.append([(votes, rho, theta)])

        # De cada grupo se selecciona la línea con mayor número de votos
        best_lines = []
        for group in groups:
            best_line = max(group, key=lambda l: l[0])
            best_lines.append(best_line)

        return np.array(best_lines)

    def compute_histogram_of_normals(self, pcd):
        # Extract normals
        normals = np.asarray(pcd.normals)

        # Get 2D histogram of normals
        # Convert normals to spherical coordinates (azimuth and elevation)
        azimuth = np.arctan2(normals[:, 1], normals[:, 0])  # Angle in XY plane

        hist = np.histogram(azimuth, bins=360, range=(-np.pi, np.pi))[0]

        # Encuentra picos prominentes
        peaks, properties = find_peaks(hist, height=self.min_votes, distance=10)

        # Plot hist graph using matplotlib in same window
        # if(True):
        #     plt.clf()
        #     plt.bar(np.arange(360), hist, width=1)
        #     plt.xlim(0, 360)
        #     plt.xlabel('Angle (degrees)')
        #     plt.ylabel('Count')
        #     plt.title('Histogram of Point Cloud Normals')
        #     plt.pause(0.01)  # Pause to update the plot
        #     plt.show(block=False)
        # plt.savefig("histogram.png")
        # Get maximum  bin

        return peaks * (2 * np.pi / 360) - np.pi

    def get_hough_lines_v2(self, points):
        """
        Calculates Hough lines from a 2D Point Cloud points using cv2.HoughLinesPointSet.

        :param points: np.asarray(pcd2D).
        :return: List of detected lines in format [(votes, rho, theta), ...] in meters.
        """

        # Build pcd from points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # estimete normals from points
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=500, max_nn=50))

        # Histogram of normals
        main_direction = self.compute_histogram_of_normals(pcd)
        normals = np.asarray(pcd.normals)
        # Convertir normales a ángulos en el plano XY
        azimuth = np.arctan2(normals[:, 1], normals[:, 0])  # Radianes entre -π y π

        # print min/max azimuth
        print("MIN/MAX AZIMUTH",np.min(azimuth), np.max(azimuth))
        # Convertir tolerancia de grados a radianes
        tolerance = np.deg2rad(10)

        # Inicializamos máscara (asignaciones de dirección)
        mask_assignments = np.zeros(len(azimuth))
        complete_color = np.zeros((len(azimuth), 3))
        lines = []

        # Asignar cada punto a su dirección más cercana (si está dentro de tolerancia)
        for dir in main_direction:
            print("DIRECCION", dir)
            # Dirección opuesta
            dir_opposite = np.arctan2(np.sin(dir + np.pi), np.cos(dir + np.pi))

            # Función para calcular diferencia angular mínima
            def angular_diff(theta1, theta2):
                return np.arctan2(np.sin(theta1 - theta2), np.cos(theta1 - theta2))

            diffs_to_dir = np.abs(angular_diff(azimuth, dir))
            diffs_to_opposite = np.abs(angular_diff(azimuth, dir_opposite))

            # Crear máscaras donde se cumple la tolerancia
            assign_mask = (diffs_to_dir < tolerance) | (diffs_to_opposite < tolerance)
            assign_indices = np.where(assign_mask)[0]
            # Asignar el valor del ángulo original (dir) a los puntos que cumplan
            mask_assignments[assign_mask] = dir
            # apply dbscan
            cluster = pcd.select_by_index(np.where(assign_mask)[0])
            labels = np.array(cluster.cluster_dbscan(eps=(self.voxel_size * 20) , min_points=self.min_votes))
            # Generar colores
            n_labels = np.max(labels)

            # DISPLAY DBSCAN
            if(0):
                if n_labels >= 0 :
                    colors = plt.get_cmap("tab10")(labels / (np.max(labels) + 1e-8))
                else:
                    colors = np.zeros((len(labels), 4))

                colors[labels == -1] = [0, 0, 0, 1]  # ruido en negro
                cluster.colors = o3d.utility.Vector3dVector(colors[:, :3])
                complete_color[assign_indices] = colors[:, :3]
                plt.clf()
                plt.scatter(points[:, 0], points[:, 1], c=complete_color, s=1)
                plt.axis('equal')
                plt.title("Point cloud directions by azimuthal cluster")
                plt.pause(0.01)  # Pause to update the plot
                plt.show(block=False)

            cluster_points_np = np.asarray(cluster.points)
            if n_labels >= 0:
                original_cluster_points_np = np.asarray(cluster.points)
                for i in range(n_labels + 1):  # <- debe incluir el último label
                    mask = labels == i
                    if np.count_nonzero(mask) < 2:
                        continue  # muy pocos puntos para Hough

                    cluster_points_np = original_cluster_points_np[mask]


                    # Filter by
                    points_2d = cluster_points_np[:, :2]  # Solo XY
                    points_mm = points_2d * 1000.0  # Convertir a mm si necesario

                    # Preparar entrada para Hough (N,1,2) float32
                    points_float = np.round(points_mm).astype(np.float32).reshape(-1, 1, 2)

                        # Limit theta_min, theta_max to the direction of the cluster + pi/2
                    self.theta_min = dir - np.deg2rad(1)
                    self.theta_max = dir + np.deg2rad(1)

                    # Parámetros ajustados para mejor resolución (modifícalos según tu caso)
                    cluster_lines = cv2.HoughLinesPointSet(points_float, 2, self.line_threshold,
                                                           self.rho_min, self.rho_max, self.rho_step,
                                                           self.theta_min, self.theta_max, self.theta_step)
                    # Guardar siempre algo para esa dirección, incluso si no detecta líneas
                    if cluster_lines is None:
                        continue
                    lines.append(cluster_lines.copy())

        if len(lines) == 0:
            return []  # No se detectó ninguna línea

        # Concatenar todos los arrays (cada uno con forma (N,1,3)) en uno solo
        all_lines = np.vstack(lines)  # shape (total_N, 1, 3)

        # Normalizar rho de mm a metros (o unidades originales)
        all_lines[:, 0, 1] /= 1000.0

        # Reorganizar a forma (N, 3)
        all_lines = all_lines.reshape(-1, 3)

        filtered_lines = self.filter_parallel_hough_lines(all_lines, self.theta_thresh, 1)

        return [(line[0], line[1], line[2]) for line in filtered_lines if len(line) >= 3]

    def simplify_polyline(self, polyline, angle_threshold_deg=5):
        """
        Simplifica una polilínea eliminando puntos intermedios que estén alineados (segmento recto).

        :param polyline: Lista de puntos (en formato [(votes, np.array([x, y]), (line1, line2)), ...]).
        :param angle_threshold_deg: Umbral en grados para considerar que el ángulo es cercano a 0° o 180°.
        :return: Lista simplificada de puntos en el mismo formato.
        """
        if len(polyline) < 3:
            return polyline

        simplified = []  # Primer punto siempre se queda

        for i in range(0, len(polyline) - 1):
            # If i == 1, check polyline[-2], polyline[0] and polyline[1]
            if i == 0:
                prev_point = np.array(polyline[-2][1])
                current_point = np.array(polyline[0][1])
                next_point = np.array(polyline[1][1])
            else:
                prev_point = np.array(polyline[i - 1][1])
                current_point = np.array(polyline[i][1])
                next_point = np.array(polyline[i + 1][1])

            vec1 = current_point - prev_point
            vec2 = next_point - current_point

            # Normaliza vectores
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)

            # Producto escalar para calcular el ángulo
            dot_product = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            angle_deg = np.degrees(angle_rad)

            # Si el ángulo es cercano a 0 o 180, se considera recto
            if not (abs(angle_deg) < angle_threshold_deg or abs(angle_deg - 180) < angle_threshold_deg):
                simplified.append(polyline[i])  # No está en línea recta, se queda
        if len(simplified) > 0:
            simplified.append(simplified[0])  # Último punto siempre se queda

        return simplified

    # ===========================================================
    # ===================== VISUALIZATION =======================
    # ===========================================================

    def initialize_application(self):
        """Inicializa la aplicación PyQt y los widgets."""

        w = gl.GLViewWidget()  # Crear una ventana de visualización
        w.show()
        w.setWindowTitle('Visualización de Puntos Aleatorios en 3D')

        # Set camera zenital at 5m
        w.setCameraPosition(distance=15, elevation=90, azimuth=270)

        empty_array = np.zeros((1, 3), dtype=np.float32)
        scatter = gl.GLScatterPlotItem(pos=empty_array, size=5)  # Configurar la nube de puntos
        w.addItem(scatter)

        # Create empty line plots for lines and hough_lines
        lines = gl.GLLinePlotItem(pos=empty_array, color=(1, 0, 0, 1), width=2)
        w.addItem(lines)

        hough_lines = gl.GLLinePlotItem(pos=empty_array, color=(1, 0, 0, 1), width=2)
        w.addItem(hough_lines)

        corners = gl.GLScatterPlotItem(pos=empty_array, size=5)  # Configurar la nube de puntos
        w.addItem(corners)

        # Create robot-aligned axes (initially at origin)
        self.robot_axes = self.create_robot_axes()
        for axis in self.robot_axes:
            w.addItem(axis)

        # Crear y agregar ejes
        x_axis, y_axis, z_axis = self.create_axes()
        w.addItem(x_axis)
        w.addItem(y_axis)
        w.addItem(z_axis)

        return w, scatter, lines, hough_lines, corners

    def create_robot_axes(self):
        """Create XYZ axes that will follow robot pose."""
        # X axis (red)
        x_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [1, 0, 0]]),
            color=(1, 0, 0, 1),  # Red
            width=3,
            antialias=True
        )

        # Y axis (green)
        y_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 1, 0]]),
            color=(0, 1, 0, 1),  # Green
            width=3,
            antialias=True
        )

        # Z axis (blue)
        z_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 1]]),
            color=(0, 0, 1, 1),  # Blue
            width=3,
            antialias=True
        )
        return [x_axis, y_axis, z_axis]

    def create_axes(self):
        """Crea los ejes X, Y, Z."""
        axis_length = 1
        x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [axis_length, 0, 0]]), color=pg.glColor('r'), width=2)
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, axis_length, 0]]), color=pg.glColor('g'), width=2)
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, axis_length]]), color=pg.glColor('b'), width=2)
        return x_axis, y_axis, z_axis

    def update_visualization(self, scatter, points, lines, hough_lines_plot, hough_lines, corners_plot, corners,
                             measured_corners):

        # If self.corners len > 0, plot the corners
        # print(corners)
        if len(corners) > 0:
            corner_coords = np.array([np.append(corner[1], 0) for corner in corners])

            # Plot as large red points
            corners_plot.setData(
                pos=corner_coords,
                color=(1, 1, 0, 1),  # Red
                size=10,  # Larger size
                pxMode=True  # Size in screen pixels
            )

            # Given list of corners, update the visualization, drawing line segments between them.
            line_segments = []

            # If corners is a list of tuples, extract the coordinates
            if isinstance(corners[0], tuple):
                corners = [corner[1] for corner in corners]

            for i in range(len(corners) - 1):
                p1 = corners[i]
                p1 = np.append(p1, 0)
                p2 = corners[i + 1]
                p2 = np.append(p2, 0)

                line_segments.append([p1, p2])
                # line_segments.append([np.append(corners[i][1], 0), np.append(corners[i+1][1], 0), 0])
            # line_segments.append([np.append(self.corners[-1][1], 0), np.append(self.corners[0][1], 0)])
            #

            line_segments = np.array(line_segments).reshape(-1, 3)
            lines.setData(pos=line_segments, color=(0, 1, 0, 1), width=2)
        else:
            lines.setData(pos=np.zeros((0, 3)))  # Clear lines if no corners

        if len(measured_corners) > 0:
            # corner_coords = np.array([np.append(corner[1], 3.0) for corner in measured_corners])
            # print("Measured corners", corner_coords)
            # Plot as large red points
            corners_plot.setData(
                pos=measured_corners,
                color=(1, 0, 0, 1),  # Red
                size=20,  # Larger size
                pxMode=True  # Size in screen pixels
            )

        # Visualización de líneas de Hough (líneas amarillas)
        if hough_lines:
            line_length = 10.0  # Longitud de las líneas en metros
            hough_segments = []

            for _, rho, theta in hough_lines:
                # Convertir coordenadas polares a cartesianas
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho

                # Calcular puntos extremos de la línea
                x1 = x0 - line_length * (-b)
                y1 = y0 - line_length * a
                x2 = x0 + line_length * (-b)
                y2 = y0 + line_length * a

                # Añadir segmento como un par de puntos independiente
                # Insertamos un punto NaN para separar las líneas
                hough_segments.append([[x1, y1, -0.5], [x2, y2, -0.5], [np.nan, np.nan, np.nan]])

            # Convertir a array y aplanar, manteniendo los separadores NaN
            hough_segments = np.array(hough_segments).reshape(-1, 3)
            hough_lines_plot.setData(pos=hough_segments, color=(1, 1, 0, 1), width=1)  # Amarillo
        else:
            hough_lines_plot.setData(pos=np.zeros((0, 3)))  # Limpiar líneas si no hay resultados
        scatter.setData(pos=np.asarray(points), color=(1, 1, 1, 1), size=3)

    # ===========================================================
    # ================= Pointcloud processing ===================
    # ===========================================================

    def get_corners(self, lines, pc2d_points, corner_detection=False):
        """
        Detects corners from lines and applies non-maximum suppression.
        Also creates dictionaries that relate corners with lines and vice versa for later search
        and extracts the polyline that delimits the contour from these dictionaries.
        :param lines: List of lines in format [(votes, rho, theta), ...].
        :param pc2d_points: 2D point cloud (e.g., np.array) of the lidar scene.
        :return: tuple containing:
                 - polyline: List of corners forming the contour.
                 - corners: List of all detected corners.
                 - corner_to_lines: Dictionary relating each corner (key: intersection as tuple) with lines.
                 - line_to_corners: Dictionary relating each line with the corners it contains.
                 - close_loop: Boolean indicating if the polyline forms a closed loop.
        """
        # Initialize collections
        corners = []
        corner_to_lines = {}  # Dictionary relating corners to their lines
        line_to_corners = {}  # Dictionary relating lines to their corners

        # Extract corners from line combinations
        for line1, line2 in itertools.combinations(lines, 2):
            # Each line is given as (votes, rho, theta)
            votes_1, rho1, theta1 = line1
            votes_2, rho2, theta2 = line2

            # Calculate the angle difference between the two lines
            angle = abs(theta1 - theta2)
            if angle > np.pi:
                angle -= np.pi
            if angle < -np.pi:
                angle += np.pi

            # Check if the lines are approximately perpendicular
            if np.pi / 2 - self.CORNERS_PERP_ANGLE_THRESHOLD < angle < np.pi / 2 + self.CORNERS_PERP_ANGLE_THRESHOLD:
                # Solve the equation system to get the intersection
                A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
                b = np.array([rho1, rho2])

                # Check that lines are not parallel (determinant != 0)
                if np.linalg.det(A) != 0:
                    intersection = np.linalg.solve(A, b)
                    # Convert intersection to tuple to use as key in dictionaries
                    intersection_tuple = (float(intersection[0]), float(intersection[1]))
                    # Save the corner with its minimum vote and the two lines that generated it
                    corner_entry = (min(votes_1, votes_2), intersection, (line1, line2))
                    corners.append(corner_entry)

                    # Update corner_to_lines: key is intersection, value is a set of lines
                    if intersection_tuple not in corner_to_lines:
                        corner_to_lines[intersection_tuple] = set()
                    corner_to_lines[intersection_tuple].update([line1, line2])

                    # Update line_to_corners: each line relates to the corner
                    for ln in [line1, line2]:
                        if ln not in line_to_corners:
                            line_to_corners[ln] = []
                        line_to_corners[ln].append(intersection_tuple)

        # Sort corners by votes in descending order
        corners.sort(key=lambda x: x[0], reverse=True)

        if corner_detection:
            return [], corners, corner_to_lines, line_to_corners, False

        # Extract the polyline that delimits the contour using the created dictionaries
        polyline = []
        if len(corners) == 0:
            return [], corners, corner_to_lines, line_to_corners, False

        # Select the highest-voted corner as starting point
        start_corner = corners[0]
        polyline.append(start_corner)
        used_corners = set()  # Track used corners

        current_corner = start_corner
        finished = False
        max_iterations = len(corners)  # To avoid infinite loops
        iteration = 0
        close_loop = False

        # Main loop to build the polyline
        while not finished and iteration < max_iterations:
            iteration += 1
            found_next = False

            # Get the tuple representation of the current corner
            current_key = (float(current_corner[1][0]), float(current_corner[1][1]))

            # Get all lines that belong to the current corner
            lines_of_current = corner_to_lines.get(current_key, set())

            # For each of these lines, find associated corners as candidates
            candidates = set()
            for ln in lines_of_current:
                for candidate_key in line_to_corners.get(ln, []):
                    if candidate_key != current_key:
                        candidates.add(candidate_key)

            # Sort candidates by distance to the current corner
            candidates = sorted(candidates, key=lambda x: np.linalg.norm(np.array(x) - np.array(current_key)))

            # Review candidates that haven't been used yet and have support in the segment
            for candidate_key in candidates:
                # Find the corner in the list with that intersection
                candidate = next((c for c in corners if (float(c[1][0]), float(c[1][1])) == candidate_key), None)
                if candidate is None or candidate_key in used_corners:
                    continue

                p_current = current_corner[1]
                p_candidate = candidate[1]

                # Check if the segment has support in the point cloud
                if self.has_support(p_current, p_candidate, pc2d_points):
                    if len(polyline) > 1:
                        # Calculate angle between current segment and new segment
                        p_prev = polyline[-2][1]
                        vec_current = p_candidate - p_current
                        vec_prev = p_current - p_prev
                        angle = np.arctan2(vec_current[1], vec_current[0]) - np.arctan2(vec_prev[1], vec_prev[0])

                        # Check if angle is within tolerance of expected angles (0°, ±90°, ±270°) avoiding 180° (returning to previous)
                        tolerance = 10
                        angle_deg = np.rad2deg(angle)
                        if any(abs(angle_deg - target) <= tolerance for target in [-90, 0, 90, 270, -270]):
                            polyline.append(candidate)
                            used_corners.add(candidate_key)
                            current_corner = candidate
                            found_next = True

                            # Check if we've closed the loop by returning to the first corner
                            if len(polyline) > 2 and candidate_key == (
                                    float(polyline[0][1][0]), float(polyline[0][1][1])):
                                finished = True
                                close_loop = True
                            break
                    else:
                        # First segment, just add the candidate
                        polyline.append(candidate)
                        used_corners.add(candidate_key)
                        current_corner = candidate
                        found_next = True
                        break

            # If no valid candidate is found, exit the current iteration
            if not found_next:
                # Go to initial corner if we are not in the first iteration
                if len(polyline) > 1 and iteration < max_iterations:
                    current_corner = polyline[0]
                finished = True

        # Simplify
        # polyline = self.simplify_polyline(polyline)

        return polyline, corners, corner_to_lines, line_to_corners, close_loop

    def douglas_peucker(self, points, epsilon):
        """Implementación del algoritmo Douglas-Peucker para simplificación de polígonos."""
        dmax = 0
        index = 0
        end = len(points) - 1

        for i in range(1, end):
            d = self.perpendicular_distance(points[i], points[0], points[end])
            if d > dmax:
                index = i
                dmax = d

        if dmax > epsilon:
            rec_results1 = self.douglas_peucker(points[:index + 1], epsilon)
            rec_results2 = self.douglas_peucker(points[index:], epsilon)
            result = np.vstack((rec_results1[:-1], rec_results2))
        else:
            result = np.vstack((points[0], points[end]))

        return result

    def perpendicular_distance(self, point, line_start, line_end):
        """Calcula la distancia perpendicular de un punto a una línea."""
        if np.allclose(line_start, line_end):
            return distance.euclidean(point, line_start)
        else:
            return np.abs(np.cross(line_end - line_start, line_start - point)) / distance.euclidean(line_start,
                                                                                                    line_end)

    def remove_false_corners(self, points, epsilon=0.5):
        """Elimina esquinas falsas de un contorno de habitación.

        Args:
            points: Array numpy de puntos (n, 2) representando las esquinas
            epsilon: Umbral de distancia para considerar un punto como falso

        Returns:
            Array numpy con las esquinas reales
        """
        # Cerramos el polígono añadiendo el primer punto al final si no está ya cerrado
        # if not np.allclose(points[0], points[-1]):
        #     closed_points = np.vstack((points, points[0]))
        # else:
        closed_points = points.copy()

        # Aplicamos Douglas-Peucker
        simplified = self.douglas_peucker(closed_points, epsilon)

        # Eliminamos el último punto si es duplicado del primero
        if len(simplified) > 1 and np.allclose(simplified[0], simplified[-1]):
            simplified = simplified[:-1]
        # Check if first point in "points" is the same than last point in "points". In that case, append first point
        # to the end of the simplified points
        if np.allclose(points[0], points[-1]):
            simplified = np.vstack((simplified, points[0]))
        return simplified

    def match_corners(self, nominal_corners, observed_corners, threshold=300):
        """
        Realiza el matching de dos listas de puntos 2D utilizando el algoritmo húngaro.
        Se imprime la matriz de asociación (costos) y sólo se realiza la asociación si la
        distancia es menor o igual al umbral especificado (por defecto 0.3).

        Parámetros:
          nominal_corners: array-like, de forma (N, 2) con los puntos nominales.
          observed_corners: array-like, de forma (M, 2) con los puntos observados.
          threshold: float, umbral máximo permitido para la asociación.

        Retorna:
          matched_points: np.array de forma (N, 2) en el que cada fila contiene el punto observado
                          asociado al punto nominal correspondiente. Si la distancia es mayor que el
                          umbral se asigna np.nan.
        """
        # Convertir a arrays de numpy en caso de que no lo sean
        nominal_corners = np.array(nominal_corners)
        # observed_corners = np.array(observed_corners)

        # Get RT pose of robot
        robot_node = self.g.get_node("Shadow")
        room_node = self.g.get_node("room")
        robot_rt_edge = self.rt_api.get_edge_RT(room_node, robot_node.id)
        # print("POSE", robot_rt_edge.attrs["rt_translation"].value)
        # rt_edge_timestamps = robot_rt_edge.attrs["rt_timestamps"].value
        # print("Robot RT edge timestamps:", rt_edge_timestamps)
        robot_rt_edge_stamped_mat = self.rt_api.get_edge_RT_as_stamped_rtmat(robot_rt_edge,
                                                                   self.act_segmented_pointcloud_timestamp)
        robot_rt_edge_mat = robot_rt_edge_stamped_mat[1]

        # print("Robot DSR pose at timestamp", self.act_segmented_pointcloud_timestamp, robot_rt_edge_stamped_mat)
        homogeneus = np.array(
            [[corner[0], corner[1], 0, 1] for corner in observed_corners])

        # Transform observed_corners to room frame
        observed_corners_transformed = np.dot(robot_rt_edge_mat, homogeneus.T).T[:, :2]

        # Remove from nominal_corner the last element TODO: DUPLICATED
        nominal_corners = nominal_corners[:-1]

        # Calcular la matriz de costos (asociación) usando la distancia euclidiana entre cada par
        cost_matrix = np.linalg.norm(nominal_corners[:, np.newaxis, :] - observed_corners_transformed[np.newaxis, :, :], axis=2)

        # print("Matriz de asociación (costos):")
        # print(cost_matrix)

        # Resolver el problema de asignación usando el algoritmo húngaro
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Inicializar el array de salida con np.nan para identificar puntos sin match
        matched_points = np.full(nominal_corners.shape, np.nan)

        # Asignar sólo si la distancia es menor o igual al umbral
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= threshold:
                matched_points[r] = observed_corners[c]

        # print("Matching: nominals", row_ind, "observed", col_ind)
        # print("Matched points:", matched_points)
        return matched_points

    def has_support(self, p1, p2, pc2d_points):
        """
        Helper function: verifies that a segment between two points has support in the point cloud.
        Divides the p1-p2 segment into N samples and checks if enough points are close to the cloud.

        :param p1: First point of the segment
        :param p2: Second point of the segment
        :param pc2d_points: Point cloud in which to check for support
        :return: Boolean indicating if the segment has enough support points
        """
        # Calculate distance between p1 and p2
        distance = np.linalg.norm(p2 - p1)

        # Calculate number of samples needed
        N = int(distance / self.voxel_size)

        if N == 0:
            return False

        # Generate points along the segment
        samples = np.linspace(0, 1, N)[:, None] * (p2 - p1) + p1
        support_count = 0

        # Maximum allowed consecutive samples without support (to detect holes in walls)
        continuous_samples = self.MAX_WALL_HOLE_WIDTH // self.voxel_size
        samples_counter = continuous_samples

        # Use Euclidean distance for each sample against all pc2d points
        for s in samples:

            if samples_counter <= 0:
                # Maximum number of consecutive samples without support has been reached
                print("Segment without support - HOLE")
                return False

            # Calculate distances between the sample and pc2d points
            distances = np.linalg.norm(pc2d_points[:, :2] - s, axis=1)

            # If any point is close enough, it's considered support
            if np.any(distances < self.SUPPORT_DISTANCE_THRESHOLD):
                support_count += 1
                samples_counter = continuous_samples
            else:
                samples_counter -= 1

        # Return True if the support percentage exceeds the threshold
        return support_count / N >= self.SUPPORT_PERCENTAGE_THRESHOLD

    def add_unique_points(self, total, actual, radius=0.01):
        """
        Add points from actual to total that don't exist within radius in total using Open3D.

        Args:
            total: numpy array of shape (n, 3) containing existing points
            actual: numpy array of shape (m, 3) containing new candidate points
            radius: maximum distance to consider a point as existing (default 0.01)

        Returns:
            A new numpy array combining total with unique points from actual
        """
        if len(total) == 0:
            return actual.copy()

        if len(actual) == 0:
            return total.copy()

        # Convert arrays to Open3D point clouds
        total_pcd = o3d.geometry.PointCloud()
        total_pcd.points = o3d.utility.Vector3dVector(total)

        actual_pcd = o3d.geometry.PointCloud()
        actual_pcd.points = o3d.utility.Vector3dVector(actual)

        # Build KDTree for fast radius search
        kdtree = o3d.geometry.KDTreeFlann(total_pcd)

        # Find points in actual that don't have neighbors in total within radius
        unique_mask = []
        for point in actual:
            _, idx, _ = kdtree.search_radius_vector_3d(point, radius)
            unique_mask.append(len(idx) == 0)  # True if no neighbors found

        unique_mask = np.array(unique_mask)

        # Combine the arrays
        return np.vstack([total, actual[unique_mask]])

    def check_wandering_affordance_node(self):
        affordance_nodes = self.g.get_nodes_by_type("affordance")
        # Get affordance node which name contains "wandering"
        affordance_node = next((node for node in affordance_nodes if "wandering" in node.name), None)
        if affordance_node:
            affordance_node_is_active = affordance_node.attrs["active"].value
            if affordance_node_is_active:
                # Check affordance_node status
                affordance_node_status = affordance_node.attrs["bt_state"].value
                print("Act bt_status", affordance_node_status)
                # If affordance_node is active, was activated by scheduler. Updating
                # bt_state to "in_progress" in order to get control
                if affordance_node_status == "waiting":
                    if len(self.target_points) > 0:
                        affordance_node.attrs["bt_state"] = Attribute("in_progress", self.agent_id)
                        self.g.update_node(affordance_node)
                        self.insert_wandering_intention()
                # If bt_state is "in_progress", it's necessary to check the current has_intention
                elif affordance_node_status == "in_progress":
                    # If in progress, set it to completed
                    intention_status = self.monitor_wander_intention()
                    if intention_status:
                        # Insert new wander intention if points in self.target_points
                        # are available
                        if len(self.target_points) > 0:
                            self.insert_wandering_intention()
                        else:
                            # If no points are available, set affordance node to completed
                            affordance_node.attrs["bt_state"] = Attribute("completed", self.agent_id)
                            self.g.update_node(affordance_node)
                            affordance_node.attrs["bt_state"] = Attribute("waiting", self.agent_id)
                            self.g.update_node(affordance_node)
                            self.intention_active = False

    def check_stabilizing_affordance_node(self, target):
        affordance_nodes = self.g.get_nodes_by_type("affordance")
        # Get affordance node which name contains "wandering"
        affordance_node = next((node for node in affordance_nodes if "stabilizing" in node.name), None)
        if affordance_node:
            affordance_node_is_active = affordance_node.attrs["active"].value
            if affordance_node_is_active:
                # Check affordance_node status
                affordance_node_status = affordance_node.attrs["bt_state"].value
                print("Act bt_status", affordance_node_status)
                # If affordance_node is active, was activated by scheduler. Updating
                # bt_state to "in_progress" in order to get control
                if affordance_node_status == "waiting":
                    affordance_node.attrs["bt_state"] = Attribute("in_progress", self.agent_id)
                    self.g.update_node(affordance_node)
                    self.insert_stabilizing_intention(target)
                # If bt_state is "in_progress", it's necessary to check the current has_intention
                elif affordance_node_status == "in_progress":
                    # If in progress, set it to completed
                    intention_status = self.monitor_stabilizing_intention()
                    if intention_status:
                        # Insert new wander intention if points in self.target_points
                        # are available
                        # If no points are available, set affordance node to completed
                        affordance_node.attrs["bt_state"] = Attribute("completed", self.agent_id)
                        self.g.update_node(affordance_node)
                        affordance_node.attrs["bt_state"] = Attribute("waiting", self.agent_id)
                        self.g.update_node(affordance_node)
                        self.intention_active = False

    def insert_stabilizing_intention(self, target):
        # Get robot node
        robot_node = self.g.get_node("Shadow")
        # Get room node
        room_nodes = self.g.get_nodes_by_type("room")

        if len(room_nodes) > 0:
            current_room = room_nodes[0]
            intention_edge = Edge(current_room.id, robot_node.id, "has_intention", self.agent_id)
            intention_edge.attrs["active"] = Attribute(True, self.agent_id)
            intention_edge.attrs["state"] = Attribute("waiting", self.agent_id)
            intention_edge.attrs["offset_xyz"] = Attribute(target, self.agent_id)
            intention_edge.attrs["tolerance"] = Attribute(self.target_tolerance, self.agent_id)
            self.g.insert_or_assign_edge(intention_edge)
            self.intention_active = True


    def monitor_stabilizing_intention(self):
        # Get intention node from robot to room
        robot_node = self.g.get_node("Shadow")
        room_node = self.g.get_node("room")
        intention_edge = self.g.get_edge(robot_node.id, room_node.id, "has_intention")
        if intention_edge:
            if intention_edge.attrs["state"].value == "completed":
                self.g.delete_edge(robot_node.id, room_node.id, "has_intention")
                self.intention_active = False
                return True
            elif intention_edge.attrs["state"].value == "waiting":
                print("Waiting to base_controller aproval")
                return False
            elif intention_edge.attrs["state"].value == "in_progress":
                print("Intention in progress")
                return False

    def insert_wandering_intention(self):
        last_target = self.target_points[0]
        # Get robot node
        robot_node = self.g.get_node("Shadow")
        # Get wander affordance
        current_edges = self.g.get_edges_by_type("current")
        if current_edges:
            current_room = current_edges[0].destination
            intention_edge = Edge(current_room, robot_node.id, "has_intention", self.agent_id)
            intention_edge.attrs["active"] = Attribute(True, self.agent_id)
            intention_edge.attrs["state"] = Attribute("waiting", self.agent_id)
            intention_edge.attrs["offset_xyz"] = Attribute(last_target * 1000, self.agent_id)
            intention_edge.attrs["tolerance"] = Attribute(self.target_tolerance, self.agent_id)
            self.g.insert_or_assign_edge(intention_edge)
            self.intention_active = True

    def monitor_wander_intention(self):
        # Get intention node from robot to room
        robot_node = self.g.get_node("Shadow")
        room_node = self.g.get_node("room")
        intention_edge = self.g.get_edge(robot_node.id, room_node.id, "has_intention")
        if intention_edge:
            if intention_edge.attrs["state"].value == "completed":
                self.g.delete_edge(robot_node.id, room_node.id, "has_intention")
                self.intention_active = False
                self.target_points.popleft()
                return True
            elif intention_edge.attrs["state"].value == "waiting":
                print("Waiting to base_controller aproval")
                return False
            elif intention_edge.attrs["state"].value == "in_progress":
                print("Intention in progress")
                return False

    ############### DRAW ##################
    def draw_point_cloud(self, explained_points, unexplained_points):
        """
        Update the point cloud with new data.
        :param points: Nx3 numpy array of 3D points.
        """
        # Concatenate all clusters
        # all_clusters = np.concatenate(clusters)
        explained_points = explained_points.cpu().numpy()
        unexplained_points = unexplained_points.cpu().numpy()
        colors_explained = np.tile([0.0, 1.0, 0.0], (explained_points.shape[0], 1))  # Green color for each point
        colors_unexplained = np.tile([1.0, 0.0, 0.0], (unexplained_points.shape[0], 1))  # Green color for each point
        self.pcd_explained.points = o3d.utility.Vector3dVector(explained_points)
        self.pcd_unexplained.points = o3d.utility.Vector3dVector(unexplained_points)
        self.pcd_explained.colors = o3d.utility.Vector3dVector(colors_explained)
        self.pcd_unexplained.colors = o3d.utility.Vector3dVector(colors_unexplained)
        self.vis.update_geometry(self.pcd_explained)
        self.vis.update_geometry(self.pcd_unexplained)
        self.vis.poll_events()
        self.vis.update_renderer()

    def draw_room(self, room, unexplained_points: torch.Tensor, face_counts: torch.Tensor):
        """
        Visualizes the mesh with colors per face, unexplained points as point cloud,
        and groups wall sections with aggregated point counts.
        """
        mesh = room.forward()
        device = unexplained_points.device
        verts = mesh.verts_packed()
        faces = mesh.faces_packed()
        F = faces.shape[0]

        # corners = room.find_corners_from_wall_mesh().detach().cpu().numpy()
        # pcd_corners = o3d.geometry.PointCloud()
        # pcd_corners.points = o3d.utility.Vector3dVector(corners)
        # pcd_corners.paint_uniform_color([1.0, 1.0, 1.0])  # Red

        # Convert to Open3D for wall grouping
        verts_np = verts.detach().cpu().numpy()
        faces_np = faces.detach().cpu().numpy()
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts_np)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces_np)
        o3d_mesh.compute_vertex_normals()

        # Group triangles into walls and get aggregated point counts
        wall_groups, wall_point_counts = self._group_wall_sections(o3d_mesh, face_counts.cpu().numpy())

        # Create colors based on wall point counts
        max_wall_count = max(wall_point_counts) if len(wall_point_counts) > 0 else 1
        wall_colors = np.zeros((F, 3))

        for group_idx, (group, count) in enumerate(zip(wall_groups, wall_point_counts)):
            normalized = count / max_wall_count
            # Color scheme: blue (low) -> yellow -> red (high)
            wall_colors[group] = [normalized, normalized * 0.5, 1.0 - normalized]

        # Assign colors to vertices (average of adjacent faces)
        vert_colors = torch.zeros_like(verts)
        counts_per_vertex = torch.zeros(verts.shape[0], device=device)

        for i in range(F):
            for j in range(3):
                vidx = faces[i, j]
                vert_colors[vidx] += torch.from_numpy(wall_colors[i]).to(device)
                counts_per_vertex[vidx] += 1

        vert_colors = vert_colors / (counts_per_vertex.unsqueeze(-1) + 1e-8)

        # Update Open3D mesh with new colors
        colors_np = vert_colors.detach().cpu().numpy()
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors_np)

        # Prepare unexplained points
        unexpl_np = unexplained_points.detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(unexpl_np)
        pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red

        # Visualization setup
        render_opt = self.vis.get_render_option()
        render_opt.mesh_show_back_face = True
        render_opt.mesh_show_wireframe = True
        render_opt.line_width = 2.0

        # Add geometries with reset bounding box
        self.vis.add_geometry(o3d_mesh, reset_bounding_box=True)
        self.vis.add_geometry(pcd)
        # self.vis.add_geometry(pcd_corners)

        # Print wall information
        print(f"Found {len(wall_groups)} wall sections")
        for i, (group, count) in enumerate(zip(wall_groups, wall_point_counts)):
            print(f"Wall {i}: {len(group)} triangles, {count} explained points")

        self.vis.poll_events()
        self.vis.update_renderer()

    def draw_room_qt(self, room_corners):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw lidar points
        pen = QPen(Qt.blue)
        pen.setWidth(3)
        painter.setPen(pen)
        for x, y in self.lidar_points:
            point = self.world_to_screen(x, y)
            painter.drawPoint(point)

        # Draw target points
        pen.setColor(Qt.red)
        painter.setPen(pen)
        for x, y in self.target_points:
            point = self.world_to_screen(x, y)
            painter.drawEllipse(point, 5, 5)

    def draw_lidar_qt(self):
        # Transform 2d lidar points to screen coordinates
        self.update()  # trigger repaint

    def transform_lidar_points_to_room(self, lidar_points):
        """
        Transform lidar points to room coordinates using the robot pose.
        :param lidar_points: Lidar points in robot coordinates
        :return: Transformed points in room coordinates
        """
        # Get RT pose of robot
        robot_node = self.g.get_node("Shadow")
        room_node = self.g.get_node("room")
        robot_rt_edge = self.rt_api.get_edge_RT(room_node, robot_node.id)
        robot_rt_edge_timestamp = self.rt_api.get_edge_RT_as_rtmat(robot_rt_edge,
                                                                   self.act_segmented_pointcloud_timestamp)
        # Transform lidar points to room coordinates
        homogeneus = np.array(
            [[corner[0]*1000, corner[1]*1000, 0, 1] for corner in lidar_points])
        transformed_points = np.dot(robot_rt_edge_timestamp, homogeneus.T).T[:, :2]
        return transformed_points

    def world_to_screen(self, x, y):
        """Convert world (meters) to screen (pixels)."""
        return QPointF(self.origin.x() + x * self.scale,
                       self.origin.y() - y * self.scale)

    def screen_to_world(self, px, py):
        """Convert screen (pixels) to world (meters)."""
        return ((px - self.origin.x()) / self.scale,
                -(py - self.origin.y()) / self.scale)

    def _group_wall_sections(self, o3d_mesh, face_counts):
        """
        Helper function to group triangles into wall sections.
        Returns wall_groups and aggregated point counts.
        """
        triangles = np.asarray(o3d_mesh.triangles)
        normals = np.asarray(o3d_mesh.vertex_normals)

        wall_groups = []
        processed = set()

        for i in range(len(triangles)):
            if i in processed:
                continue

            current_wall = [i]
            processed.add(i)
            queue = [i]

            ref_normal = np.mean(normals[triangles[i]], axis=0)

            while queue:
                tri_idx = queue.pop()

                for j in range(len(triangles)):
                    if j in processed:
                        continue

                    # Check edge adjacency and normal similarity
                    shared_vertices = set(triangles[tri_idx]) & set(triangles[j])
                    if len(shared_vertices) >= 2:
                        j_normal = np.mean(normals[triangles[j]], axis=0)
                        if np.dot(ref_normal, j_normal) > 0.99:  # ~8 degree tolerance
                            current_wall.append(j)
                            processed.add(j)
                            queue.append(j)

            wall_groups.append(current_wall)

        # Aggregate point counts per wall
        wall_point_counts = [np.sum(face_counts[group]) for group in wall_groups]

        return wall_groups, wall_point_counts

    def update_robot_axes(self, pose):
        """
        Update robot axes position and orientation based on current pose.

        Args:
            pose: (x, y, z, roll, pitch, yaw) or 4x4 transformation matrix
        """
        if isinstance(pose, (tuple, list)) and len(pose) == 6:
            # Convert Euler angles to rotation matrix
            x, y, z, roll, pitch, yaw = pose
            transform = self.euler_to_matrix(roll, pitch, yaw)
            transform[:3, 3] = [x, y, z]  # Set translation
        else:
            transform = pose  # Assume it's already a 4x4 matrix

        # Base axis endpoints (before transformation)
        base_points = [
            np.array([[0, 0, 0], [1, 0, 0]]),  # X
            np.array([[0, 0, 0], [0, 1, 0]]),  # Y
            np.array([[0, 0, 0], [0, 0, 1]])  # Z
        ]

        # Apply transformation to each axis
        for i, axis in enumerate(self.robot_axes):
            # Transform points
            homogeneous = np.column_stack([base_points[i], np.ones(2)])
            transformed = (transform @ homogeneous.T).T[:, :3]
            axis.setData(pos=transformed)

    # =============== Methods for Component SubscribesTo ================
    # ===================================================================

    #
    # SUBSCRIPTION to pushLidarData method from Lidar3DPub interface
    #
    def Lidar3DPub_pushLidarData(self, lidarData):
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Lidar data timestamnp", lidarData.timestamp)
        self.read_deque.append(lidarData)

    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        if id == self.odometry_node_id:
            odom_node = self.g.get_node("Shadow")
            odom_attrs = odom_node.attrs
            self.odometry_queue.append([odom_attrs["robot_current_advance_speed"].value,
                                        odom_attrs["robot_current_side_speed"].value,
                                        odom_attrs["robot_current_angular_speed"].value,
                                        odom_attrs["timestamp_alivetime"].value])
        pass

    def update_node(self, id: int, type: str):
        # console.print(f"UPDATE NODE: {id} {type}", style='green')
        pass

    def delete_node(self, id: int):
        # console.print(f"DELETE NODE:: {id} ", style='green')
        pass

    def update_edge(self, fr: int, to: int, type: str):
        # console.print(f"UPDATE EDGE: {fr} to {type}", type, style='green')
        pass

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        # console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')
        pass

    def delete_edge(self, fr: int, to: int, type: str):
        # console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
        pass


