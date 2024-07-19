#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2024 by YOUR NAME HERE
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

from PySide2.QtCore import QTimer, QMutex
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
from g2o_graph import G2OGraph
from g2o_visualizer import G2OVisualizer
import g2o
import numpy as np
import interfaces as ifaces
from collections import deque
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import time
import setproctitle
from pydsr import *


sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

try:
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
    print("Process title set to", os.path.basename(os.getcwd()))
except:
    pass

class SpecificWorker(GenericWorker):
    """
    Manages a graph-based representation of an environment and its objects, providing
    functions for updating node and edge attributes, deleting nodes and edges, and
    handling RT messages. It also maintains a set of rooms and their initial positions.

    Attributes:
        Period (float|int): Set to `200` by default, indicating the time interval
            (in milliseconds) between successive calls to the `update` method. It
            determines how frequently the worker updates its state.
        agent_id (int|str): Used to identify the worker's agent in the simulation
            environment.
        g (Graph|NetworkXGraph): Used for managing graphs in the worker's update
            and delete operations.
        startup_check (QTimersingleShot200,QApplicationinstancequit): Used to check
            if the application should quit after a certain period of time.
        rt_api (str|int): Used to store the last RT (Real-time) edge set id that
            was received by the worker, so it can check if a new RT edge set has
            been received since the last time it was checked.
        inner_api (Callable[[],None]): Used to define a custom API for the worker
            to interact with its internal state and update its graph.
        odometry_node_id (int): Used to store the ID of the node representing the
            shadow robot in the graph, which is used for updating the robot's odometry.
        odometry_queue (List[Tuple[float,float,float,int]]): Used to store the
            odometry data points received from the ROS node. It is updated every
            200 milliseconds.
        last_odometry (float|List[float]): Used to store the last known odometry
            values (position, orientation, and advance speed) for the robot.
        g2o (None|Graph2O): Used to store the graph data in Graph2O format for the
            robot's environment.
        odometry_noise_std_dev (float|int): Used to control the noise level in
            robot odometry readings.
        odometry_noise_angle_std_dev (float|int): 0.1 by default, representing the
            standard deviation of the angle noise added to the robot's odometry
            measurements during simulation.
        measurement_noise_std_dev (float|double): Used to represent the standard
            deviation of measurement noise in the worker's measurements. It is
            used to simulate random fluctuations in the measurements during training.
        last_room_id (int|str): Used to store the last room ID seen by the worker
            before changing rooms, which allows the worker to track the current
            room ID.
        actual_room_id (str|int): Used to keep track of the current room ID that
            the worker is in, during its execution
        elapsed (float|int): Used to store the elapsed time since the worker's
            last call to the `update` method, which can be used to control the
            worker's execution rate.
        room_initialized (bool): Used to track whether a specific room has been
            initialized for RT mapping. It is set to False when the room is first
            encountered, and True when it has been successfully mapped with the
            robot's pose.
        iterations (int|float): Used to keep track of the number of iterations of
            the worker's tasks that have been performed.
        hide (bool): Used to hide or show the worker's updates during simulation.
        init_graph (bool): Set to True when the worker initializes its graph and
            False otherwise, indicating whether the graph has been initialized or
            not.
        current_edge_set (bool): Used to keep track of whether the current edge
            set has been updated recently, indicating when a new RT translation
            or rotation should be computed. It is set to True after each edge set
            update and reset to False after a certain time interval (defined as
            `rt_time_min` in the code) has passed without any updates, to avoid
            computing unnecessary RT translations or rotations.
        first_rt_set (bool): Set to True when the RT translation and rotation are
            first detected, False otherwise. It indicates whether the RT set has
            been detected for a given edge.
        translation_to_set (float|List[float]): Used to store the translation of
            the robot to set. It is updated when the RT transmission is received
            and its value represents the distance from the origin of the robot's
            frame to the origin of the set frame in the global coordinate system.
        rotation_to_set (float|List[float]): Used to store the Euler angles
            representing the rotation from the current room to the target set.
        room_polygon (List[float]): Used to store the coordinates of a polygon
            representing the boundary of a room in the environment, which is used
            for collision detection and avoidance.
        security_polygon (Shape|str): Used to store a polygon that defines the
            security area around the robot, which is used for collision detection
            and obstacle avoidance.
        initialize_g2o_graph (void): Used to initialize a Graph2O graph, which is
            a data structure representing a robot's environment, and perform other
            operations such as adding nodes and edges, setting attributes, and
            updating node positions.
        rt_set_last_time (float|int): Used to track the time since the last RT set
            was created by the worker. It is used to determine when to set a new
            RT translation and rotation value.
        rt_time_min (float|int): Defined as the minimum time interval between
            consecutive RT sets. Its purpose is to ensure that the robot's motion
            is smooth and does not oscillate excessively during navigation.
        last_update_with_corners (int|bool): Used to keep track of the last time
            corners were updated. When the corner of a room was updated, it's set
            to True, otherwise it's set to False.
        timer (QTimersingleShot200,QApplicationinstancequit): Used to schedule a
            call to the `QApplication.instance().quit()` function after 200
            milliseconds. This allows the worker to run indefinitely until the
            user presses the "Stop" button.
        compute (Callable[[],float]): Used to compute the next node id to update
            based on the current node id and the edge id. It takes no arguments
            and returns a floating-point value representing the next node id to update.
        update_node_att (List[str]): Used to update the attributes of a node in
            the graph based on certain conditions, such as when the node's id
            matches the `odometry_node_id` constant.
        update_edge (Callable[[int,int,str],None]): Called when an edge's type
            changes to "RT". It sets the `translation_to_set`, `rotation_to_set`,
            and `rt_set_last_time` attributes based on the new RT translation and
            rotation.
        update_edge_att (List[str]): Used to update the attributes of an edge based
            on a specific condition. It takes three parameters: `fr`, `to`, and
            `type`, which are the ID of the edge being updated, its destination
            ID, and the type of update respectively.

    """
    def __init__(self, proxy_map, startup_check=False):
        """
        Initializes various components of the worker, including the graph, odometry
        queue, and timer. It also sets up signal connections for updates on nodes
        and edges.

        Args:
            proxy_map (Dict[str, Any]): Used to store map-related information,
                such as the graph's nodes and edges, which are required for the
                worker's functionality.
            startup_check (bool): Used to check if the graph has been properly
                initialized before proceeding with the worker's tasks.

        """
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 50

        # YOU MUST SET AN UNIQUE ID FOR THIS AGENT IN YOUR DEPLOYMENT. "_CHANGE_THIS_ID_" for a valid unique integer
        self.agent_id = 20
        self.g = DSRGraph(0, "G2O_agent", self.agent_id)

        if startup_check:
            self.startup_check()
        else:
            self.rt_api = rt_api(self.g)
            self.inner_api = inner_api(self.g)

            self.odometry_node_id = None
            self.odometry_queue = deque(maxlen=15)
            self.last_odometry = None
            # Initialize g2o graph with visualizer
            self.g2o = G2OGraph(verbose=False)
            # self.visualizer = G2OVisualizer("G2O Graph")

            self.odometry_noise_std_dev = 1  # Standard deviation for odometry noise
            self.odometry_noise_angle_std_dev = 1  # Standard deviation for odometry noise
            self.measurement_noise_std_dev = 1  # Standard deviation for measurement noise

            self.last_room_id = None
            self.actual_room_id = None

            # time.sleep(2)
            self.elapsed = time.time()
            self.room_initialized = False
            self.iterations = 0
            self.hide()

            self.init_graph = False

            self.current_edge_set = False
            self.first_rt_set = False

            self.translation_to_set = None
            self.rotation_to_set = None

            self.room_polygon = None
            self.security_polygon = None

            self.room_initialized = True if self.initialize_g2o_graph() else False

            self.rt_set_last_time = time.time()
            self.rt_time_min = 1

            self.last_update_with_corners = time.time()

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

        try:
            signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.update_node_att)
            # signals.connect(self.g, signals.UPDATE_NODE, self.update_node)
            # signals.connect(self.g, signals.DELETE_NODE, self.delete_node)
            signals.connect(self.g, signals.UPDATE_EDGE, self.update_edge)
            signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.update_edge_att)
            # signals.connect(self.g, signals.DELETE_EDGE, self.delete_edge)
            console.print("signals connected")
        except RuntimeError as e:
            print(e)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        return True

    @QtCore.Slot()
    def compute(self):
        """
        Performs the following tasks:
        1/ Checks if enough time has passed since the last frame was computed.
        2/ Updates the robot's position and orientation using odometry data.
        3/ Generates a new RT edge for the robot based on its current position and
        orientation.
        4/ Adds the new RT edge to the graph.

        """
        if time.time() - self.elapsed > 1:
            print("Frame rate: ", self.iterations, " fps")
            self.elapsed = time.time()
            self.iterations = 0
        self.iterations += 1

        if self.room_initialized:
            self.first_rt_set = False
            # print("Room initialized")
            # Get robot odometry
            if self.odometry_queue:
                robot_node = self.g.get_node("Shadow")
                room_nodes = [node for node in self.g.get_nodes_by_type("room") if self.g.get_edge(node.id, node.id, "current")]
                if len(room_nodes) > 0:
                    room_node = room_nodes[0]
                else:
                    # Get last room node
                    room_node = self.g.get_node("room_" + str(self.actual_room_id))

                robot_edge_rt = self.rt_api.get_edge_RT(room_node, robot_node.id)
                robot_tx, robot_ty, _ = robot_edge_rt.attrs['rt_translation'].value
                robot_point = QPointF(robot_tx, robot_ty)

                robot_odometry = self.odometry_queue[-1]
                # print("Robot odometry:", robot_odometry)
                time_1 = time.time()
                adv_displacement, side_displacement, ang_displacement = self.get_displacement(robot_odometry)

                if len(self.g2o.pose_vertex_ids) == self.g2o.queue_max_len:
                    if self.security_polygon.containsPoint(robot_point, Qt.OddEvenFill):
                        print("Robot inside security polygon")
                        self.g2o.clear_graph()
                        self.initialize_g2o_graph()
                    else:
                        print("Robot outside security polygon")
                        self.g2o.remove_first_vertex()

                # Generate information matrix considering the noise
                odom_information = np.array([[1, 0.0, 0.0],
                                            [0.0, 1, 0.0],
                                            [0.0, 0.0, 0.001]])
                self.g2o.add_odometry(adv_displacement,
                                            side_displacement,
                                            ang_displacement, odom_information)

                # Check if robot pose is inside room polygon

                no_valid_corners_counter = 0
                if self.room_polygon is not None:
                    if self.room_polygon.containsPoint(robot_point, Qt.OddEvenFill):
                        for i in range(4):
                            corner_node = self.g.get_node("corner_"+str(i)+"_measured")
                            if corner_node is not None:
                                is_corner_valid = corner_node.attrs["valid"].value
                                if is_corner_valid:
                                    corner_edge = self.rt_api.get_edge_RT(robot_node, corner_node.id)
                                    corner_edge_mat = self.rt_api.get_edge_RT_as_rtmat(corner_edge, robot_odometry[3])[0:3, 3]
                                    self.g2o.add_landmark(corner_edge_mat[0], corner_edge_mat[1], 0.05 * np.eye(2), pose_id=self.g2o.vertex_count-1, landmark_id=int(corner_node.name[7])+1)
                                    # print("Landmark added:", corner_edge_mat[0], corner_edge_mat[1], "Landmark id:", int(corner_node.name[7]), "Pose id:", self.g2o.vertex_count-1)
                                else:
                                    no_valid_corners_counter += 1

                        door_nodes = [node for node in self.g.get_nodes_by_type("door") if not "pre" in node.name and
                                      node.name in self.g2o.objects]
                        # Iterate over door nodes
                        if self.security_polygon.containsPoint(robot_point, Qt.OddEvenFill):
                            for door_node in door_nodes:
                                try:
                                    is_door_valid = door_node.attrs["valid"].value
                                    if is_door_valid:
                                        door_measured_rt = door_node.attrs["rt_translation"].value
                                        if door_measured_rt[0] != 0.0 or door_measured_rt[1] != 0.0:
                                            self.g2o.add_landmark(door_measured_rt[0], door_measured_rt[1], 0.05 * np.eye(2),
                                                                  pose_id=self.g2o.vertex_count - 1,
                                                                  landmark_id=self.g2o.objects[door_node.name])
                                    else:
                                        print("Door is not valid")
                                except KeyError:
                                    print("Door node does not have valid attribute")

                chi_value = self.g2o.optimize(iterations=50, verbose=False)

                last_vertex = self.g2o.optimizer.vertices()[self.g2o.vertex_count - 1]
                opt_translation = last_vertex.estimate().translation()
                opt_orientation = last_vertex.estimate().rotation().angle()

                # print("Optimized translation:", opt_translation, "Optimized orientation:", opt_orientation)
                # cov_matrix = self.get_covariance_matrix(last_vertex)
                # print("Covariance matrix:", cov_matrix)
                # self.visualizer.update_graph(self.g2o)

                # print("No valid corners counter:", no_valid_corners_counter, self.last_update_with_corners)
                # affordance_nodes = [node for node in self.g.get_nodes_by_type("affordance") if node.attrs["active"].value]
                # if no_valid_corners_counter > 1 and self.security_polygon.containsPoint(robot_point, Qt.OddEvenFill):
                #     if time.time() - self.last_update_with_corners > 3:
                #         print("No affordance nodes active. Rotating robot")
                #         opt_orientation += np.pi/4
                # else:
                #     self.last_update_with_corners = time.time()
                #
                # # Substract pi/2 to opt_orientation and keep the number between -pi and pi
                # if opt_orientation > np.pi:
                #     opt_orientation -= np.pi
                # elif opt_orientation < -np.pi:
                #     opt_orientation += np.pi

                rt_robot_edge = Edge(robot_node.id, room_node.id, "RT", self.agent_id)
                rt_robot_edge.attrs['rt_translation'] = Attribute(np.array([opt_translation[0], opt_translation[1], .0],dtype=np.float32), self.agent_id)
                rt_robot_edge.attrs['rt_rotation_euler_xyz'] = Attribute(np.array([.0, .0, opt_orientation],dtype=np.float32), self.agent_id)
                self.g.insert_or_assign_edge(rt_robot_edge)
                self.last_odometry = robot_odometry   # Save last odometry
                # print("Time elapsed compute:", timfe.time() - init_time)
                return

        elif (self.first_rt_set and self.current_edge_set and self.translation_to_set is not None and self.rotation_to_set is not None) or time.time() - self.rt_set_last_time > 3:
            # print("Initializing g2o graph")
            # if self.last_room_id is not None:
            #     self.g.delete_edge(self.g.get_node("room_"+str(self.last_room_id)).id, self.g.get_node("Shadow").id, "RT")
            self.initialize_g2o_graph()
            self.room_initialized = True
            self.current_edge_set = False
            self.translation_to_set = None
            self.rotation_to_set = None

    def add_noise(self, value, std_dev):
        # print("Value", value, "Noise", np.random.normal(0, std_dev))
        return value + np.random.normal(0, std_dev)

    def initialize_g2o_graph(self):
        # print("Initializing g2o graph")
        # get robot pose in room
        """
        1) retrieves nodes and edges from the ROS topic, 2) validates node and
        edge presence, 3) adds nominal corners and fixed poses to the G2O graph
        for robot and door nodes, and 4) initializes last odometry timestamp.

        Returns:
            bool: 1 if the initialization of the G2O graph was successful, and 0
            otherwise.

        """
        self.g2o.clear_graph()
        robot_node = self.g.get_node("Shadow")
        room_nodes = [node for node in self.g.get_nodes_by_type("room") if self.g.get_edge(node.id, node.id, "current")]
        if len(room_nodes) > 0:
            room_node = room_nodes[0]
            room_node_id = room_node.attrs['room_id'].value
            if self.actual_room_id is not None and self.actual_room_id != room_node_id:
                self.last_room_id = self.actual_room_id
            self.actual_room_id = room_node.attrs['room_id'].value
            # print("###########################################################")
            # print("INITIALIZ>INDºG G2O GRAPH")
            # print("Room changed to", self.actual_room_id)
            # print("###########################################################")
            self.odometry_node_id = robot_node.id
            # Check if room and robot nodes exist
            if room_node is None or robot_node is None:
                print("Room or robot node does not exist. g2o graph cannot be initialized")
                return False

            if self.translation_to_set is None and self.rotation_to_set is None:
                robot_edge_rt = self.rt_api.get_edge_RT(room_node, robot_node.id)
                robot_tx, robot_ty, _ = robot_edge_rt.attrs['rt_translation'].value
                _, _, robot_rz = robot_edge_rt.attrs['rt_rotation_euler_xyz'].value
            else:
                robot_tx, robot_ty, _ = self.translation_to_set
                _, _, robot_rz = self.rotation_to_set

            # Add fixed pose to g2o
            self.g2o.add_fixed_pose(g2o.SE2(robot_tx, robot_ty, robot_rz))
            self.last_odometry = (.0, .0, .0, int(time.time()*1000))
            print("Fixed pose added to g2o graph", robot_tx, robot_ty, robot_rz)

            corner_list = []

            # Generate QPolygonF with corner values
            self.room_polygon = QPolygonF()
            room_center = QPointF(0, 0)
            for i in range(4):
                corner_node = self.g.get_node("corner_"+str(i)+"_"+str(self.actual_room_id))
                corner_edge_rt = self.inner_api.transform(room_node.name, corner_node.name)
                corner_tx, corner_ty, _ = corner_edge_rt
                corner_list.append(corner_edge_rt)
                self.room_polygon.append(QPointF(corner_tx, corner_ty))
                room_center += self.room_polygon.at(i)
                # Insert in security polygon the same point but with and offset towards the room center (0, 0)

            # Calculate room center
            room_center /= 4

            # Get room_polygon shortest side # TODO: set security polygon as a parameter that depends on room dimensions
            room_poly_bounding = self.room_polygon.boundingRect()
            d = 500
            self.security_polygon = QPolygonF()
            if self.room_polygon is not None:
                landmark_information = np.array([[0.05, 0.0],
                                                 [0.0, 0.05]])
                robot_point = QPointF(robot_tx, robot_ty)
                if self.room_polygon.containsPoint(robot_point, Qt.OddEvenFill):
                    for i in range(4):

                        # Variables for security polygon
                        dir_vector = self.room_polygon.at(i) - room_center
                        dir_vector /= np.linalg.norm(np.array([dir_vector.x(), dir_vector.y()]))
                        corner_in = self.room_polygon.at(i) - d * dir_vector
                        self.security_polygon.append(corner_in)
                        # print("Corner in:", corner_in, "corresponding to corner", corner_list[i])
                        corner_measured_node = self.g.get_node("corner_"+str(i)+"_measured")
                        if corner_measured_node is not None:
                            is_corner_valid = corner_measured_node.attrs["valid"].value
                            if is_corner_valid:
                                corner_edge_measured_rt = self.rt_api.get_edge_RT(robot_node, corner_measured_node.id)
                                # print("Eye matrix", 0.1 * np.eye(2))
                                # print("Landmark information:", landmark_information)
                                if corner_tx != 0.0 or corner_ty != 0.0:
                                    # self.g2o.add_landmark(corner_tx, corner_ty, 0.1 * np.eye(2), pose_id=0)
                                    self.g2o.add_nominal_corner(corner_list[i],
                                                              corner_edge_measured_rt.attrs['rt_translation'].value,
                                                              landmark_information, pose_id=0)
                            else:
                                print("Corner is not valid")
                                self.g2o.add_nominal_corner(corner_list[i],
                                                            None,
                                                            landmark_information, pose_id=0)

                    door_nodes = [node for node in self.g.get_nodes_by_type("door") if not "pre" in node.name and
                                  node.attrs["room_id"].value == self.actual_room_id]
                    # Iterate over door nodes
                    for door_node in door_nodes:
                        door_room_rt = self.inner_api.transform(room_node.name, door_node.name)
                        door_tx, door_ty, _ = door_room_rt
                        # Check if door is valid
                        try:
                            is_door_valid = door_node.attrs["valid"].value
                            if is_door_valid:
                                if door_tx != 0.0 or door_ty != 0.0:
                                    door_measured_rt = door_node.attrs["rt_translation"].value
                                    self.g2o.add_nominal_corner(door_room_rt,
                                                                door_measured_rt,
                                                                landmark_information, pose_id=0)
                            else:
                                print("Door is not valid")
                                self.g2o.add_nominal_corner(door_room_rt,
                                                            None,
                                                            landmark_information, pose_id=0)

                        except KeyError:
                            print("Door node does not have valid attribute")
                            self.g2o.add_nominal_corner(door_room_rt,
                                                        None,
                                                        landmark_information, pose_id=0)
                        self.g2o.objects[door_node.name] = self.g2o.vertex_count - 1
                    return True
        elif robot_node.attrs["parent"].value != 100:
            robot_parent = robot_node.attrs["parent"].value
            room_node = self.g.get_node(robot_parent)
            if self.translation_to_set is None and self.rotation_to_set is None:
                robot_edge_rt = self.rt_api.get_edge_RT(room_node, robot_node.id)
                robot_tx, robot_ty, _ = robot_edge_rt.attrs['rt_translation'].value
                _, _, robot_rz = robot_edge_rt.attrs['rt_rotation_euler_xyz'].value
            else:
                robot_tx, robot_ty, _ = self.translation_to_set
                _, _, robot_rz = self.rotation_to_set

            # Add fixed pose to g2o
            self.g2o.add_fixed_pose(g2o.SE2(robot_tx, robot_ty, robot_rz))
            self.last_odometry = (.0, .0, .0, int(time.time()*1000))
            print("Fixed pose added to g2o graph", robot_tx, robot_ty, robot_rz)
        else:
            print("Room node does not exist. g2o graph cannot be initialized")
            return False

    # def initialize_g2o_graph(self):
    #     print("Initializing g2o graph")
    #     room_nodes = [node for node in self.g.get_nodes_by_type("room") if self.g.get_edge(node.id, node.id, "current")]
    #     if len(room_nodes) > 0:
    #         self.g2o.clear_graph()
    #         room_node = room_nodes[0]
    #         if self.actual_room_id is not None and self.actual_room_id != room_node.attrs['room_id'].value:
    #             self.last_room_id = self.actual_room_id
    #         self.actual_room_id = room_node.attrs['room_id'].value
    #         # print("###########################################################")
    #         # print("INITIALIZ>INDºG G2O GRAPH")
    #         # print("Room changed to", self.actual_room_id)
    #         # print("###########################################################")
    #
    #         # get robot pose in room
    #         robot_node = self.g.get_node("Shadow")
    #         self.odometry_node_id = robot_node.id
    #         # Check if room and robot nodes exist
    #         if room_node is None or robot_node is None:
    #             print("Room or robot node does not exist. g2o graph cannot be initialized")
    #             return False
    #
    #         if self.translation_to_set is None and self.rotation_to_set is None:
    #             robot_edge_rt = self.rt_api.get_edge_RT(room_node, robot_node.id)
    #             robot_tx, robot_ty, _ = robot_edge_rt.attrs['rt_translation'].value
    #             _, _, robot_rz = robot_edge_rt.attrs['rt_rotation_euler_xyz'].value
    #         else:
    #             robot_tx, robot_ty, _ = self.translation_to_set
    #             _, _, robot_rz = self.rotation_to_set
    #
    #         # Add fixed pose to g2o
    #         self.g2o.add_fixed_pose(g2o.SE2(robot_tx, robot_ty, robot_rz))
    #         self.last_odometry = (.0, .0, .0, int(time.time()*1000))
    #         print("Fixed pose added to g2o graph", robot_tx, robot_ty, robot_rz)
    #
    #         # Get corner values from room node
    #         # corner_nodes = self.g.get_nodes_by_type("corner")
    #         # Order corner nodes by id
    #
    #         corner_list = []
    #         corner_list_measured = []
    #         for i in range(4):
    #             corner_list.append(self.g.get_node("corner_"+str(i)+"_"+str(self.actual_room_id)))
    #             corner_list_measured.append(self.g.get_node("corner_"+str(i)+"_measured"))
    #
    #         # Generate QPolygonF with corner values
    #         self.room_polygon = QPolygonF()
    #
    #         for i in range(4):
    #             corner_edge_measured_rt = self.rt_api.get_edge_RT(robot_node, corner_list_measured[i].id)
    #             corner_measured_tx, corner_measured_ty, _ = corner_edge_measured_rt.attrs['rt_translation'].value
    #             corner_edge_rt = self.inner_api.transform(room_node.name, corner_list[i].name)
    #             corner_tx, corner_ty, _ = corner_edge_rt
    #             print("Nominal corners", corner_tx, corner_ty)
    #             print("Measured corners", corner_measured_tx, corner_measured_ty)
    #             landmark_information = np.array([[0.05, 0.0],
    #                                              [0.0, 0.05]])
    #             print("Eye matrix", 0.1 * np.eye(2))
    #             print("Landmark information:", landmark_information)
    #             if corner_tx != 0.0 or corner_ty != 0.0:
    #                 # self.g2o.add_landmark(corner_tx, corner_ty, 0.1 * np.eye(2), pose_id=0)
    #                 self.g2o.add_nominal_corner(corner_edge_rt,
    #                                           corner_edge_measured_rt.attrs['rt_translation'].value,
    #                                           landmark_information, pose_id=0)
    #                 # Add corner to polygon
    #                 self.room_polygon.append(QPointF(corner_tx, corner_ty))
    #         return True
    #     else:
    #         print("Room node does not exist. g2o graph cannot be initialized")
    #         return False

    def get_displacement(self, odometry):
        """
        Calculates the displacement of a robot based on its odometry data, computing
        the lateral displacement, forward displacement, and angular displacement.

        Args:
            odometry (Tuple[float, float, float]): A sequence of 3-element tuples
                representing the robot's position, velocity, and timestamp.

        Returns:
            Tuple[float,float,float]: 3 components of displacement (lateral, avance
            and angular) calculated based on the odometry data.

        """
        desplazamiento_avance = 0
        desplazamiento_lateral = 0
        desplazamiento_angular = 0
        try:
            indice = next(index for index, (_, _, _,timestamp) in enumerate(self.odometry_queue) if timestamp == self.last_odometry[3])
        except StopIteration:
            self.last_odometry = odometry
            return desplazamiento_lateral, desplazamiento_avance, desplazamiento_angular
        # print("Index range", indice, len(self.odometry_queue) - 1)
        # Sumar las velocidades lineales entre el timestamp pasado y el actual
        for i in range(indice, len(self.odometry_queue)-1):
            # print("Diferencia tiempo actual y pasado:", self.odometry_queue[i + 1][3] - self.odometry_queue[i][3])

            desplazamiento_avance += self.odometry_queue[i + 1][0] * (self.odometry_queue[i + 1][3] - self.odometry_queue[i][3]) * 0.8
            desplazamiento_lateral += self.odometry_queue[i + 1][1] * (self.odometry_queue[i + 1][3] - self.odometry_queue[i][3]) *0.8
            desplazamiento_angular -= self.odometry_queue[i + 1][2] * (self.odometry_queue[i + 1][3] - self.odometry_queue[i][3]) *0.8 / 1000
        return desplazamiento_lateral, desplazamiento_avance, desplazamiento_angular

    def get_covariance_matrix(self, vertex):
        """
        Computes the covariance matrix of a set of vertices in a graph using the
        G2O optimizer and returns the result.

        Args:
            vertex (G2O.HessianIndex): Used to represent a vertex in the graph.

        Returns:
            Tuple[bool,npndarray]: A result of computing marginals and a covariance
            matrix.

        """
        cov_vertices = [(vertex.hessian_index(), vertex.hessian_index())]
        covariances, covariances_result = self.g2o.optimizer.compute_marginals(cov_vertices)
        if covariances_result:
            print("Covariance computed")
            matrix = covariances.block(vertex.hessian_index(), vertex.hessian_index())
            upper_triangle = np.triu(matrix)  # Use k=1 to exclude diagonal
            print("Covariance matrix", upper_triangle)
            return (covariances_result, covariances)
        else:
            print("Covariance not computed")
            return (covariances_result, None)

    def visualize_g2o_realtime(self, optimizer):
        """
        Loads G2O files, visualizes their positions and edges in 3D, and updates
        the display in real-time using matplotlib.

        Args:
            optimizer (object): Used to load a G2O file for visualization.

        """
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        while True:
            optimizer.load("archivo.g2o")  # Reemplaza "archivo.g2o" con tu archivo .g2o
            positions = []
            for vertex_id in range(optimizer.vertices().size()):
                vertex = optimizer.vertex(vertex_id)
                position = vertex.estimate()
                positions.append(position)
            positions = np.array(positions)

            ax.clear()
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', marker='o', label='Vertices')

            edges = optimizer.edges()
            for edge_id in range(edges.size()):
                edge = edges[edge_id]
                measurement = edge.measurement()
                ax.plot(measurement[:, 0], measurement[:, 1], measurement[:, 2], c='r')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()

            plt.draw()
            plt.pause(0.1)  # Pausa para permitir que la visualización se actualice

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        # pass
        # check if room node is created
        # room_node = self.g.get_node("room")
        # if room_node is not None and not self.init_graph:
        #     if id == room_node.id and "valid" in attribute_names:
        #         # self.init_graph = True
        #         print("INIT GRAPH")

        """
        Updates the attributes of a node in a graph, specifically the odometry
        node, and appends a new entry to the odometry queue based on the current
        robot position and speed.

        Args:
            id (int): Used to identify the node for which attributes are being
                updated, specifically the odometry node.
            attribute_names ([str]): An array of strings that represents the names
                of attributes to be updated on a node in the graph.

        """
        if id == self.odometry_node_id:
            odom_node = self.g.get_node("Shadow")
            odom_attrs = odom_node.attrs
            self.odometry_queue.append((odom_attrs["robot_current_advance_speed"].value, odom_attrs["robot_current_side_speed"].value, -odom_attrs["robot_current_angular_speed"].value, int(time.time()*1000)))
            # self.odometry_queue.append((odom_attrs["robot_ref_adv_speed"].value, odom_attrs["robot_ref_side_speed"].value, odom_attrs["robot_ref_rot_speed"].value, odom_attrs["timestamp_alivetime"].value))


    def update_node(self, id: int, type: str):
        # console.print(f"UPDATE NODE: {id} {type}", style='green')
        # if type == "corner":
        #     # check if there are 4 corners and room node
        #     corner_nodes = self.g.get_nodes_by_type("corner")
        #     room_node = self.g.get_node("room")
        #     if len(corner_nodes) == 8 and room_node:
        #         if not self.room_initialized:
        #             self.room_initialized = True if self.initialize_g2o_graph() else False
        #             self.init_graph = True
        """
        Updates an individual node's type, specifically "corner". If the node
        belongs to a room, it initializes the graph's room structure if necessary.

        Args:
            id (int): Used to identify the node being updated.
            type (str): Used to identify the node's type, which can be either
                "corner" or "room".

        """
        pass

    def delete_node(self, id: int):
        """
        Deletes a node from a graph data structure represented in the method
        signature, by setting the `room_initialized` attribute to `False`.

        Args:
            id (int): Intended to represent the unique identifier of the node to
                be deleted.

        """
        pass
        # if type == "room":
        # #         TODO: reset graph and wait until new room appears
        #     self.room_initialized = False


        # console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):
        """
        Updates the room ID and sets the current edge set based on node type and
        RT edge information.

        Args:
            fr (int): Representing the from node ID in the graph.
            to (int): Used to represent the ID of the node that follows the edge
                being updated.
            type (str): Used to specify the edge's type, which can be either
                "current" or "RT".

        """
        if type == "current" and self.g.get_node(fr).type == "room":
            self.room_initialized = False
            # Get number after last "_" in room name
            if self.actual_room_id is not None and self.actual_room_id != self.g.get_node(fr).attrs['room_id'].value:
                self.last_room_id = self.actual_room_id
            self.actual_room_id = self.g.get_node(fr).attrs['room_id'].value
            print("###########################################################")
            print("Room changed to", self.actual_room_id)
            print("###########################################################")
            self.current_edge_set = True
            return

        if type == "RT" and self.g.get_node(fr).type == "room" and self.g.get_node(to).name == "Shadow":
            rt_edge = self.g.get_edge(fr, to, type)
            if rt_edge.attrs['rt_translation'].agent_id != self.agent_id and time.time()-self.rt_set_last_time > self.rt_time_min:
                self.room_initialized = False
                self.translation_to_set = rt_edge.attrs['rt_translation'].value
                self.rotation_to_set = rt_edge.attrs['rt_rotation_euler_xyz'].value
                self.first_rt_set = True
                print("Translation to set", self.translation_to_set)
                print("Rotation to set", self.rotation_to_set)
                self.rt_set_last_time = time.time()

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        pass


    def delete_edge(self, fr: int, to: int, type: str):
        """
        Deletes an edge from a graph, specified by its index (fr), type (to), and
        the worker instance.

        Args:
            fr (int): 1st in the function signature, indicating that it should be
                the first argument passed to the function when calling it.
            to (int): Representing the second vertex index of an edge to be deleted.
            type (str): Used to specify the edge type to be deleted, which can be
                either "weighted" or "unweighted".

        """
        pass
        # console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
