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
    Is responsible for processing robot poses and creating a graph representation
    of the environment for a specific worker in a multi-robot system, including
    handling edges, vertices, odometry, and room changes.

    Attributes:
        Period (instance): Used to specify the time interval between updates of
            the worker's state, allowing for more efficient processing of large datasets.
        agent_id (int): Used as a identifier for the agent that owns the specific
            worker.
        g (Graph): Used to store the graph representation of the environment, which
            is updated based on the worker's observations and actions.
        startup_check (QTimersingleShot): Used to check if the application should
            quit after a certain time has elapsed since startup. It's used to
            implement the main loop of the worker, where the worker periodically
            checks if it should quit based on the current time.
        rt_api (instance): Used to store the RT API object, which is used for
            real-time data processing and transmission.
        inner_api (instance): Used to store the inner API object of the worker,
            which allows for communication between the worker and the main thread.
        odometry_node_id (int): 3 in this example, indicating that the node with
            ID 3 represents the odometry information.
        odometry_queue (3D): A list of tuples containing the current pose of the
            robot, which is updated at each step by the worker. The queue stores
            the last `N` poses for each node in the graph, where `N` is a user-defined
            parameter.
        last_odometry (3D): Used to store the last known odometry of the robot,
            which can be used for visualization and debugging purposes.
        g2o (3D): Used to represent the pose (position, orientation, and scale)
            of a robot or other object in 3D space. It is used in conjunction with
            the `get_covariance_matrix` method to compute the covariance matrix
            of the robot's position and orientation.
        visualizer (instance): A reference to an instance of the `Visualizer`
            class, which provides a user interface for visualizing the robot's
            pose and other relevant data in real-time.
        odometry_noise_std_dev (float): 0.1 by default, which represents the
            standard deviation of the noise added to the odometry values during
            the simulation for more realistic results.
        odometry_noise_angle_std_dev (float): 0.8 by default, which represents the
            standard deviation of the angular noise in the odometry measurement.
            It affects the optimization process by controlling the spread of the
            angle values in the graph.
        measurement_noise_std_dev (float): Used to represent the standard deviation
            of noise in the robot's measurements. It is used in the worker's
            implementation to scale the measurement values to account for the noise
            in the data.
        last_room_id (int): Used to store the last room ID seen before the agent
            switched rooms, which is used to handle room changes during navigation
            tasks.
        actual_room_id (int): Used to keep track of the current room id that the
            worker is in, it's updated when the worker moves to a new room.
        elapsed (int): Used to track the elapsed time since the worker was created,
            which can be used to control the worker's execution time and prevent
            it from running for too long.
        room_initialized (bool): Used to track whether the robot's room has been
            initialized or not. It is set to False when the robot moves to a new
            room, and True otherwise.
        iterations (int): Used to count the number of iterations performed by the
            worker during its execution. It is incremented each time the worker
            processes a new iteration of the graph optimization problem.
        hide (str): Used to indicate whether the worker should be hidden or not,
            defaulting to False.
        init_graph (bool): Used to keep track of whether the graph has been
            initialized or not, during the processing of the ROS bag file. It's
            set to True when the graph is first constructed and False when the
            graph is reconstructed after a change in the environment.
        current_edge_set (bool): Used to indicate whether the current edge set has
            been updated or not. It's used to track the updates of the edges in real-time.
        first_rt_set (bool): Set to `True` when the worker receives its first RT
            (Real-time) edge from a "room" node, indicating that the worker should
            start tracking the RT translation and rotation for this room.
        translation_to_set (3D): Used to store the translation of a robot to a set
            point, calculated based on the RT messages it receives. It is updated
            in the `update_edge_att` method.
        rotation_to_set (3D): Used to store the rotation of the robot's end effector
            with respect to a set reference frame. It is updated whenever the
            robot's pose changes, and is used in the computation of the robot's
            covariance matrix.
        room_polygon (Polygon): Used to store the polygon representation of a room
            in the environment. It is used for collision detection and other
            purposes related to navigation and mapping.
        security_polygon (3D): Used to store the security polygon of the robot,
            which is a convex polytope used for collision detection and avoidance.
        initialize_g2o_graph (instance): Used to initialize the graph2onedges (G2O)
            graph during the initialization of the worker, allowing for faster
            computation of marginals in the optimize method.
        rt_set_last_time (int): Used to store the time when the first RT set was
            observed by the worker, for updating the translation and rotation of
            the robot to the set.
        rt_time_min (float): Defined as `self.rt_time_min = 10`. It represents the
            minimum time interval between two RT sets for the robot to be considered
            in the RT algorithm.
        timer (QTimer): Used to create a timer that triggers the worker's method
            to update the graph every 200 milliseconds.
        compute (lambda): Called when a task is assigned to the worker. It takes
            the task as input, processes it using a provided function, and returns
            the result as output.
        update_node_att (Python): Used to update the attributes of a node in the
            graph when its ID matches the specified ID.
        update_edge (update): Called whenever an edge's attributes change. It
            checks if the edge is a "room" edge and updates the
            "translation to set", "rotation to set", or "first RT set" variables
            based on the edge's type and
            attributes.
        update_edge_att (edge): Used to update the attributes of an edge in a graph.

    """
    def __init__(self, proxy_map, startup_check=False):
        """
        Initializes member variables and sets up event handling for signals related
        to the graph, nodes, edges, and attributes.

        Args:
            proxy_map (dict): Used to map agent id's to their corresponding DSRGraph
                objects.
            startup_check (bool): Used to determine whether to run the `startup_check`
                method or not when initializing the SpecificWorker object.

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
            self.visualizer = G2OVisualizer("G2O Graph")

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
        Updates the robot's position, orientations, and security polygon based on
        the odometry data and the RT graph.

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

                        door_nodes = [node for node in self.g.get_nodes_by_type("door") if not "pre" in node.name and
                                      node.name in self.g2o.objects]
                        # Iterate over door nodes
                        if self.security_polygon.containsPoint(robot_point, Qt.OddEvenFill):
                            for door_node in door_nodes:
                                is_door_valid = door_node.attrs["valid"].value
                                if is_door_valid:
                                    door_measured_rt = door_node.attrs["rt_translation"].value
                                    if door_measured_rt[0] != 0.0 or door_measured_rt[1] != 0.0:
                                        self.g2o.add_landmark(door_measured_rt[0], door_measured_rt[1], 0.05 * np.eye(2),
                                                              pose_id=self.g2o.vertex_count - 1,
                                                              landmark_id=self.g2o.objects[door_node.name])

                chi_value = self.g2o.optimize(iterations=50, verbose=False)

                last_vertex = self.g2o.optimizer.vertices()[self.g2o.vertex_count - 1]
                opt_translation = last_vertex.estimate().translation()
                opt_orientation = last_vertex.estimate().rotation().angle()
                # Substract pi/2 to opt_orientation and keep the number between -pi and pi
                if opt_orientation > np.pi:
                    opt_orientation -= np.pi
                elif opt_orientation < -np.pi:
                    opt_orientation += np.pi

                # print("Optimized translation:", opt_translation, "Optimized orientation:", opt_orientation)
                # cov_matrix = self.get_covariance_matrix(last_vertex)
                # print("Covariance matrix:", cov_matrix)
                self.visualizer.update_graph(self.g2o)
                    # rt_robot_edge = Edge(room_node.id, robot_node.id, "RT", self.agent_id)
                    # rt_robot_edge.attrs['rt_translation'] = [opt_translation[0], opt_translation[1], .0]
                    # rt_robot_edge.attrs['rt_rotation_euler_xyz'] = [.0, .0, opt_orientation]
                    # # rt_robot_edge.attrs['rt_se2_covariance'] = cov_matrix
                    # self.g.insert_or_assign_edge(rt_robot_edge)

                # self.rt_api.insert_or_assign_edge_RT(room_node, robot_node.id, [opt_translation[0], opt_translation[1], 0], [0, 0, opt_orientation])
                rt_robot_edge = Edge(robot_node.id, room_node.id, "RT", self.agent_id)
                rt_robot_edge.attrs['rt_translation'] = Attribute(np.array([opt_translation[0], opt_translation[1], .0],dtype=np.float32), self.agent_id)
                rt_robot_edge.attrs['rt_rotation_euler_xyz'] = Attribute(np.array([.0, .0, opt_orientation],dtype=np.float32), self.agent_id)
                self.g.insert_or_assign_edge(rt_robot_edge)
                self.last_odometry = robot_odometry   # Save last odometry
                # print("Time elapsed compute:", timfe.time() - init_time)

        elif self.first_rt_set and self.current_edge_set and self.translation_to_set is not None and self.rotation_to_set is not None:
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
        Initializes a Graph-based Object Recognition (G2O) graph for a specific
        room by:
        1/ Finding the corners of the room using ROS topics.
        2/ Adding nominal corners to the G2O graph.
        3/ Adding fixed poses to the G2O graph for the robot and doors in the room.

        Returns:
            bool: 1 if the initialization of the g2o graph was successful, and 0
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
            d = 300
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
        Calculates the displacement of an object based on its odometry data,
        computing the lateral displacement, forward displacement, and angular
        displacement using a moving window approach.

        Args:
            odometry (3element): Used to store the robot's motion information at
                each time step, including its linear displacement, lateral
                displacement, and angular displacement.

        Returns:
            3element: Displacement in three directions (lateral, advance and
            angular) calculated using the odometry data from a queue.

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
        Computes the covariance matrix for a given vertex in a graph, using an
        optimization algorithm to compute the marginals of the vertices and then
        constructing the covariance matrix.

        Args:
            vertex (g2overtex): Used to compute the covariance matrix of the
                vertices in the graph.

        Returns:
            tuple: 2-element, where the first element is a boolean value indicating
            whether the covariance matrix was computed or not and the second element
            is the actual covariance matrix.

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
        Within the `SpecificWorker` class loads an G2O file, visualizes its vertices
        and edges in 3D, and updates the visualization in real-time as the optimizer
        processes new measurements.

        Args:
            optimizer (instance): Used to load G2O files.

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
        node, by appending values to an odometry queue.

        Args:
            id (int): Used to represent the node ID of interest for updating attributes.
            attribute_names ([str]): An array of strings representing the names
                of attributes to be updated on the node.

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
        Updates an unknown node with the specified ID, performing different actions
        based on the type of node it represents.

        Args:
            id (int): Used to identify the node being updated.
            type (str): Used to specify the node's type, with possible values being
                "corner".

        """
        pass

    def delete_node(self, id: int):
        """
        Deletes a node from the graph represented by the `SpecificWorker` class,
        setting the `room_initialized` attribute to `False`.

        Args:
            id (int): Used to identify the node to be deleted.

        """
        pass
        # if type == "room":
        # #         TODO: reset graph and wait until new room appears
        #     self.room_initialized = False


        # console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):
        """
        Updates the room ID and sets the current edge set to true when the type
        is "current" or "RT" and the source node is a room and the target node is
        "Shadow". It also sets the translation and rotation to set variables when
        RT is set for the first time.

        Args:
            fr (int): Used to represent the ID of the current node being processed
                in the graph.
            to (int): The id of the next node in the graph that the edge is pointing
                to.
            type (str): Used to determine which room the edge updates are for,
                either the current room or RT (room to Shadow)

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
        Deletes an edge from a graph with specified source and target nodes, based
        on the type of edge.

        Args:
            fr (int): Given a value of 13.
            to (int): Specified as an index of an edge to be deleted.
            type (str): Used to specify the edge type to be deleted.

        """
        pass
        # console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
