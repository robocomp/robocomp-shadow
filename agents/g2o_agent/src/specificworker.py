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
    Manages a specific worker's tasks and updates in a graph. It has methods for
    updating node attributes, deleting nodes, and setting up RT transmission, as
    well as checking for updates and handling errors.

    Attributes:
        Period (float|int): Used to control the update rate of the graph. It
            determines how often the graph is updated, with a higher value resulting
            in more frequent updates.
        agent_id (int|str): Used to store the ID of the agent that will be using
            the specific worker.
        g (Graph|QGraphicsScene): Used to store the graph data for the worker's
            algorithm, including the nodes, edges, and their attributes.
        startup_check (QTimersingleShot200,QApplicationinstancequit): Used to check
            if the application is still running after a certain time period (200
            milliseconds) and quit if so.
        rt_api (Union[str,int]): Used to store the RT (Real-time) edge set ID for
            the worker's specific room.
        inner_api (Callable[[],None]): Used to define a custom API for the worker
            to interact with its internal data structures and algorithms without
            exposing them to the outer world.
        odometry_node_id (int): Used to identify the node in the graph that
            represents the robot's odometry information.
        odometry_queue (List[Tuple[float,float,float,int]]): Used to store odometry
            data from the environment, containing the robot's current advance
            speed, side speed, and angular speed, as well as the timestamp in
            milliseconds since the epoch.
        last_odometry (float|List[float]): Used to store the last known odometry
            values (position, orientation, and translation) of the robot, which
            are updated in real-time during the simulation.
        g2o (Graph|Tuple[str,str]): Used to represent the graph structure of the
            robot's environment. It contains a tuple of two elements: the first
            element is the name of the graph file, and the second element is the
            name of the node file.
        visualizer (Callable[[QWidget],None]): Used to create a QWidget instance
            for visualizing the robot's movement and orientation in real-time.
        odometry_noise_std_dev (float|int): 0.1 by default, representing the
            standard deviation of noise added to the odometry measurements for the
            robot's position, orientation, and angular velocity.
        odometry_noise_angle_std_dev (float|double): 0.2 by default, representing
            the standard deviation of angle noise in the odometry estimates. It
            affects how much the worker's estimate of the agent's position and
            orientation deviates from the true values due to random noise in the
            sensor readings.
        measurement_noise_std_dev (float|int): 0.1 by default, indicating the
            standard deviation of the noise present in the robot's measurements.
        last_room_id (int|str): Used to store the room ID of the previous room
            that the agent has entered before changing rooms, which is useful for
            determining when the agent has moved to a new room.
        actual_room_id (int|str): Used to store the current room ID of the agent,
            which is updated when the agent moves to a new room.
        elapsed (float|int): Used to store the time since the last call to the
            `update()` method, which is used to update the node's attributes.
        room_initialized (bool): Set to False when a room change is detected,
            indicating that the worker has not yet initialized its current room's
            graph. It is reset to True once the worker has successfully initialized
            the new room's graph.
        iterations (int|float): Used to keep track of the number of iterations
            (i.e., frames) that the worker has processed. It is updated with each
            frame processed by the worker, and can be used to control the speed
            of the worker or to implement various algorithms.
        hide (bool): Used to indicate whether the worker should be hidden or not,
            affecting its visibility in the graph.
        init_graph (bool): Used to indicate whether the graph has been initialized
            or not. It is set to True when the worker is initialized and False
            otherwise, allowing for proper handling of the graph and its nodes and
            edges in the update methods.
        current_edge_set (bool): Used to track whether an edge set has been computed
            for the current room id. It is set to True when the first RT edge set
            is computed and False otherwise, allowing the worker to avoid recomputing
            the edge set if it has already been computed for the same room id.
        first_rt_set (bool): Set to `True` when the RT translation and rotation
            values are being set for the first time, and `False` otherwise.
        translation_to_set (float|List[float]): Used to store the translation
            values set by the RT algorithm.
        rotation_to_set (float|int): A value representing the rotation of the robot
            to set its end effector at a specific location.
        room_polygon (List[float]): Used to store the vertices of a polygon that
            represents the room boundary in the environment.
        security_polygon (Union[int,List[float]]): Used to store the security
            polygon of the worker's workspace in 3D space. It is used to detect
            collisions between the worker's manipulator and objects in the environment.
        initialize_g2o_graph (void): Used to create a new graph for the given
            worker instance, which will be used to store the robot's position and
            orientation in the environment.
        rt_set_last_time (int|float): Used to store the last time a RT (Real-Time)
            edge was set for a specific agent ID. It is used in conjunction with
            the `rt_time_min` attribute to determine when to reset the RT translation
            and rotation values.
        rt_time_min (float): Defined as the minimum time duration between two
            consecutive RT sets in milliseconds. It is used to determine when to
            reset the translation and rotation values for the shadow agent during
            RT tracking.
        last_update_with_corners (int|bool): Used to keep track of the last time
            a node or edge update was performed with corner information. It is
            initially set to False, and its value changes when updates are performed
            with corners. Its purpose is to check if the worker has updated with
            corners before, so as to avoid unnecessary updates in the future.
        timer (float|int): Used to schedule a call to the `QApplication.instance().quit()`
            function after 2 seconds.
        compute (Callable[[],Any]): Used to define a function that computes the
            worker's task based on the graph and other attributes.
        update_node_att (Callable[int,[str]]): Used to update the attributes of a
            specific node in the graph based on its ID and the type of update. It
            takes two arguments: id, which is the ID of the node to be updated,
            and attribute_names, which is a list of attribute names to be updated.
        update_edge (Callable[int,str,str]): Used to update the attributes of an
            edge in the graph based on its type (current or RT translation) and
            the node it connects.
        update_edge_att (Tuple[str,]): Used to update the attributes of an edge
            in a graph based on its type and the name of the attribute.

    """
    def __init__(self, proxy_map, startup_check=False):
        """
        Initializes instance variables and sets up event handling for signals
        related to node and edge attributes, as well as a timer to compute the
        worker's output every `Period` seconds.

        Args:
            proxy_map (Dict[str, Any]): Used to store the mapping between the
                original node attributes and their proxies.
            startup_check (bool): Used to check if the graph has already been
                initialized before starting the worker's execution.

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
        Computes and updates the robot's position, orientation, and covariance
        matrix based on RT messages received from the environment.

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
                self.visualizer.update_graph(self.g2o)

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
        Initializes a Graph-Based Odometry (G2O) graph for a specific room by:
        1/ Retrieving node and edge information from a robot's sensor data.
        2/ Filtering out invalid nodes and edges based on their position and orientation.
        3/ Adding nominal corners and fixed poses to the G2O graph using the
        retrieved data.

        Returns:
            bool: True if the g2o graph is successfully initialized and False otherwise.

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
        Computes the displacement of a robot based on its odometry data, taking
        into account the advance, lateral movement, and angular velocity of the robot.

        Args:
            odometry (Tuple[float, float, float]): An immutable sequence containing
                the robot's odometry data at each time step, consisting of the
                linear displacement, lateral displacement, and angular velocity,
                respectively.

        Returns:
            Tuple[float,float,float]: The displacement along the x, y and z axes,
            respectively, calculated using the odometry data from a robot's sensors.

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
        Computes the covariance matrix of a set of vertices in a graph, using an
        optimization algorithm to compute the marginals of the vertices and then
        constructing the covariance matrix from the resulting upper triangle.

        Args:
            vertex (G2O.Vertex | G2O.HessianIndex): Used to compute the covariance
                matrix for a specific vertex in a graph.

        Returns:
            Tuple[bool,npndarray]: A result of computing covariance matrix and the
            actual computed matrix.

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
        Visualizes a real-time 3D graphical representation of a G2O optimization
        problem using Python's Matplotlib library.

        Args:
            optimizer (object): Used to load G2O files for visualization.

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
        Updates attributes of a node in a graph based on the current time and robot
        state, and appends the updated values to an odometry queue for later use.

        Args:
            id (int): Used to identify the node for which the attributes are being
                updated.
            attribute_names ([str]): An array of strings representing the names
                of attributes to be updated on the given node ID.

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
        Updates the node with the given ID based on the type parameter. If the
        type is "corner", it initializes a room graph and sets the `init_graph`
        attribute to `True`.

        Args:
            id (int): An identifier for the node to be updated.
            type (str): Used to specify the node's type, which can be either
                "corner" or any other valid value.

        """
        pass

    def delete_node(self, id: int):
        """
        Deletes a node from a list of nodes maintained by the worker.

        Args:
            id (int): Used to represent the unique identifier of the node to be deleted.

        """
        pass
        # if type == "room":
        # #         TODO: reset graph and wait until new room appears
        #     self.room_initialized = False


        # console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):
        """
        Updates the room ID and RT translation and rotation values based on the
        incoming edge type and node attributes.

        Args:
            fr (int): The index of the current edge being processed.
            to (int): Representing the index of the next node in the graph that
                is being processed.
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
        Deletes an edge from a graph, identified by its index (fr) and type (to).

        Args:
            fr (int): 0-based index of the edge to be deleted.
            to (int): Used to specify the target vertex for edge deletion.
            type (str): Used to specify the type of edge to delete, with possible
                values 'direct' or 'indirect'.

        """
        pass
        # console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
