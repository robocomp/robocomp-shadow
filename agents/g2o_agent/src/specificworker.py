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
    Manages a specific robot's state and updates its graph, edges, and attributes
    based on messages from the occupancy grid. It also handles room changes and
    RT sets for the specified robot.

    Attributes:
        Period (float|int): 200 by default, indicating the time interval (in
            milliseconds) between updates of the graph. It can be modified to
            adjust the update rate of the worker.
        agent_id (int|str): A unique identifier for the agent, which can be used
            to identify specific agents within the system.
        g (Graph|QGraphicsScene): Used to store and manipulate the graph data structure.
        startup_check (QTimersingleShot200,QApplicationinstancequit): Set to call
            the quit function of the QApplication instance after a delay of 200 milliseconds.
        rt_api (Union[int,str]): Used to store the API for the Robot Translation
            (RT) functionality in the worker. It can take the value of either an
            integer representing the RT algorithm or a string representing the RT
            algorithm name.
        inner_api (Dict[str,Any]): Used to store additional internal data for the
            worker, such as the current room ID or the first RT set. It is only
            accessible within the worker's implementation and is not part of the
            public API.
        odometry_node_id (int): Used to identify the node in the graph that
            represents the robot's position and orientation.
        odometry_queue (List[Tuple[float,float,float,int]]): Used to store the
            latest odometry data from the robot's sensors for graph updating.
        last_odometry (float|List[float]): Used to store the last known odometry
            information of the robot, including its position, velocity, and angular
            velocity, which are updated every 200 milliseconds.
        g2o (Graph|npndarray): Used to represent the graph of the environment,
            allowing the worker to perform operations on it such as adding or
            deleting nodes and edges.
        odometry_noise_std_dev (float|int): Used to represent the standard deviation
            of noise in the odometry measurements. It determines how much the
            actual robot position, orientation, and velocity are expected to deviate
            from their predicted values based on the odometry measurements.
        odometry_noise_angle_std_dev (float|int): 1.0 by default, which represents
            the standard deviation of the noise angle in the odometry measurement.
            It helps to quantify the uncertainty in the estimated angle of the
            robot's movement.
        measurement_noise_std_dev (float|int): Used to represent the standard
            deviation of measurement noise in the robot's odometry readings. It
            is used to estimate the uncertainty of the robot's position and velocity.
        last_room_id (str|int): Used to store the last room ID seen by the agent
            before changing rooms, used for updating the RT set.
        actual_room_id (str|int): Used to store the current room ID of the agent,
            which is updated when the agent moves to a new room.
        elapsed (float|int): Used to keep track of the time since the last call
            to the `update()` method, which is used to control the frequency of
            updates to the graph.
        room_initialized (bool): Set to False when a room change is detected,
            indicating that the worker has switched rooms. It is then reset to
            True once the new room's ID is determined.
        iterations (int|List[int]): Used to keep track of the number of iterations
            of the worker's function that have been performed, or the list of
            iteration numbers if the function is called multiple times with different
            inputs.
        hide (bool): Used to hide the worker from the graphical user interface
            (GUI) when set to True.
        init_graph (bool): Used to track whether the graph has been initialized
            by the worker during its lifetime. It is used to prevent unnecessary
            work from being performed when the graph has already been initialized.
        current_edge_set (bool): Used to indicate whether the current edge being
            processed is part of a RT set or not. It is set to True when a new RT
            edge is encountered, and False otherwise.
        first_rt_set (bool): Set to True when the robot performs its first RT
            (Real-time) movement, indicating that the worker has started tracking
            RT movements.
        translation_to_set (float|List[float]): Used to store the translation value
            to set for the shadow robot. The list stores the values of the translation
            in the x, y, and z dimensions respectively.
        rotation_to_set (float|int): Used to store the rotation of the robot to
            set its position relative to its starting point.
        room_polygon (Tuple[float,float,float,]): Used to store the coordinates
            of a room's polygon in a graph.
        security_polygon (Polygon|List[Point]): Used to store the security polygon
            of a room, which is used for collision detection and avoidance during
            robot navigation.
        initialize_g2o_graph (void|QTimersingleShot200,QApplicationinstancequit):
            Used to initialize a Graph2O graph for representing the environment.
        rt_set_last_time (int|float): Used to track the time since the last RT
            (Room To) set by the agent. It is used to determine when to reset the
            translation and rotation to set values.
        rt_time_min (float|int): Defined as a minimum time gap between two successive
            RT edge sets, used to determine when to update the translation and
            rotation values for the agent.
        last_update_with_corners (int|bool): Used to keep track of when the worker
            has last updated its state with corners data. It is initially set to
            True, indicating that the worker has not yet received any corners data,
            and then is updated to False whenever the worker receives new corners
            data.
        timer (int|float): Used to track the time elapsed since the last update
            of the graph, with the purpose of checking if it's been a certain
            amount of time (200ms) since the last update, and if so, quit the application.
        compute (Callable[[float,float],float]): Used to compute the next node ID
            for the worker's node. It takes two arguments: the current time and a
            seed value, and returns the next node ID as a float value.
        update_node_att (Tuple[int,str,int,int]): Used to update the attributes
            of a node in the graph when the node's ID matches the given ID. The
            attribute takes four arguments: the node ID, the attribute names, the
            old value, and the new value.
        update_edge (Callable[float,float,str]): Used to update the edge attributes
            of a graph based on specific conditions. It takes three arguments: the
            first is the source node ID, the second is the target node ID, and the
            third is the edge type. The attribute is called whenever a new edge
            is added or an existing edge is modified in the graph, and it can be
            used to set or update edge attributes based on specific conditions.
        update_edge_att (List[str]): Used to update the attributes of an edge based
            on a specific type and name. It takes three parameters: the first is
            the id of the edge, the second is the type of edge, and the third is
            a list of attribute names to be updated.

    """
    def __init__(self, proxy_map, startup_check=False):
        """
        Initializes the worker's internal state, including its graph, odometry
        queue, and other components essential for its operation.

        Args:
            proxy_map (Dict[str, Any]): Used to pass additional data to the
                SpecificWorker object.
            startup_check (bool): Used to check if the graph has been properly
                initialized before starting the worker's computation. If set to
                `False`, it will skip this check and proceed with the computation.

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
        Processes robot odometry data and updates the graph using G2O optimization
        to estimate the robot's position and orientation.

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
        1) initializes the g2o graph based on room and robot nodes, 2) adds nominal
        corners for rooms and doors, and 3) fixes the pose of a robot node using
        odometry information.

        Returns:
            bool: 1 if the g2o graph is successfully initialized with nominal
            corners and fixed poses, and 0 otherwise.

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
        Calculates the displacement of the robot in three dimensions (lateral,
        forward, and angular) based on the odometry data stored in a queue.

        Args:
            odometry (Tuple[float, float, float]): A sequence of 3-element tuples
                representing the robot's position, velocity, and timestamp in a
                time interval.

        Returns:
            Tuple[float,float,float]: 3 floating-point values representing the
            lateral displacement, forward displacement, and angular displacement
            of an object over a given time interval.

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
        Computes the covariance matrix between a given vertex and all other vertices
        in the graph, using the G2O optimizer to compute marginals and upper
        triangle matrix representation.

        Args:
            vertex (G2O.Vertex | G2O.HessianIndex): Used to compute the covariance
                matrix for a specific vertex in the graph.

        Returns:
            numpyndarray: 2-dimensional and upper-triangular matrix representing
            the covariance matrix between a given vertex and all other vertices
            in the graph.

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
        Visualizes the real-time optimization process of a G2O algorithm by plotting
        the positions and edges of the vertices in 3D space.

        Args:
            optimizer (object | Optimizer): Used to load G2O files, estimate vertex
                positions, and visualize them in real-time.

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
        Updates the attributes of a node in a graph, specifically the odometry queue.

        Args:
            id (int): Represented as an integer value that identifies the node to
                be updated.
            attribute_names ([str]): An array containing the names of attributes
                to be updated for the given node ID.

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
        Updates a node with ID `id`. The method checks if the node type is "corner",
        and if so, it retrieves the room node from the graph and sets a flag
        indicating that the room has been initialized.

        Args:
            id (int): Used to identify the node to be updated.
            type (str): Used to specify the node's type.

        """
        pass

    def delete_node(self, id: int):
        """
        Deletes a node from a list or dict based on its ID, setting `self.room_initialized`
        to `False`.

        Args:
            id (int): Intended to represent the unique identifier for the node
                being deleted.

        """
        pass
        # if type == "room":
        # #         TODO: reset graph and wait until new room appears
        #     self.room_initialized = False


        # console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):
        """
        Updates the room ID and sets the current edge set based on the type of
        edge received from the graph.

        Args:
            fr (int): Representing the starting node index of an edge in a graph.
            to (int): The target node index of the edge to be updated, which can
                be either a room or the Shadow node.
            type (str): Used to specify the edge type, which can be either "current"
                or "RT".

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
        Deletes an edge from a graph, specified by its index (fr) and the index
        of its adjacent vertex (to). The edge's type can be specified for further
        filtering.

        Args:
            fr (int): 1st argument or input of the function indicating the first
                vertex or node to delete an edge from.
            to (int): Used to specify the destination vertex ID for the edge
                deletion operation.
            type (str): Used to indicate the edge type to be deleted, with possible
                values 'in' or 'out'.

        """
        pass
        # console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
