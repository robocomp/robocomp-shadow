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
    Performs real-time visual odometry estimation for a robot using a G2O graph
    and RT communication. It initializes a graph, adds nominal corners, and updates
    edges with measured data from an RT communication agent.

    Attributes:
        Period (float): Used to control the interval at which the worker performs
            its tasks, with a default value of 1 second. It can be used to adjust
            the worker's execution speed.
        agent_id (int): A unique identifier assigned to each agent in the system.
        g (Graph): Used for representing the graph of a robot's motion, where each
            node represents a pose of the robot and each edge represents a motion
            of the robot between two poses. The graph is used to compute the
            marginal probability distribution of the robot's motion given its
            initial position and velocity.
        startup_check (QTimersingleShot): Used to start a quit timer after 200
            milliseconds of inactivity, which allows the worker to exit gracefully
            when no further work is available.
        rt_api (str): 8 characters long. It stores a string value that represents
            the ROS launch file path for the RT node.
        inner_api (instance): Used as a reference to the inner API object, which
            is responsible for handling the actual work of the worker.
        odometry_node_id (int): 0-based index of the node representing the robot's
            odometry information in the graph. It is used to identify the node in
            the graph that represents the robot's current position and velocity.
        odometry_queue (3D): A queue that stores the odometry data of the robot,
            which is used to update the graph and calculate the robot's pose.
        last_odometry (3D): Used to store the last known odometry value of the
            robot, which is used for calculating the displacement and covariance
            matrix.
        g2o (3D): A G2O graph, which is a data structure used to represent the
            robot's pose in a 3D environment. It stores the information of the
            robot's pose in a set of nodes and edges, where each node represents
            a location in space and each edge represents a connection between two
            locations. The `g2o` attribute is used to perform optimization tasks
            such as finding the robot's pose that minimizes a cost function.
        odometry_noise_std_dev (float): Used to represent the standard deviation
            of odometry noise in the worker's graph. It is used in the `get_displacement`
            method to compute the displacement between two poses based on the
            odometry noise.
        odometry_noise_angle_std_dev (floatingpoint): 1.5 by default, which
            represents the standard deviation of angle noise in the odometry
            measurement of a robot. It affects the accuracy of the robot's positioning
            and movement tracking.
        measurement_noise_std_dev (float): 10% of the robot's minimum distance
            from its reference configuration to its target position, indicating
            the standard deviation of measurement noise in the robot's pose estimation.
        last_room_id (int): Used to store the previous room ID that was visited
            by the agent before the current room. It is updated when the agent
            moves to a new room, allowing the worker to detect changes in the environment.
        actual_room_id (int): Used to store the current room ID of the agent, which
            is updated when the robot moves from one room to another.
        elapsed (int): Used to store the time elapsed since the last call to the
            `startup_check` function, which is called every 0.1 seconds. It is
            used to determine when to quit the application.
        room_initialized (bool): Used to keep track of whether the current room
            has been initialized or not. It's set to False when a new room is
            entered, and True otherwise.
        iterations (int): Used to keep track of the number of iterations performed
            by the worker. It increases by one each time a new iteration is performed
            and can be accessed or modified from within the worker's methods.
        hide (str): Used to specify whether a worker should be hidden or not when
            multiple workers are running simultaneously.
        init_graph (bool): Used to indicate whether the graph has been initialized
            or not. It is set to True when the graph is initialized and False
            otherwise, allowing the worker to distinguish between initialized and
            uninitialized graphs.
        current_edge_set (bool): Used to indicate whether the current edge set has
            been computed or not. It is set to true when the edge set is first
            computed and false otherwise, so that only new computations are performed.
        first_rt_set (bool): Set to `True` when a new RT translation and rotation
            are detected, and `False` otherwise. It indicates whether the worker
            has received its first RT set or not.
        translation_to_set (3D): Used to store the translation from the robot's
            reference frame to the set frame.
        rotation_to_set (3D): Used to store the rotation of a robot's end effector
            relative to its base frame, which is necessary for setting the desired
            orientation of the robot's end effector.
        room_polygon (numpyarray): Used to store the coordinates of a room's
            boundary in a list of (x,y) tuples.
        initialize_g2o_graph (instance): Used to initialize a Graph2ONode graph
            for a specific robotic arm. It creates nodes, edges, and defines the
            room node, and sets up the optimization algorithm and visualization tools.
        rt_set_last_time (float): Used to store the last time when a RT set was
            set by the worker, which is used to filter out unnecessary RT sets.
        rt_time_min (float): Set to the minimum time between two consecutive RT
            sets. It is used to determine when to reset the translation and rotation
            values for RT tracking purposes.
        timer (QTimer): Used to schedule periodic updates of the graph with the
            current robot state. It is set to single shot every 200 milliseconds,
            allowing for real-time updates of the graph.
        compute (instance): Used to compute the marginal probability distributions
            of the robot's state given its current observations. It takes an
            argument `vertex` which is a vertex in the graph representing the
            robot's state, and returns a tuple containing the marginal probabilities
            and the Hessian matrix of the robot's state.
        update_node_att (update): Called when a new node with the specified id is
            added to the graph, or an existing node's attributes are updated. It
            updates the odometry queue based on the new node's attributes and sets
            the room initialization flag accordingly.
        update_edge (edge): Used to update the attributes of an edge in a graph
            when the edge's node types change or when a new edge is added to the
            graph. It sets the `rt_translation` and `rotation_euler_xyz` attributes
            of the edge based on the new node types.
        update_edge_att (attribute): Used to update the attributes of an edge in
            a graph. It takes the edge ID, the new value for the attribute, and
            the name of the attribute as input parameters.

    """
    def __init__(self, proxy_map, startup_check=False):
        """
        Initializes a SpecificWorker instance, defining its attributes and connections
        to graphs, timers, and signal handlers.

        Args:
            proxy_map (dict): Used to store a mapping from agent IDs to graphs.
            startup_check (Optionalbool): Used to check if the robot's graph has
                already been initialized before starting the main loop.

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
        Generates high-quality summaries of Java code that is given to it. It takes
        into account the robot's odometry, computes the translation and rotation
        of the robot relative to a reference frame, and adds the information to
        the graph. It also updates the landmark in the graph.

        """
        if time.time() - self.elapsed > 1:
            print("Frame rate: ", self.iterations, " fps")
            self.elapsed = time.time()
            self.iterations = 0
        self.iterations += 1

        if self.room_initialized:
            self.first_rt_set = False
            print("Room initialized")
            # Get robot odometry
            if self.odometry_queue:
                init_time = time.time()
                robot_node = self.g.get_node("Shadow")
                room_nodes = [node for node in self.g.get_nodes_by_type("room") if self.g.get_edge(node.id, node.id, "current")]
                robot_odometry = self.odometry_queue[-1]
                # print("Robot odometry:", robot_odometry)
                time_1 = time.time()
                adv_displacement, side_displacement, ang_displacement = self.get_displacement(robot_odometry)
                # print("Time elapsed get_displacement:", time.time() - time_1)
                if len(self.g2o.pose_vertex_ids) == self.g2o.queue_max_len:
                    self.g2o.remove_first_vertex()
                # Generate information matrix considering the noise
                odom_information = np.array([[1, 0.0, 0.0],
                                            [0.0, 1, 0.0],
                                            [0.0, 0.0, 0.001]])

                self.g2o.add_odometry(adv_displacement,
                                            side_displacement,
                                            ang_displacement, odom_information)

                # Check if robot pose is inside room polygon
                if len(room_nodes) > 0:
                    room_node = room_nodes[0]
                    if self.room_polygon is not None:
                        robot_edge_rt = self.rt_api.get_edge_RT(room_node, robot_node.id)
                        robot_tx, robot_ty, _ = robot_edge_rt.attrs['rt_translation'].value
                        robot_point = QPointF(robot_tx, robot_ty)
                        if self.room_polygon.containsPoint(robot_point, Qt.OddEvenFill):
                            for i in range(4):
                                corner_node = self.g.get_node("corner_"+str(i)+"_measured")
                                is_corner_valid = corner_node.attrs["valid"].value
                                if is_corner_valid:
                                    corner_edge = self.rt_api.get_edge_RT(robot_node, corner_node.id)
                                    corner_edge_mat = self.rt_api.get_edge_RT_as_rtmat(corner_edge, robot_odometry[3])[0:3, 3]
                                    self.g2o.add_landmark(corner_edge_mat[0], corner_edge_mat[1], 0.05 * np.eye(2), pose_id=self.g2o.vertex_count-1, landmark_id=int(corner_node.name[7])+1)
                                    print("Landmark added:", corner_edge_mat[0], corner_edge_mat[1], "Landmark id:", int(corner_node.name[7]), "Pose id:", self.g2o.vertex_count-1)
                else:
                    # Get last room node
                    room_node = self.g.get_node("room_" + str(self.actual_room_id))
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
                # self.visualizer.update_graph(self.g2o)
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
            print("Initializing g2o graph")
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
        """
        Initializes a Graph2O graph for a specific robot and room, adding nominal
        corners and measuring them to create a polygonal representation of the
        room. It also sets the robot's initial position and orientation in the graph.

        Returns:
            bool: 1 if the g2o graph was successfully initialized with the given
            room id, and 0 otherwise.

        """
        print("Initializing g2o graph")
        room_nodes = [node for node in self.g.get_nodes_by_type("room") if self.g.get_edge(node.id, node.id, "current")]
        if len(room_nodes) > 0:
            self.g2o.clear_graph()
            room_node = room_nodes[0]
            if self.actual_room_id is not None and self.actual_room_id != room_node.attrs['room_id'].value:
                self.last_room_id = self.actual_room_id
            self.actual_room_id = room_node.attrs['room_id'].value
            print("###########################################################")
            print("INITIALIZ>INDºG G2O GRAPH")
            print("Room changed to", self.actual_room_id)
            print("###########################################################")

            # get robot pose in room
            robot_node = self.g.get_node("Shadow")
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

            # Get corner values from room node
            # corner_nodes = self.g.get_nodes_by_type("corner")
            # Order corner nodes by id

            corner_list = []
            corner_list_measured = []
            for i in range(4):
                corner_list.append(self.g.get_node("corner_"+str(i)+"_"+str(self.actual_room_id)))
                corner_list_measured.append(self.g.get_node("corner_"+str(i)+"_measured"))

            # Generate QPolygonF with corner values
            self.room_polygon = QPolygonF()

            for i in range(4):
                corner_edge_measured_rt = self.rt_api.get_edge_RT(robot_node, corner_list_measured[i].id)
                corner_measured_tx, corner_measured_ty, _ = corner_edge_measured_rt.attrs['rt_translation'].value
                corner_edge_rt = self.inner_api.transform(room_node.name, corner_list[i].name)
                corner_tx, corner_ty, _ = corner_edge_rt
                print("Nominal corners", corner_tx, corner_ty)
                print("Measured corners", corner_measured_tx, corner_measured_ty)
                landmark_information = np.array([[0.05, 0.0],
                                                 [0.0, 0.05]])
                print("Eye matrix", 0.1 * np.eye(2))
                print("Landmark information:", landmark_information)
                if corner_tx != 0.0 or corner_ty != 0.0:
                    # self.g2o.add_landmark(corner_tx, corner_ty, 0.1 * np.eye(2), pose_id=0)
                    self.g2o.add_nominal_corner(corner_edge_rt,
                                              corner_edge_measured_rt.attrs['rt_translation'].value,
                                              landmark_information, pose_id=0)
                    # Add corner to polygon
                    self.room_polygon.append(QPointF(corner_tx, corner_ty))
            return True
        else:
            print("Room node does not exist. g2o graph cannot be initialized")
            return False

    def get_displacement(self, odometry):
        """
        Calculates the displacement of a robot based on its odometry data, updating
        the robot's position and orientation using the lateral, angular, and linear
        displacements.

        Args:
            odometry (stdvectordouble): 3D vector representing the robot's position,
                orientation, and velocity at a given time stamp.

        Returns:
            triplet: 3 elements long (desplazamiento lateral, displacement avance
            and angular displacement).

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
        G2O optimization algorithm. It takes a vertex as input, returns the computed
        covariance matrix and a result flag indicating whether the computation was
        successful.

        Args:
            vertex (g2oVertex): Passed to compute marginals for computing covariance
                matrix.

        Returns:
            tuple: 2-element, containing the result of computing covariance matrix
            and the marginals of the vertices.

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
        Visualizes a G2O optimization problem in real-time, loading the problem
        from an file and plotting the vertices and edges as 3D scatter plots while
        updating the display with small intervals of time.

        Args:
            optimizer (instance): Used to load G2O files for visualization in real-time.

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
        Updates the attributes of a node in a graph, specifically the `odometry_queue`
        attribute, when the node's ID matches the `odometry_node_id`.

        Args:
            id (int): Used to identify the node for which attribute values are
                being updated.
            attribute_names ([str]): A list of attribute names to update in the node.

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
        Updates the node with the specified ID, based on the type of node it represents.

        Args:
            id (int): Used to identify the node being updated.
            type (str): Used to specify the node type.

        """
        pass

    def delete_node(self, id: int):
        """
        Deletes a node from the graph represented by the `SpecificWorker` class.

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
        Updates the room ID and sets the current edge set based on the provided
        edge type and node types.

        Args:
            fr (int): Representative of the start node of an edge in a graph.
            to (int): Used to represent the target node of an edge in the graph.
            type (str): Used to specify the type of edge to update, either "current"
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
        Deletes an edge from a graph, specified by the vertex indices `fr` and
        `to`, and the edge type.

        Args:
            fr (int): Used to represent the starting vertex index of an edge to
                be deleted.
            to (int): Used to indicate the destination node ID for edge deletion.
            type (str): Used to specify the edge type to be deleted, with possible
                values 'in' or 'out'.

        """
        pass
        # console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
