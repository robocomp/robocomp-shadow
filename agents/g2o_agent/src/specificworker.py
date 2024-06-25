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
    1/ Initializes a graph and G2O optimizer for pose estimation using a ROS bag
    file as input.
    2/ Handles incoming odometry messages, updates the graph, and computes the
    covariance matrix of the vertices.
    3/ Visualizes the G2O graph in real-time using the `matplotlib` library.
    4/ Updates node and edge attributes based on the incoming odometry messages.
    5/ Deletes nodes and edges from the graph when necessary.

    Attributes:
        Period (float): Used to specify the time interval between successive
            executions of the worker's method, which in this case is the `update_node`,
            `update_edge`, `delete_node`, or `delete_edge` methods. It determines
            how often the worker will check for new nodes or edges to process and
            how long it will wait before checking again.
        agent_id (int): Used to store the unique identifier of the agent that the
            worker represents.
        g (SparseGraph): Used to store the graph of corners and edges in a room.
            It allows the worker to perform tasks such as tracking the position
            of objects in the room, computing the covariance matrix between pairs
            of vertices, and visualizing the graph in real-time.
        startup_check (QTimersingleShot): Used to check if the worker's function
            should be run repeatedly until the user quits the application.
        rt_api (str): A reference to a ROS node that implements the RT API, allowing
            the worker to interact with the robot's real-time state.
        inner_api (instance): Used to store a reference to the inner API of the
            worker, which is used to interact with the
            worker's internal state and perform actions such as transforming rooms
            or updating node attributes.
        odometry_node_id (int): Used to identify the node in the graph that
            corresponds to the odometry data, allowing the worker to access and
            process the relevant information from the robot's sensors.
        odometry_queue (3D): Used to store the odometry data of the robot in a
            queue. It is used by the worker's method `get_displacement` to compute
            the displacement of the robot.
        last_odometry (3D): Used to store the previous odometry information of the
            robot, which is used for computing the displacement and covariance matrix.
        g2o (3D): A graph representing the robot's pose and orientation in real-time,
            created using the G2O algorithm for SLAM. It is used to store the
            nominal corners of the room and their corresponding measurements from
            the sensor data.
        visualizer (instance): A reference to an object of type `Visualizer`. It
            is used to visualize the graph constructed by the worker during its execution.
        odometry_noise_std_dev (float): Used to store the standard deviation of
            the noise in the odometry measurements. It is used in the `get_displacement`
            method to compute the displacement between consecutive measurements
            and to determine if the odometry is reliable or not.
        odometry_noise_angle_std_dev (float): 0.1 by default, which represents the
            standard deviation of angle noise in the g2o graph. It helps to control
            the noise level in the optimization process.
        measurement_noise_std_dev (float): Used to specify the standard deviation
            of the noise added to the robot's measurements during optimization.
            It represents the level of uncertainty in the measurements, which can
            affect the accuracy of the pose estimate.
        last_room_id (int): Used to store the last room ID seen by the worker
            before it changed rooms. It is updated when the worker changes rooms,
            indicating that a new room has been entered.
        actual_room_id (int): Used to store the current room ID of the worker's
            graph. It keeps track of the room that the worker is currently exploring
            or working in, which can change as the worker moves around the space.
        elapsed (float): 0 by default. It represents the time elapsed since the
            worker was created, and it is updated every time a new task is assigned
            to the worker.
        room_initialized (bool): Set to False by default, indicating that the
            worker has not yet initialized the room graph. Once the worker initializes
            the room graph, this attribute becomes True.
        iterations (int): Used to keep track of the number of iterations performed
            by the worker. It is incremented each time a new iteration is started,
            and can be used to control the termination condition of the worker.
        hide (instance): Used to indicate whether the worker should be hidden or
            not. It is set to `True` by default, but can be modified by the user
            to hide the worker if desired.
        timer (QTimer): Used to schedule a function call every 200 milliseconds
            to check if the user wants to quit the application.
        compute (instance): Used to perform a specific computation on the robot's
            state, such as computing the covariance matrix of the robot's position
            and orientation. It returns a tuple containing the computed value and
            a boolean indicating whether the computation was successful.
        init_graph (bool): Used to indicate whether the graph has been initialized
            or not. It's set to true when the graph is first created and false
            otherwise, allowing the worker to skip unnecessary computations if the
            graph has already been initialized.
        translation_to_set (3D): Used to store the translation from the robot's
            current position to a predefined set position.
        rotation_to_set (3D): Set to the Euler angles representing the rotation
            from the base frame to the set frame of reference, used for visualizing
            the transformation between frames in a robotic system.
        room_polygon (numpyndarray): Used to store a list of QPointF objects
            representing the corners of a room as detected by the worker.
        initialize_g2o_graph (method): Used to initialize a Graph-based 6 DOF (G2O)
            graph for a specific robot task, such as room detection and navigation.
            It takes in various parameters such as the room node ID, edge list,
            and corner list, and initializes the G2O graph with the corresponding
            nominal corners and edges.
        update_node_att (void): Used to update the attributes of a specific node
            in the graph, based on the current odometry data. It is called every
            time new odometry data is received and updates the node's attributes
            accordingly.
        update_edge (void): Used to update the edge attributes when a new edge is
            added to the graph. It takes three arguments: the first node of the
            edge (fr), the second node of the edge (to), and the type of the edge
            (type).
        update_edge_att (edge): Used to update the attributes of an edge in a
            graph. It takes three arguments: `fr`, `to`, and `type`, which are the
            index of the edge, the destination node, and the edge type, respectively.
            The attribute updates the attributes of the edge based on the specified
            type and attribute names.

    """
    def __init__(self, proxy_map, startup_check=False):
        """
        Initializes a SpecificWorker object, setting up various components such
        as graphs, timers, and signal connections. It also performs startup checks
        and hides the object.

        Args:
            proxy_map (dict): Used to store the map of proxies for the agent.
            startup_check (bool): Used to perform a check during initialization
                if the graph has been modified after being created, and if so,
                whether to restart the worker or continue with the current state.

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
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)
            self.init_graph = False

            self.translation_to_set = None
            self.rotation_to_set = None

            self.room_polygon = None

            self.room_initialized = True if self.initialize_g2o_graph() else False

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
        In the `SpecificWorker` class processes robot odometry data and updates
        the graph using the G2O optimization algorithm. It also adds landmarks to
        the graph based on the robot's position and orientation.

        """
        if time.time() - self.elapsed > 1:
            print("Frame rate: ", self.iterations, " fps")
            self.elapsed = time.time()
            self.iterations = 0
        self.iterations += 1

        if self.room_initialized:
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
                                            [0.0, 0.0, 0.1]])

                self.g2o.add_odometry(adv_displacement,
                                            side_displacement,
                                            ang_displacement, odom_information)


                landmark_information = np.array([[1/self.measurement_noise_std_dev, 0.0],
                                                [0.0, 1/self.measurement_noise_std_dev]])

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
                                    # print("Vertex count ", self.g2o.vertex_count)
                                    # self.g2o.add_landmark(corner_edge_mat[0], corner_edge_mat[1], 1 * np.eye(2), pose_id=self.g2o.vertex_count-1, landmark_id=int(corner_node.name[7]))
                                    self.g2o.add_landmark(corner_edge_mat[0], corner_edge_mat[1], 0.05 * np.eye(2), pose_id=self.g2o.vertex_count-1, landmark_id=int(corner_node.name[7])+1)
                                    # self.g2o.add_landmark(self.add_noise(corner_edge_mat[0], self.measurement_noise_std_dev), self.add_noise(corner_edge_mat[1], self.measurement_noise_std_dev), landmark_information, pose_id=self.g2o.vertex_count-1, landmark_id=int(corner_node.name[7]))
                                    # print("Landmark added:", corner_edge_mat[0], corner_edge_mat[1], "Landmark id:", int(corner_node.name[7]), "Pose id:", self.g2o.vertex_count-1)
                else:
                    # Get last room node
                    room_node = self.g.get_node("room_" + str(self.actual_room_id))
                chi_value = self.g2o.optimize(iterations=100, verbose=False)

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

        elif self.init_graph and self.translation_to_set is not None and self.rotation_to_set is not None:
            print("Initializing g2o graph")
            if self.last_room_id is not None:
                self.g.delete_edge(self.g.get_node("room_"+str(self.last_room_id)).id, self.g.get_node("Shadow").id, "RT")
            self.initialize_g2o_graph()
            self.room_initialized = True
            self.init_graph = False
            self.translation_to_set = None
            self.rotation_to_set = None

    def add_noise(self, value, std_dev):
        # print("Value", value, "Noise", np.random.normal(0, std_dev))
        return value + np.random.normal(0, std_dev)

    def initialize_g2o_graph(self):
        """
        Initializes an G2O graph for a specific robot and room, adds fixed pose,
        and adds nominal corners based on RT information.

        Returns:
            bool: 1 if the g2o graph was successfully initialized and 0 otherwise.

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
        Calculates the displacement of an agent based on its odometry data, taking
        into account advancement, lateral movement, and angular movement.

        Args:
            odometry (3D): An instance of an odometry sequence containing timestamped
                measurements of vehicle displacement, velocity, and acceleration.

        Returns:
            3tuple: The displacement in the lateral direction, displacement in the
            forward
            direction and angular displacement

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
        Computes and returns the covariance matrix of a set of vertices in a graph,
        using an optimization algorithm to compute the marginals and then constructing
        the covariance matrix from the resulting upper triangle.

        Args:
            vertex (g2overtex): Used to represent a vertex in a graph.

        Returns:
            2tuple: A tuple containing two elements: a boolean value and a covariance
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
        Within the SpecificWorker class loads an G2O file, visualizes the vertices
        and edges, and updates the visualization in real-time as new data is available.

        Args:
            optimizer (object): Used to load an G2O file into the optimizer for
                visualization purposes.

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
        Updates the attributes of a node in a graph based on the `id` and
        `attribute_names` parameters. It appends a new value to an odometry queue
        for the specified node.

        Args:
            id (int): Representing an identifier for a specific node in the graph.
            attribute_names ([str]): A list of strings representing the names of
                attributes to be updated on the node.

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
        Updates a node with ID `id` based on its type, performing different actions
        depending on the type.

        Args:
            id (int): Passed as an argument to update a node's attributes.
            type (str): Used to identify the node type, specifically "corner".

        """
        pass

    def delete_node(self, id: int):
        """
        Deletes a node from a collection based on its ID, setting a flag to indicate
        if the delete was successful for a specific type of node.

        Args:
            id (int): Passed as an input to delete a node from the graph represented
                by the object.

        """
        if type == "room":
        #         TODO: reset graph and wait until new room appears
            self.room_initialized = False


        # console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):
        """
        Updates the room ID and initializes graph variables when a 'current' or
        'RT' edge is encountered, and sets translation and rotation values when
        an 'RT' edge is encountered with a 'Shadow' node on the other side.

        Args:
            fr (int): Representative of the first node's position in the graph.
            to (int): Used to represent the id of the node that follows the edge
                being updated.
            type (str): Used to specify the type of update being performed on the
                edge, either "current" or "RT".

        """
        if type == "current" and self.g.get_node(fr).type == "room":
            # Get number after last "_" in room name
            if self.actual_room_id is not None and self.actual_room_id != self.g.get_node(fr).attrs['room_id'].value:
                self.last_room_id = self.actual_room_id
            self.actual_room_id = self.g.get_node(fr).attrs['room_id'].value
            print("###########################################################")
            print("Room changed to", self.actual_room_id)
            print("###########################################################")
            self.init_graph = True

        if type == "RT" and self.g.get_node(fr).type == "room" and self.g.get_node(to).name == "Shadow":
            rt_edge = self.g.get_edge(fr, to, type)
            print(rt_edge.attrs['rt_translation'].agent_id, self.agent_id)
            if rt_edge.attrs['rt_translation'].agent_id != self.agent_id:
                self.translation_to_set = rt_edge.attrs['rt_translation'].value
                self.rotation_to_set = rt_edge.attrs['rt_rotation_euler_xyz'].value
                self.room_initialized = False
                print("Translation to set", self.translation_to_set)
                print("Rotation to set", self.rotation_to_set)

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        pass


    def delete_edge(self, fr: int, to: int, type: str):
        """
        Deletes an edge from a graph represented by the `SpecificWorker` class,
        specified by the `fr` and `to` integers and Edge type.

        Args:
            fr (int): Valued at 0 in the given code snippet.
            to (int): Used to specify the destination node ID for edge deletion.
            type (str): Used to specify the edge type to be deleted.

        """
        pass
        # console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
