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
    Is a part of a larger system for real-time SLAM (Simultaneous Localization and
    Mapping) in a robotic environment. It provides methods for updating the graph,
    computing the covariance matrix, and visualizing the g2o graph. Additionally,
    it handles the startup check, update edge attribute names, update edge set,
    delete edge set, and delete node set.

    Attributes:
        Period (QTimersingleShot): Used to set the interval between successive
            calls to the `startup_check` function, which is responsible for checking
            if the application is still running after a certain period of time.
        agent_id (int): Used to store the ID of the agent that the worker represents.
        g (Graph): Used to represent the graph structure of the environment, which
            is essential for the worker's tasks such as edge measurement and room
            initialization.
        startup_check (QTimersingleShot): Used to check if the main thread is still
            running after a certain period of time (200 milliseconds) and quit the
            application if it is.
        rt_api (str): Used to specify the API for the RT (Real-Time) module of the
            worker, which controls the interaction between the worker and the
            robot's RT system.
        inner_api (instance): A reference to the inner API of the worker's robot,
            allowing access to its methods and attributes for communication with
            other components of the system.
        odometry_node_id (int): 0-based indexed, which represents the node ID of
            the odometry node in the graph. It's used to identify the node in the
            graph that corresponds to the robot's current pose.
        odometry_queue (3element): Used to store odometry data points from the ROS
            bag file, which are used to compute the nominal corners of the room.
        last_odometry (3D): Used to store the last known odometry of the robot,
            which is used to compute the covariance matrix of the robot's pose.
            It is updated every time a new odometry measurement is received from
            the robot.
        g2o (Optimizer): Used to configure the G2O optimization algorithm for
            real-time SLAM. It enables the worker to interact with the G2O optimizer
            and perform operations such as setting the initial guess, adding edges,
            and updating node attributes.
        visualizer (instance): A reference to an instance of the `Visualizer`
            class, which is used for visualizing the graph and its evolution during
            the optimization process.
        odometry_noise_std_dev (float): 1.0 by default, representing the standard
            deviation of the noise in the odometry measurements. It is used to
            scale the odometry readings for better convergence of the optimization
            algorithm.
        odometry_noise_angle_std_dev (float): Representing the standard deviation
            of the angle noise in the odometry measurements. It affects how much
            the robot's pose is likely to deviate from its intended path due to
            noisy angle measurements.
        measurement_noise_std_dev (float): Used to represent the standard deviation
            of measurement noise in the robot's odometry readings. It helps to
            quantify the uncertainty in the robot's position and orientation estimates.
        last_room_id (int): Used to keep track of the room ID that was last processed
            by the worker before switching rooms. It is updated when the worker
            moves to a new room, so it can be used to determine which room the
            worker is currently processing.
        actual_room_id (int): Used to keep track of the current room ID during a
            navigation task, updating it whenever the worker moves to a new room
            through its `update_node` method.
        elapsed (int): Updated every time a task is processed by the worker,
            indicating the elapsed time since the worker was created or since the
            last task was processed.
        room_initialized (bool): Set to False when the room ID changes, indicating
            that the worker has not yet initialized the new room. It is updated
            to True after the worker initializes the new room.
        iterations (int): 0 by default. It represents the number of iterations to
            perform during the optimization process. Each iteration consists of
            solving the optimization problem for a subset of the edges in the
            graph, and the value of this attribute determines how many times the
            optimization algorithm will execute. A higher value of `iterations`
            can lead to a more accurate solution, but also increases the computational
            time required for the optimization process.
        hide (attribute): Used to hide or show specific parts of the worker's
            functionality, such as the `visualize_g2o_realtime`, `update_node_att`,
            and `delete_edge` methods.
        timer (QTimer): Used to schedule a call to the `startup_check` function
            every 200 milliseconds to check if the application should quit.
        compute (instance): Used to compute the marginals of the graph, which are
            then used for pose estimation.
        init_graph (Python): Utilized to track whether the graph has been initialized
            or not by the worker. It is set to True when the
            worker finishes initializing the graph and False when it has not. This
            feature helps ensure that
            the robot's actions are taken in light of the most recent information
            about its environment
        current_edge_set (bool): Used to keep track of whether the edge set has
            been updated with the latest room information. It is initialized as
            True initially and is updated to False when the edge set is updated
            with new room information, allowing the worker to stop processing edges
            and move on to other tasks.
        first_rt_set (Python): Set to `True` when the first RT translation, rotation,
            or both are received from the G2O graph. It is used to keep track of
            whether the RT data has been received for the first time.
        translation_to_set (3D): Used to store the translation of a specific RT
            edge set from a reference frame to a target frame. It represents the
            difference in position between the two frames.
        rotation_to_set (3D): Used to store the rotation of the edge from its
            original position to the desired position for g2o optimization.
        room_polygon (list): Used to store a list of vertices that represent the
            boundary of a virtual room in the robot's workspace, as perceived by
            the worker's sensor. The vertices are represented by (x, y) coordinates
            in the workspace.
        initialize_g2o_graph (instance): Used to initialize a graph structure with
            the g2o library, which is used for visualizing and optimizing the
            robot's path. It takes no arguments and returns no value.
        update_node_att (QMetaObjectMetaCallable): Used to update the attributes
            of a node in the graph when it's a room node. It takes as arguments
            the node ID, the attribute names, and the new values for each attribute.
        update_edge (QMetaObjectPropertyChange): Used to update the edge attributes
            based on the ROS messages received by the worker. It handles updates
            for current, RT, and nominal edges and sets the translation and rotation
            towers to their corresponding values in the ROS message.
        update_edge_att (QMetaObjectPropertyChange): Used to update the attributes
            of an edge in a graph based on its type (current or RT). It takes the
            edge's ID, type, and attribute names as inputs and updates the
            corresponding attributes in the graph.

    """
    def __init__(self, proxy_map, startup_check=False):
        """
        Initializes an object of the SpecificWorker class, setting up various
        components such as graphs, nodes, edges, and timers. It also sets properties
        like period, odometry noise, and room ID.

        Args:
            proxy_map (dict): Passed to the superclass's constructor, indicating
                that the SpecificWorker class will use this map as its proxy map
                for accessing the graph.
            startup_check (bool): Used to check if the G2O graph has already been
                initialized or not.

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

            self.current_edge_set = False
            self.first_rt_set = False

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
        the graph using G2O optimization. It adds landmarks to the graph based on
        the robot's movement and computes the RT edges between nodes.

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
                                    print("Landmark added:", corner_edge_mat[0], corner_edge_mat[1], "Landmark id:", int(corner_node.name[7]), "Pose id:", self.g2o.vertex_count-1)
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

        elif self.first_rt_set and self.current_edge_set and self.translation_to_set is not None and self.rotation_to_set is not None:
            print("Initializing g2o graph")
            if self.last_room_id is not None:
                self.g.delete_edge(self.g.get_node("room_"+str(self.last_room_id)).id, self.g.get_node("Shadow").id, "RT")
            self.initialize_g2o_graph()
            self.room_initialized = True
            self.first_rt_set = False
            self.current_edge_set = False
            self.translation_to_set = None
            self.rotation_to_set = None

    def add_noise(self, value, std_dev):
        # print("Value", value, "Noise", np.random.normal(0, std_dev))
        return value + np.random.normal(0, std_dev)

    def initialize_g2o_graph(self):
        """
        Initializes a G2O graph for a specific room based on the robot's odometry
        and RoomTune API information. It adds nominal corners to the graph and
        calculates the room polygon.

        Returns:
            bool: 1 if the g2o graph could be initialized successfully, and 0 otherwise.

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
        Calculates the displacement of an object based on its odometry data, taking
        into account acceleration, lateral movement, and angular velocity.

        Args:
            odometry (3dimensional): An enumeration of odometry values recorded
                by the agent over time.

        Returns:
            3element: The lateral displacement (in meters), the advance displacement(in
            meters) and the angular displacement (in radians).

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
        G2O optimizer. It returns the computed covariance matrix and an indication
        of whether the computation was successful.

        Args:
            vertex (g2overtexVertex): Used to compute the covariance matrix for a
                specific vertex in the graph.

        Returns:
            2tuple: A tuple containing two elements: (1) a boolean value indicating
            whether the covariance matrix was computed successfully, and (2) the
            covariance matrix itself.

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
        Within the SpecificWorker class loads G2O data from an archivo, visualizes
        the vertices and edges in a 3D scatter plot with colors and markers, and
        updates the plot in real-time as new data is loaded.

        Args:
            optimizer (Optimizer): Used to load a G2O file and estimate its vertices
                positions in real-time.

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
        node, by appending a new value to an internal queue for further processing.

        Args:
            id (int): Used to represent the unique identifier of the node being updated.
            attribute_names ([str]): An array of strings that represents the names
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
        Updates a node with ID `id` based on its type, performing different actions
        depending on the type.

        Args:
            id (int): Passed as an argument to update the node with the given id.
            type (str): Passed as "corner"

        """
        pass

    def delete_node(self, id: int):
        """
        In the `SpecificWorker` class deletes a node from the graph represented
        by the class.

        Args:
            id (int): Passed as an argument to delete a node from the tree.

        """
        pass
        # if type == "room":
        # #         TODO: reset graph and wait until new room appears
        #     self.room_initialized = False


        # console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):
        """
        Updates the current room ID and translation/rotation to set for an RT agent
        based on the received edge message. It also sets the `room_initialized`
        flag and prints debug messages.

        Args:
            fr (int): Representative of an internal node index. It plays a role
                in identifying the particular edge being updated.
            to (int): A target node index indicating the destination node for the
                edge to be updated.
            type (str): Either "current" or "RT". It indicates the type of edge
                to be updated.

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
            if rt_edge.attrs['rt_translation'].agent_id != self.agent_id:
                self.room_initialized = False
                self.translation_to_set = rt_edge.attrs['rt_translation'].value
                self.rotation_to_set = rt_edge.attrs['rt_rotation_euler_xyz'].value
                self.first_rt_set = True
                print("Translation to set", self.translation_to_set)
                print("Rotation to set", self.rotation_to_set)

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        pass


    def delete_edge(self, fr: int, to: int, type: str):
        """
        Deletes an edge from a graph represented as an adjacency list, identified
        by its source and destination vertices and the edge type.

        Args:
            fr (int): Used as the first index of the edge to be deleted.
            to (int): The index of the edge to be deleted after it has been selected
                from the range
                [fr, to).
            type (str): Used to specify the type of edge being deleted, with
                possible values 'in' or 'out'.

        """
        pass
        # console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
