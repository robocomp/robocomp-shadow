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

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

from pydsr import *


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        """
        Initializes a SpecificWorker instance by setting up its internal graphs,
        queues, and connections to signals for updates and deletions.

        Args:
            proxy_map (dict): 2D map that the worker will operate on, providing
                it to the `super()` method as the first argument before calling
                any other constructor methods.
            startup_check (bool): check if the worker should execute its code.

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
            # self.room_initialized = True if self.initialize_g2o_graph() else False
            # time.sleep(2)
            self.room_initialized = False
            self.iterations = 0
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)
            self.init_graph = False

        try:
            signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.update_node_att)
            signals.connect(self.g, signals.UPDATE_NODE, self.update_node)
            # signals.connect(self.g, signals.DELETE_NODE, self.delete_node)
            # signals.connect(self.g, signals.UPDATE_EDGE, self.update_edge)
            # signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.update_edge_att)
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
        Performs pose graph optimization for a RT-based SLAM system using the
        Graph2O framework. It initializes the G2O graph, adds odometry data as
        landmarks, optimizes the graph using a Levenberg-Marquardt algorithm, and
        updates the visualizer with the optimized translation and orientation.

        """
        if self.room_initialized:
            # Get robot odometry
            if self.odometry_queue:
                init_time = time.time()
                robot_node = self.g.get_node("Shadow")
                room_node = self.g.get_node("room")
                robot_odometry = self.odometry_queue[-1]
                # print("Robot odometry:", robot_odometry)
                time_1 = time.time()
                adv_displacement, side_displacement, ang_displacement = self.get_displacement(robot_odometry)
                # print("Time elapsed get_displacement:", time.time() - time_1)
                if len(self.g2o.pose_vertex_ids) == self.g2o.queue_max_len:
                    self.g2o.remove_first_vertex()
                # Generate information matrix considering the noise
                odom_information = np.array([[10/self.odometry_noise_std_dev, 0.0, 0.0],
                                            [0.0, 10/self.odometry_noise_std_dev, 0.0],
                                            [0.0, 0.0, 1/self.odometry_noise_angle_std_dev]])
                # print("Odometry information:", odom_information)



                # self.g2o.add_odometry(0.0,
                #                             0.0,
                #                             0.0, 0.2 * np.eye(3))
                self.g2o.add_odometry(adv_displacement,
                                            side_displacement,
                                            ang_displacement, odom_information)


                landmark_information = np.array([[1/self.measurement_noise_std_dev, 0.0],
                                                [0.0, 1/self.measurement_noise_std_dev]])
                # print("Landmark information:", landmark_information)

                for i in range(4):
                    print("Corner", i, "....................................................................................")
                    corner_node = self.g.get_node("corner_"+str(i)+"_measured")
                    is_corner_valid = corner_node.attrs["valid"].value
                    if is_corner_valid:
                        corner_edge = self.rt_api.get_edge_RT(robot_node, corner_node.id)
                        corner_edge_mat = self.rt_api.get_edge_RT_as_rtmat(corner_edge, robot_odometry[3])[0:3, 3]
                        # print("Vertex count ", self.g2o.vertex_count)
                        # self.g2o.add_landmark(corner_edge_mat[0], corner_edge_mat[1], 1 * np.eye(2), pose_id=self.g2o.vertex_count-1, landmark_id=int(corner_node.name[7]))
                        self.g2o.add_landmark(corner_edge_mat[0], corner_edge_mat[1], 1 * np.eye(2), pose_id=self.g2o.vertex_count-1, landmark_id=int(corner_node.name[7])+1)
                        # self.g2o.add_landmark(self.add_noise(corner_edge_mat[0], self.measurement_noise_std_dev), self.add_noise(corner_edge_mat[1], self.measurement_noise_std_dev), landmark_information, pose_id=self.g2o.vertex_count-1, landmark_id=int(corner_node.name[7]))
                        print("Landmark added:", corner_edge_mat[0], corner_edge_mat[1], "Landmark id:", int(corner_node.name[7]), "Pose id:", self.g2o.vertex_count-1)
                time_2 = time.time()
                chi_value = self.g2o.optimize(iterations=100, verbose=False)
                # print("Time elapsed optimize:", time.time() - time_2)
                # print("Chi value:", chi_value)
                time_3 = time.time()

                # print("Time elapsed update_graph:", time.time() - time_3)

                last_vertex = self.g2o.optimizer.vertices()[self.g2o.vertex_count - 1]
                opt_translation = last_vertex.estimate().translation()
                opt_orientation = last_vertex.estimate().rotation().angle()
                # Substract pi/2 to opt_orientation and keep the number between -pi and pi
                if opt_orientation > np.pi:
                    opt_orientation -= np.pi
                elif opt_orientation < -np.pi:
                    opt_orientation += np.pi

                print("Optimized translation:", opt_translation, "Optimized orientation:", opt_orientation)
                cov_matrix = self.get_covariance_matrix(last_vertex)
                # print("Covariance matrix:", cov_matrix)
                self.visualizer.update_graph(self.g2o, None, cov_matrix)
                    # rt_robot_edge = Edge(room_node.id, robot_node.id, "RT", self.agent_id)
                    # rt_robot_edge.attrs['rt_translation'] = [opt_translation[0], opt_translation[1], .0]
                    # rt_robot_edge.attrs['rt_rotation_euler_xyz'] = [.0, .0, opt_orientation]
                    # # rt_robot_edge.attrs['rt_se2_covariance'] = cov_matrix
                    # self.g.insert_or_assign_edge(rt_robot_edge)

                self.rt_api.insert_or_assign_edge_RT(room_node, robot_node.id, [opt_translation[0], opt_translation[1], 0], [0, 0, opt_orientation])
                # self.g2o.optimizer.save("test_post.g2o")
                # self.iterations += 1
                # if self.iterations == 100:
                self.last_odometry = robot_odometry   # Save last odometry
                # print("Time elapsed compute:", time.time() - init_time)
        elif self.init_graph:
            self.room_initialized = True
            self.init_graph = False
            self.initialize_g2o_graph()



    def add_noise(self, value, std_dev):
        # print("Value", value, "Noise", np.random.normal(0, std_dev))
        """
        Adds a noise variable to an initial value.

        Args:
            value (`object`.): initial value of the output variable that will be
                affected by the added noise.
                
                		- `value`: This is the original value that will be noisy added
                with random noise.
                		- `std_dev`: The standard deviation of the noise distribution,
                which determines the magnitude of the added noise.
            std_dev (float): standard deviation of random noise to be added to the
                given value in the `add_noise` function.

        Returns:
            int: a modified version of the input value with noise added following
            a normal distribution.

        """
        return value + np.random.normal(0, std_dev)

    def initialize_g2o_graph(self):
        # get robot pose in room
        """
        Initializes a Graph2o graph for robot pose estimation in an unknown
        environment, by adding fixed and nominal corners to the graph and creating
        a landmark matrix for each corner. It also sets up an RT transform matrix
        for corner measurements.

        Returns:
            bool: a success flag indicating whether the graph was successfully
            initialized, as well as the pose of the robot and the corner values.

        """
        odom_node = self.g.get_node("Shadow")
        self.odometry_node_id = odom_node.id
        room_node = self.g.get_node("room")
        robot_node = self.g.get_node("Shadow")
        # Check if room and robot nodes exist
        if room_node is None or robot_node is None:
            print("Room or robot node does not exist. g2o graph cannot be initialized")
            return False
        robot_edge_rt = self.rt_api.get_edge_RT(room_node, robot_node.id)
        print("Robot edge RT", robot_edge_rt.attrs)
        robot_tx, robot_ty, _ = robot_edge_rt.attrs['rt_translation'].value
        _, _, robot_rz = robot_edge_rt.attrs['rt_rotation_euler_xyz'].value
        # Add fixed pose to g2o
        self.g2o.add_fixed_pose(g2o.SE2(self.add_noise(robot_tx, 0), self.add_noise(robot_ty, 0), self.add_noise(robot_rz, 0.0)))
        self.last_odometry = (.0, .0, .0, int(time.time()*1000))
        print("Fixed pose added to g2o graph", robot_tx, robot_ty, robot_rz)

        # Get corner values from room node
        # corner_nodes = self.g.get_nodes_by_type("corner")
        # Order corner nodes by id

        corner_list = []
        corner_list_measured = []
        for i in range(4):
            corner_list.append(self.g.get_node("corner_"+str(i)))
            corner_list_measured.append(self.g.get_node("corner_"+str(i)+"_measured"))

        for i in range(4):
            corner_edge_measured_rt = self.rt_api.get_edge_RT(odom_node, corner_list_measured[i].id)
            corner_measured_tx, corner_measured_ty, _ = corner_edge_measured_rt.attrs['rt_translation'].value
            corner_edge_rt = self.inner_api.transform(room_node.name, corner_list[i].name)
            corner_tx, corner_ty, _ = corner_edge_rt
            print("Nominal corners", corner_tx, corner_ty)
            print("Measured corners", corner_measured_tx, corner_measured_ty)
            landmark_information = np.array([[1 / self.measurement_noise_std_dev, 0.0],
                                             [0.0, 1 / self.measurement_noise_std_dev]])
            print("Eye matrix", 0.1 * np.eye(2))
            print("Landmark information:", landmark_information)
            if corner_tx != 0.0 or corner_ty != 0.0:
                # self.g2o.add_landmark(corner_tx, corner_ty, 0.1 * np.eye(2), pose_id=0)
                self.g2o.add_nominal_corner(corner_edge_rt,
                                          corner_edge_measured_rt.attrs['rt_translation'].value,
                                          landmark_information, pose_id=0)

        self.visualizer.update_graph(self.g2o)
        # self.g2o.optimizer.save("init_graph.g2o")

        return True

    def get_displacement(self, odometry):
        """
        Processes an odometry sequence and calculates the total displacement along
        the x, y, and z axes, as well as the angular displacement, based on the
        lineal and angular velocities extracted from the sequence.

        Args:
            odometry (3D vector containing position (x, y, z), velocity (x, y, z),
                and time (s) data.): 3D position and orientation of a robot over
                time, which is used to calculate the displacement, velocity, and
                angular velocity of the robot through successive time steps.
                
                		- `odometry`: A tuple containing 4 elements (x, y, z, timestamp)
                representing the robot's position and orientation at a given time
                stamp.
                		- `last_odometry`: A tuple storing the last known timestamp and
                position of the robot. It is used to compute the displacement of
                the robot over time.
                		- `odometry_queue`: A queue storing a sequence of `odometry`
                tuples, which are extracted from a file or sensor readings.
                		- `indice`: An integer index indicating the current position in
                the `odometry_queue`.
                
                	The function processes the `odometry` tuple by tuple and computes
                the robot's displacement in 3D space as well as its angular
                displacement, based on the difference in timestamps between adjacent
                tuples. The computed displacements are then printed to the console.

        Returns:
            tuple: the total displacement in the x, y, and z axes, calculated based
            on the odometry data provided.

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
        print("Desplazamiento total avance:", desplazamiento_avance)
        print("Desplazamiento total lateral:", desplazamiento_lateral)
        print("Desplazamiento total angular:", desplazamiento_angular)
        return desplazamiento_lateral, desplazamiento_avance, desplazamiento_angular

    def get_covariance_matrix(self, vertex):
        """
        Computes the covariance matrix for a given vertex in a graph using the
        Optimal Margin Estimator (G2O). It returns the computed covariance matrix
        or an indication that it was not computed.

        Args:
            vertex (instance/instance of `g2o.vertex.Vertex`.): 3D vertex being
                processed and used to compute its covariance matrix with other
                vertices in the scene.
                
                		- `hessian_index`: An integer value representing the index of
                the vertex in the Hessian matrix.
                
                	The function then proceeds to compute and return the covariance
                matrix for the given vertex using the `g2o.optimizer.compute_marginals`
                method, which is a built-in function in the `G2O` library.

        Returns:
            bool: a tuple containing the result of computing the covariance matrix
            and the resulting matrix.

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

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        # pass
        # check if room node is created
        """
        Updates attributes for a given node in a graph and appends to an odometry
        queue based on node ID.

        Args:
            id (int): id of the node for which the attribute names are being checked.
            attribute_names ([str]): list of node attributes that should be checked
                for changes, allowing the function to filter which nodes it updates
                based on their names.

        """
        room_node = self.g.get_node("room")
        if room_node is not None and not self.init_graph:
            if id == room_node.id and "valid" in attribute_names:
                self.init_graph = True
                print("INIT GRAPH")

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
        Updates a node's type and checks if it is a corner node, retrieves other
        nodes with the same type, and initializes the room graph if necessary.

        Args:
            id (int): unique identifier of the node to be updated.
            type (str): type of node to be updated, which determines the specific
                action taken by the function.

        """
        pass

    def delete_node(self, id: int):
        """
        Deletes a node from a graph based on its ID, waiting until a new room is
        visible before completing the operation.

        Args:
            id (int): integer identifier of the node to be deleted.

        """
        if type == "room":
        #         TODO: reset graph and wait until new room appears
            self.room_initialized = False


        # console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):
        """
        Updates an edge in a graph with given frame and type.

        Args:
            fr (int): 1st endpoint of an edge being updated.
            to (int): destination node or vertex that the edge is being updated for.
            type (str): edge's new type.

        """
        pass
        # console.print(f"UPDATE EDGE: {fr} to {type}", type, style='green')

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        """
        Updates an edge's attributes based on its frame (fr) and type (type), while
        providing a brief notification in green.

        Args:
            fr (int): first vertex ID of an edge to be updated in the graph.
            to (int): 2-element list of target attribute names that should be
                updated when the edge is updated, in accordance with the provided
                type.
            type (str): type of attribute that needs to be updated.
            attribute_names ([str]): 0 or more attribute names that will be updated
                in the database when the `fr` frame number and `to` type are used
                to update an edge attribute.

        """
        pass
        # console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')

    def delete_edge(self, fr: int, to: int, type: str):
        """
        Deletes an edge from a graph, identified by its from and type labels,
        respectively stored in the `fr` and `type` arguments.

        Args:
            fr (int): ID of the edge to be deleted.
            to (int): 2nd node ID in the edge being deleted.
            type (str): type of edge being deleted, which is displayed in the
                console using the `console.print()` method in a designated color
                based on its value.

        """
        pass
        # console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
