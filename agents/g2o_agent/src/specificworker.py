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
                odom_information = np.array([[1, 0.0, 0.0],
                                            [0.0, 1, 0.0],
                                            [0.0, 0.0, 0.1]])
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
                        self.g2o.add_landmark(corner_edge_mat[0], corner_edge_mat[1], 0.05 * np.eye(2), pose_id=self.g2o.vertex_count-1, landmark_id=int(corner_node.name[7])+1)
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
                # cov_matrix = self.get_covariance_matrix(last_vertex)
                # print("Covariance matrix:", cov_matrix)
                # self.visualizer.update_graph(self.g2o, None, cov_matrix)
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
        return value + np.random.normal(0, std_dev)

    def initialize_g2o_graph(self):
        # get robot pose in room
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
        self.g2o.add_fixed_pose(g2o.SE2(robot_tx, robot_ty, robot_rz))
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
            landmark_information = np.array([[0.05, 0.0],
                                             [0.0, 0.05]])
            print("Eye matrix", 0.1 * np.eye(2))
            print("Landmark information:", landmark_information)
            if corner_tx != 0.0 or corner_ty != 0.0:
                # self.g2o.add_landmark(corner_tx, corner_ty, 0.1 * np.eye(2), pose_id=0)
                self.g2o.add_nominal_corner(corner_edge_rt,
                                          corner_edge_measured_rt.attrs['rt_translation'].value,
                                          landmark_information, pose_id=0)

        # self.visualizer.update_graph(self.g2o)
        # self.g2o.optimizer.save("init_graph.g2o")

        return True

    def get_displacement(self, odometry):
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
        pass

    def delete_node(self, id: int):
        if type == "room":
        #         TODO: reset graph and wait until new room appears
            self.room_initialized = False


        # console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):
        pass
        # console.print(f"UPDATE EDGE: {fr} to {type}", type, style='green')

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        pass
        # console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')

    def delete_edge(self, fr: int, to: int, type: str):
        pass
        # console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
