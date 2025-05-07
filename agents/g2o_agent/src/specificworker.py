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

from PySide6.QtCore import QTimer, QPointF
from PySide6.QtGui import QPolygonF
from rich.console import Console
from genericworker import *
from g2o_graph import G2OGraph
import g2o  # DOES NOT WORK WITH NUMPY 2
import numpy as np
import interfaces as ifaces
from collections import deque
import matplotlib.pyplot as plt
import time
import setproctitle
from g2o_visualizer import G2OVisualizer
from pydsr import *


sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

try:
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
    print("Process title set to", os.path.basename(os.getcwd()))
except:
    pass

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 100

        # YOU MUST SET AN UNIQUE ID FOR THIS AGENT IN YOUR DEPLOYMENT. "_CHANGE_THIS_ID_" for a valid unique integer
        self.agent_id = 20
        self.g = DSRGraph(0, "G2O_agent", self.agent_id)

        if startup_check:
            self.startup_check()
        else:
            self.hide()

            # DSR APIs
            self.rt_api = rt_api(self.g)
            self.inner_api = inner_api(self.g)

            # Odometry reading variables
            robot_node = self.g.get_node("Shadow")
            self.odometry_node_id = robot_node.id
            self.odometry_queue = deque(maxlen=50)
            self.last_odometry = None

            # G2O variables
            self.g2o = G2OGraph(verbose=False)
            self.visualizer = G2OVisualizer("G2O Graph")

            self.odometry_noise_std_dev = 1  # Standard deviation for odometry noise
            self.odometry_noise_angle_std_dev = 1  # Standard deviation for odometry noise
            self.measurement_noise_std_dev = 1  # Standard deviation for measurement noise

            self.last_room_id = None
            self.actual_room_id = None

            # Variables for counting agent FPS
            self.elapsed = time.time()
            self.iterations = 0

            self.init_graph = False

            self.current_edge_set = False
            self.first_rt_set = False

            # Variables when robot pose is set from a external agent (e.g. room_detector)
            self.translation_to_set = None
            self.rotation_to_set = None
            self.timestamp_to_set = None

            self.room_polygon = None
            self.security_polygon = None

            self.room_initialized = True if self.initialize_g2o_graph() else False

            self.rt_set_last_time = time.time()
            self.rt_time_min = 1

            # self.last_update_with_corners = time.time()

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
        if time.time() - self.elapsed > 1:
            print("Frame rate: ", self.iterations, " fps")
            self.elapsed = time.time()
            self.iterations = 0
        self.iterations += 1

        if self.room_initialized:
            self.first_rt_set = False
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
                robot_displacement = self.get_robot_displacement()

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
                self.g2o.add_odometry(robot_displacement[0], robot_displacement[1], robot_displacement[2], odom_information)

                # Check if robot pose is inside room polygon
                corner_nodes = [corner for corner in self.g.get_nodes_by_type("corner") if "measured" in corner.name]
                no_valid_corners_counter = 0
                if self.room_polygon is not None:
                    if self.room_polygon.containsPoint(robot_point, Qt.OddEvenFill):
                        for i, corner_node_ in enumerate(corner_nodes):
                            corner_node = self.g.get_node("corner_measured_" + str(i))
                            if corner_node is not None:
                                is_corner_valid = corner_node.attrs["valid"].value
                                if is_corner_valid:
                                    corner_edge = self.rt_api.get_edge_RT(robot_node, corner_node.id)
                                    corner_edge_mat = self.rt_api.get_edge_RT_as_rtmat(corner_edge, robot_displacement[3])[0:3, 3]
                                    corner_id = corner_node.attrs["corner_id"].value
                                    self.g2o.add_landmark(corner_edge_mat[0], corner_edge_mat[1], 0.05 * np.eye(2), pose_id=self.g2o.vertex_count-1, landmark_id=corner_id+1)
                                    print("Landmark added:", corner_edge_mat[0], corner_edge_mat[1], "Landmark id:", corner_id+1, "Pose id:", self.g2o.vertex_count-1)
                                else:
                                    no_valid_corners_counter += 1

                        # door_nodes = [node for node in self.g.get_nodes_by_type("door") if not "pre" in node.name and
                        #               node.name in self.g2o.objects]
                        # # Iterate over door nodes
                        # if self.security_polygon.containsPoint(robot_point, Qt.OddEvenFill):
                        #     for door_node in door_nodes:
                        #         try:
                        #             is_door_valid = door_node.attrs["valid"].value
                        #             if is_door_valid:
                        #                 door_measured_rt = door_node.attrs["rt_translation"].value
                        #                 if door_measured_rt[0] != 0.0 or door_measured_rt[1] != 0.0:
                        #                     self.g2o.add_landmark(door_measured_rt[0], door_measured_rt[1], 0.05 * np.eye(2),
                        #                                           pose_id=self.g2o.vertex_count - 1,
                        #                                           landmark_id=self.g2o.objects[door_node.name])
                        #             else:
                        #                 print("Door is not valid")
                        #         except KeyError:
                        #             print("Door node does not have valid attribute")

                chi_value = self.g2o.optimize(iterations=50, verbose=False)

                last_vertex = self.g2o.optimizer.vertices()[self.g2o.vertex_count - 1]
                opt_translation = last_vertex.estimate().translation()
                opt_orientation = last_vertex.estimate().rotation().angle()

                print("Optimized translation:", opt_translation, "Optimized orientation:", opt_orientation)
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
                #     self.last_update_with_corners = time.time()f
                #
                # Substract pi/2 to opt_orientation and keep the number between -pi and pi
                if opt_orientation > np.pi:
                    opt_orientation -= np.pi
                elif opt_orientation < -np.pi:
                    opt_orientation += np.pi

                rt_robot_edge = Edge(robot_node.id, room_node.id, "RT", self.agent_id)
                rt_robot_edge.attrs['rt_translation'] = Attribute(np.array([opt_translation[0], opt_translation[1], .0],dtype=np.float32), self.agent_id)
                rt_robot_edge.attrs['rt_rotation_euler_xyz'] = Attribute(np.array([.0, .0, opt_orientation],dtype=np.float32), self.agent_id)
                rt_robot_edge.attrs['timestamp_alivetime'] = Attribute(int(self.last_odometry[3]), self.agent_id)
                self.g.insert_or_assign_edge(rt_robot_edge)

        elif (self.first_rt_set and self.current_edge_set and self.translation_to_set is not None and self.rotation_to_set is not None) or time.time() - self.rt_set_last_time > 3:
            print("Initializing g2o graph")
            # if self.last_room_id is not None:
            #     self.g.delete_edge(self.g.get_node("room_"+str(self.last_room_id)).id, self.g.get_node("Shadow").id, "RT")
            if(self.initialize_g2o_graph()):
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
            # print("INITIALIZING G2O GRAPH")
            # print("Room changed to", self.actual_room_id)
            # print("###########################################################")
            # self.odometry_node_id = robot_node.id
            # Check if room and robot nodes exist
            if room_node is None or robot_node is None:
                print("Room or robot node does not exist. g2o graph cannot be initialized")
                return False

            if self.translation_to_set is None and self.rotation_to_set is None:
                robot_edge_rt = self.rt_api.get_edge_RT(room_node, robot_node.id)
                robot_tx, robot_ty, _ = robot_edge_rt.attrs['rt_translation'].value
                _, _, robot_rz = robot_edge_rt.attrs['rt_rotation_euler_xyz'].value
                robot_pose_timestamp = robot_edge_rt.attrs['timestamp_alivetime'].value
                print("Robot pose timestamp:", robot_pose_timestamp)
            else:
                robot_tx, robot_ty, _ = self.translation_to_set
                _, _, robot_rz = self.rotation_to_set
                robot_pose_timestamp = self.timestamp_to_set

            # Add fixed pose to g2o
            self.g2o.add_fixed_pose(g2o.SE2(robot_tx, robot_ty, robot_rz))
            self.last_odometry = (.0, .0, .0, robot_pose_timestamp)
            print("Fixed pose added to g2o graph", robot_tx, robot_ty, robot_rz)

            corner_list = []

            # Generate QPolygonF with corner values
            self.room_polygon = QPolygonF()
            room_center = QPointF(0, 0)
            corner_nodes = [corner for corner in self.g.get_nodes_by_type("corner") if not "measured" in corner.name]
            corner_measured_nodes = [corner for corner in self.g.get_nodes_by_type("corner") if "measured" in corner.name]
            for i, corner_node_ in enumerate(corner_nodes):
                corner_node = self.g.get_node("corner_"+str(self.actual_room_id)+"_"+str(i))
                corner_edge_rt = self.inner_api.transform(room_node.name, corner_node.name)
                print("Corner", i, corner_edge_rt)
                corner_tx, corner_ty, _ = corner_edge_rt
                corner_list.append(corner_edge_rt)
                self.room_polygon.append(QPointF(corner_tx, corner_ty))
                room_center += self.room_polygon.at(i)
                # Insert in security polygon the same point but with and offset towards the room center (0, 0)

            # Calculate room center
            room_center /= len(corner_nodes)
            # Get room_polygon shortest side # TODO: set security polygon as a parameter that depends on room dimensions
            room_poly_bounding = self.room_polygon.boundingRect()
            d = 350
            self.security_polygon = QPolygonF()
            if self.room_polygon is not None:
                landmark_information = np.array([[0.05, 0.0],
                                                 [0.0, 0.05]])
                robot_point = QPointF(robot_tx, robot_ty)
                if self.room_polygon.containsPoint(robot_point, Qt.OddEvenFill):
                    
                    for i, corner_measured_node_ in enumerate(corner_measured_nodes):
                        corner_measured_node = self.g.get_node("corner_measured_"+str(i))
                        # Variables for security polygon
                        dir_vector = self.room_polygon.at(i) - room_center
                        dir_vector /= np.linalg.norm(np.array([dir_vector.x(), dir_vector.y()]))
                        corner_in = self.room_polygon.at(i) - d * dir_vector
                        self.security_polygon.append(corner_in)
                        # print("Corner in:", corner_in, "corresponding to corner", corner_list[i])
                        if corner_measured_node is not None:
                            is_corner_valid = corner_measured_node.attrs["valid"].value
                            if is_corner_valid:
                                corner_edge_measured_rt = self.rt_api.get_edge_RT(robot_node, corner_measured_node.id)
                                # print("Eye matrix", 0.1 * np.eye(2))
                                # print("Landmark information:", landmark_information)
                                if corner_tx != 0.0 or corner_ty != 0.0:
                                    # self.g2o.add_landmark(corner_tx, corner_ty, 0.1 * np.eye(2), pose_id=0)
                                    print("corner",i, corner_list[i], "corner edge measured", corner_edge_measured_rt.attrs['rt_translation'].value)
                                    self.g2o.add_nominal_corner(corner_list[i],
                                                              corner_edge_measured_rt.attrs['rt_translation'].value,
                                                              landmark_information, pose_id=0)
                            else:
                                print("Corner is not valid")
                                self.g2o.add_nominal_corner(corner_list[i],
                                                            None,
                                                            landmark_information, pose_id=0)
                    # door_nodes = [node for node in self.g.get_nodes_by_type("door") if not "pre" in node.name and
                    #               node.attrs["room_id"].value == self.actual_room_id]
                    # # Iterate over door nodes
                    # for door_node in door_nodes:
                    #     door_room_rt = self.inner_api.transform(room_node.name, door_node.name)
                    #     door_tx, door_ty, _ = door_room_rt
                    #     # Check if door is valid
                    #     try:
                    #         is_door_valid = door_node.attrs["valid"].value
                    #         if is_door_valid:
                    #             if door_tx != 0.0 or door_ty != 0.0:
                    #                 door_measured_rt = door_node.attrs["rt_translation"].value
                    #                 self.g2o.add_nominal_corner(door_room_rt,
                    #                                             door_measured_rt,
                    #                                             landmark_information, pose_id=0)
                    #         else:
                    #             print("Door is not valid")
                    #             self.g2o.add_nominal_corner(door_room_rt,
                    #                                         None,
                    #                                         landmark_information, pose_id=0)
                    #
                    #     except KeyError:
                    #         print("Door node does not have valid attribute")
                    #         self.g2o.add_nominal_corner(door_room_rt,
                    #                                     None,
                    #                                     landmark_information, pose_id=0)
                    #     self.g2o.objects[door_node.name] = self.g2o.vertex_count - 1
                    return True
        elif robot_node.attrs["parent"].value != 100:
            robot_parent = robot_node.attrs["parent"].value
            room_node = self.g.get_node(robot_parent)
            if self.translation_to_set is None and self.rotation_to_set is None:
                robot_edge_rt = self.rt_api.get_edge_RT(room_node, robot_node.id)
                robot_tx, robot_ty, _ = robot_edge_rt.attrs['rt_translation'].value
                _, _, robot_rz = robot_edge_rt.attrs['rt_rotation_euler_xyz'].value
                robot_pose_timestamp = robot_edge_rt.attrs['timestamp_alivetime'].value
            else:
                robot_tx, robot_ty, _ = self.translation_to_set
                _, _, robot_rz = self.rotation_to_set
                robot_pose_timestamp = self.timestamp_to_set

            # Add fixed pose to g2o
            self.g2o.add_fixed_pose(g2o.SE2(robot_tx, robot_ty, robot_rz))
            self.last_odometry = (.0, .0, .0, robot_pose_timestamp)
            print("Fixed pose added to g2o graph", robot_tx, robot_ty, robot_rz)
            return False
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

    def get_robot_displacement(self):
        queue_copy = self.get_odometry_simple(np.array(self.odometry_queue.copy()),
                                              self.last_odometry)

        if not queue_copy:
            print("No odometry values or invalid format")
            return np.array([0, 0, 0], dtype=np.float32)

        self.last_odometry = queue_copy[-1]

        accum = np.eye(4)

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
        robot_displacement = np.array([accum[0, 3] * 1000, accum[1, 3] * 1000, np.arctan2(accum[1, 0], accum[0, 0]), self.last_odometry[3]], dtype=np.float32)

        return robot_displacement

    def get_odometry_simple(self, queue, last_odometry):
        return [odom for odom in queue if last_odometry[3] <= float(odom[3])]

    # def get_robot_displacement(self, odometry_queue):
    #     desplazamiento_avance = 0
    #     desplazamiento_lateral = 0
    #     desplazamiento_angular = 0
    #     try:
    #         indice = next(index for index, (_, _, _,timestamp) in enumerate(odometry_queue) if timestamp == self.last_odometry[3])
    #     except StopIteration:
    #         self.last_odometry = odometry
    #         return desplazamiento_lateral, desplazamiento_avance, desplazamiento_angular
    #     # print("Index range", indice, len(self.odometry_queue) - 1)
    #     # Sumar las velocidades lineales entre el timestamp pasado y el actual
    #     for i in range(indice, len(odometry_queue)-1):
    #         # print("Diferencia tiempo actual y pasado:", self.odometry_queue[i + 1][3] - self.odometry_queue[i][3])
    #
    #         desplazamiento_avance += odometry_queue[i + 1][0] * (self.odometry_queue[i + 1][3] - self.odometry_queue[i][3]) * 0.8 / 1000
    #         desplazamiento_lateral += odometry_queue[i + 1][1] * (self.odometry_queue[i + 1][3] - self.odometry_queue[i][3]) *0.8 / 1000
    #         desplazamiento_angular -= self.odometry_queue[i + 1][2] * (self.odometry_queue[i + 1][3] - self.odometry_queue[i][3]) *0.8 / 1000
    #     return desplazamiento_lateral, desplazamiento_avance, desplazamiento_angular

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
        # room_node = self.g.get_node("room")
        # if room_node is not None and not self.init_graph:
        #     if id == room_node.id and "valid" in attribute_names:
        #         # self.init_graph = True
        #         print("INIT GRAPH")

        if id == self.odometry_node_id:
            odom_node = self.g.get_node("Shadow")
            odom_attrs = odom_node.attrs
            self.odometry_queue.append(np.array([odom_attrs["robot_current_advance_speed"].value, odom_attrs["robot_current_side_speed"].value, odom_attrs["robot_current_angular_speed"].value, odom_attrs["timestamp_alivetime"].value]))
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
        pass
        # if type == "room":
        # #         TODO: reset graph and wait until new room appears
        #     self.room_initialized = False


        # console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):
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
                self.timestamp_to_set = rt_edge.attrs['timestamp_alivetime'].value
                self.first_rt_set = True
                print("Translation to set", self.translation_to_set)
                print("Rotation to set", self.rotation_to_set)
                self.rt_set_last_time = time.time()

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        pass


    def delete_edge(self, fr: int, to: int, type: str):
        pass
        # console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
