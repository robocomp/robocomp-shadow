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

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from rich.console import Console
from sympy.physics.units import current
from torch.futures import wait_all
from transformers.models.flava.modeling_flava import FlavaSelfAttention

from genericworker import *
import interfaces as ifaces

console = Console(highlight=False)

from pydsr import *
import pybullet as p
import pybullet_data
import time
import numpy as np
import math
import os

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 1

        # YOU MUST SET AN UNIQUE ID FOR THIS AGENT IN YOUR DEPLOYMENT. "_CHANGE_THIS_ID_" for a valid unique integer
        self.agent_id = 15
        self.g = DSRGraph(0, "internalrepresentation", self.agent_id)
        self.inner_api = inner_api(self.g)

        # Pybullet
        self.physicsClient = p.connect(p.GUI)
        # Hide some GUI debug elements
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setTimeStep(1/50,0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)

        # Boolean flags for room and door creation
        self.room_created = False
        self.created_door = False
        self.current_room_id = -1

        # Pybullet variables
        self.pybullet_robot_id = -1
        self.py_walls_ids = []
        self.created_doors = []
        self.walls_height = 2.
        self.doors_height = 2.
        self.robot_height = 1.26443

        # Sintetic Lidar
        self.lidar_height = 1.1169  # lidar height from the ground in meters
        self.lidar_range = 5.  # Lidar max range in meters
        # self.angles_polares = [(np.radians(az), np.radians(el)) for el in range(0, 60, 10) for az in range(-180, 180, 10)]
        # One lidar ray in robot middle
        self.angles_polares = [(np.radians(az), np.radians(el)) for el in range(-1, 61, 5) for az in range(-181, 181, 5)]

        # Helios lidar angles
        self.helios_angles_polares = [(np.radians(az), np.radians(el)) for el in range(-1, 41, 5) for az in range(-181, 181, 5)]
        self.helios_color = [0, 1, 0]
        self.robot_helios_pose = [0, -0.16, (self.lidar_height / 2) - 0.08 ]
        self.helios_debug_lines_ids = []
        self.helios_interface_lidar_data = []
        self.helios_create_lines = False

        # Bpearl lidar angles
        self.bpearl_angles_polares = [(np.radians(az), np.radians(el)) for el in range(-41, -1, 5) for az in range(-181, 181, 5)]
        self.bpearl_color = [0, 0, 1]
        self.robot_bpearl_pose = [0, 0.14, (self.lidar_height / 2) - 0.48]
        self.bpearl_debug_lines_ids = []
        self.bpearl_interface_lidar_data = []
        self.bpearl_create_lines = False
        # Debug lines array to store the ids of the lines


        #Robot integration in PyBullet
        user = os.getlogin()
        stl_path = f"/home/{user}/robocomp/components/robocomp-shadow/agents/internal_representation/Shadow_Assembly_res.STL"
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=stl_path)
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=stl_path)

        # Create the robot in the simulation
        self.pybullet_robot_id = p.createMultiBody(baseCollisionShapeIndex=collision_shape_id,
                                    baseVisualShapeIndex=visual_shape_id,
                                    basePosition=[0., 0., self.robot_height / 2],  # Initial robot pose
                                    baseOrientation=[0, 0 , 0])

        # Change the color of the robot
        p.changeVisualShape(self.pybullet_robot_id, -1, rgbaColor=[0.827, 0.827, 0.827, 1])

        # Disable collisions of the robot with the sintetic lidar
        p.setCollisionFilterGroupMask(self.pybullet_robot_id, -1, collisionFilterGroup=0, collisionFilterMask=0)

        # # Create a wall close to the robot FOR DEBUGGING
        # wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 0.1, 1])
        # wall_id = p.createMultiBody(0, wall, basePosition=[1, 1, 0.5], baseOrientation=[0, 0, 0, 1])
        # p.changeVisualShape(wall_id, -1, rgbaColor=[0, 0, 1, 1])

        # Initialize the lidar for the first time and draw the debug lines
        distances, ray_from, ray_to, hit_positions = self.robot_sintetic_lidar_distances(self.robot_bpearl_pose, self.bpearl_angles_polares, "bpearl")
        self.bpearl_create_lines = self.draw_debug_lines(distances, ray_from, ray_to, hit_positions, self.bpearl_color, self.bpearl_debug_lines_ids, self.bpearl_create_lines)

        distances, ray_from, ray_to, hit_positions = self.robot_sintetic_lidar_distances(self.robot_helios_pose, self.helios_angles_polares, "helios")
        self.helios_create_lines = self.draw_debug_lines(distances, ray_from, ray_to, hit_positions, self.helios_color, self.helios_debug_lines_ids, self.helios_create_lines)

        try:
            # signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.update_node_att)
            # signals.connect(self.g, signals.UPDATE_NODE, self.update_node)
            # signals.connect(self.g, signals.DELETE_NODE, self.delete_node)
            # signals.connect(self.g, signals.UPDATE_EDGE, self.update_edge)
            # signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.update_edge_att)
            signals.connect(self.g, signals.DELETE_EDGE, self.delete_edge)
            console.print("signals connected")
        except RuntimeError as e:
            print(e)

        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        # try:
        #	self.innermodel = InnerModel(params["InnerModelPath"])
        # except:
        #	traceback.print_exc()
        #	print("Error reading config params")
        return True


    @QtCore.Slot()
    def compute(self):

        # Get the robot pose in the world from the working memory graph if the robot is in a room
        robot_pose_x, robot_pose_y = self.update_robot_pose()
        if robot_pose_x is None or robot_pose_y is None:
            return
        else:
            print("Beparl debug lines len:", len(self.bpearl_debug_lines_ids), "BOLEAN BPEARL", self.bpearl_create_lines)
            print("Helios debug lines len:", len(self.helios_debug_lines_ids), "BOLEAN HELIOS", self.helios_create_lines)
            # Compute the sintetic lidar distances and draw the debug lines
            distances, ray_from, ray_to, hit_positions = self.robot_sintetic_lidar_distances(self.robot_bpearl_pose,
                                                                                             self.bpearl_angles_polares,
                                                                                             "bpearl")
            self.bpearl_create_lines = self.draw_debug_lines(distances, ray_from, ray_to, hit_positions, self.bpearl_color,
                                  self.bpearl_debug_lines_ids, self.bpearl_create_lines)

            distances, ray_from, ray_to, hit_positions = self.robot_sintetic_lidar_distances(self.robot_helios_pose,
                                                                                             self.helios_angles_polares,
                                                                                             "helios")
            self.helios_create_lines = self.draw_debug_lines(distances, ray_from, ray_to, hit_positions, self.helios_color,
                                  self.helios_debug_lines_ids, self.helios_create_lines)

        # Check if the room has been created
        if not self.room_created:
            # Get the nominal corners of the current room
            nominal_corners = self.get_room_nominal_corners()
            if len(nominal_corners) == 4:
                # Create the room in PyBullet using the nominal corners
                if self.create_room(nominal_corners):
                    self.room_created = True
            else:
                print("Nominal corners no initialized yet")
        else:
            # Iterate over the nominal doors (not "pre_door") in the graph and create them in PyBullet
            for door in self.g.get_nodes_by_type("door"):
                if "pre" not in door.name and door.attrs["room_id"].value == self.current_room_id and door.name not in self.created_doors:
                    # Create door
                    self.create_door(door)
                    # Add the door to the list of created doors
                    self.created_doors.append(door.name)

        p.stepSimulation()
        time.sleep(1./240.)  # 240Hz de simulación

    def robot_sintetic_lidar_distances(self, lidar_from_local, lidar_angles, name):
        # Get robot position and orientation from PyBullet
        robot_pos, robot_orient = p.getBasePositionAndOrientation(self.pybullet_robot_id)

        # Crete the LIDAR in the local position (0, 0, height)
        # lidar_from_local = [0, 0, self.lidar_height / 2]  # El LIDAR en la posición local (0, 0, altura)

        # Create the ray initial and final points in global coordinates for each angle
        ray_from = []
        ray_to = []
        ray_from_global = []

        for azimuth, elevation in lidar_angles:
            # Calcular la posición del rayo en coordenadas polares
            y_local = self.lidar_range * np.cos(azimuth) * np.cos(elevation)
            x_local = self.lidar_range * np.cos(elevation) * np.sin(azimuth)
            z_local = self.lidar_range * np.sin(elevation) # Maybe lidar_height should be added here

            # Transform the beam from local to global using robot position and orientation
            ray_from_global, _ = p.multiplyTransforms(robot_pos, robot_orient, lidar_from_local, [0, 0, 0, 1])
            ray_to_global, _ = p.multiplyTransforms(robot_pos, robot_orient, [x_local, y_local, z_local], [0, 0, 0, 1])

            ray_from.append(ray_from_global)
            ray_to.append(ray_to_global)

        # Perform the ray test in PyBullet to get the distances and hit positions of the rays
        ray_results = p.rayTestBatch(ray_from, ray_to)

        distances = []
        hit_positions = []
        lidar_points = []
        hit_distance = -1

        for i, result in enumerate(ray_results):
            hit_object_id = result[0]  # ID of the object hit by the beam (-1 if nothing or the robot is hit)
            hit_position = result[3]  # Hit position in global coordinates
            azimuth, elevation = lidar_angles[i]

            # Check if the hit object is the robot itself or nothing TODO: Check if the hit object is the robot it is necessary
            if hit_object_id == -1 or hit_object_id == self.pybullet_robot_id:
                # Set the hit distance to the maximum range of the LIDAR and the hit position to the final point of the corresponding ray
                hit_distance = self.lidar_range
                hit_position = ray_to[i]
            else:
                # Calculate the distance from the hit position to the LIDAR origin in global coordinates
                hit_distance = np.linalg.norm(np.array(hit_position) - np.array([ray_from_global[0], ray_from_global[1], self.lidar_height / 2]))

            # Calculate the 2D distance from the hit position to the LIDAR origin
            distance2d = np.linalg.norm(np.array(hit_position[:2]) - np.array(ray_from[i][:2]))
            # Store the hit position and distances in the corresponding lists for further processing and visualization
            hit_positions.append(hit_position)
            distances.append(hit_distance)

            # Transform the impact position to the robot's reference system for visualization in Lidar3D viewer component
            ray_to_robot_ref, _ = p.multiplyTransforms([0, 0, 0], p.invertTransform(robot_pos, robot_orient)[1],
                                                             hit_position, [0, 0, 0, 1])

            # Transform to RoboCompLidar3D.TPoint format and append to the list of lidar points for RoboCompLidar3D component
            point = ifaces.RoboCompLidar3D.TPoint(x = ray_to_robot_ref[0]*1000, y = ray_to_robot_ref[1]*1000, z = ray_to_robot_ref[2]*1000,
                                                  intensity = 1, phi = azimuth, theta = elevation, r = hit_distance*1000,
                                                  distance2d = distance2d*1000)
            lidar_points.append(point)

        # Create the RoboCompLidar3D.TData object
        if name == "helios":
            self.helios_interface_lidar_data = ifaces.RoboCompLidar3D.TData(points = lidar_points, period = 1./240. ,timestamp = int(time.time())*1000)
        elif name == "bpearl":
            self.bpearl_interface_lidar_data = ifaces.RoboCompLidar3D.TData(points = lidar_points, period = 1./240. ,timestamp = int(time.time())*1000)

        return distances, ray_from, ray_to, hit_positions

    # Draw the Lidar debug lines in PyBullet, the red lines are the ones that do not hit anything, the black ones are the ones that hit something
    def draw_debug_lines(self, distances, ray_from, ray_to, hit_positions, color, debug_lines_ids, create_lines):
        point = [0, 0, 0]
        for i, (from_pos, hit_position) in enumerate(zip(ray_from, hit_positions)):
            if distances[i] == self.lidar_range:
                # color = [1, 0, 0]
                point = ray_to[i]
            else:
                # color = [0, 0, 0]  # Rojo si no golpea nada, negro si golpea algo
                point = hit_position

            if not create_lines:
                line_id = p.addUserDebugLine(from_pos, point, color)
                time.sleep(0.01)
                debug_lines_ids.append(line_id)
            else:
                # Actualizar la línea existente
                if len(debug_lines_ids) > 0:
                    p.addUserDebugLine(from_pos, point, lineColorRGB=color, lineWidth=1,
                                       replaceItemUniqueId=debug_lines_ids[i])

        if not create_lines: create_lines = True
        return create_lines

    # TODO: UPDATE FUNCTION USING ROOBOT_SINTETIC_LIDAR FUNCTION
    def sintetic_lidar_items(self, robot_pose_x, robot_pose_y):
        self.ray_from = []
        self.ray_to = []
        angles_polares = [(np.radians(az), 0) for az in range(0, 360, 10)]

        for azimuth, elevation in angles_polares:
            # Calcular la posición del rayo en coordenadas polares
            x = robot_pose_x + (self.lidar_range * np.cos(azimuth) * np.cos(elevation))
            y = robot_pose_y + (self.lidar_range * np.cos(elevation) * np.sin(azimuth))
            z = self.lidar_height + (self.lidar_range * np.sin(elevation))

            # Añadir el rayo a la lista
            self.ray_from.append([robot_pose_x, robot_pose_y, self.lidar_height])
            self.ray_to.append([x, y, z])

        ray_results = p.rayTestBatch(self.ray_from, self.ray_to)

        # Inicializar un diccionario para almacenar resultados
        hit_info = {}
        for result in ray_results:
            hit_object_id = result[0]  # ID del objeto que el rayo ha golpeado (-1 si no golpea nada)
            hit_position = result[3]  # Posición de impacto

            if hit_object_id != -1:
                hit_distance = np.linalg.norm(np.array(hit_position) - np.array([robot_pose_x, robot_pose_y, self.lidar_height]))
                hit_info[hit_object_id] = {'hit_position': hit_position, 'hit_distance': hit_distance }

        return hit_info  # Devolver el diccionario con la información de impactos

    # Returns the nominal corners of the room as a list of nodes
    def get_room_nominal_corners(self):
        current_edge = self.g.get_edges_by_type("current")
        if len(current_edge) > 0:
            room_node = self.g.get_node(current_edge[0].origin)
            corners  = self.g.get_nodes_by_type("corner")

            if len(corners) > 0:
                nominal_corners = [node for node in corners if "measured" not in node.name and node.attrs["room_id"].value == room_node.attrs["room_id"].value]
                if len(nominal_corners) > 0:
                    return nominal_corners

        return []

    # Returns the distance between the door and the wall
    def is_door_in_wall_area(self, door_x, door_y, wall, width):
        # Función para calcular la distancia desde un punto a una línea
        def distance_point_to_line(px, py, ax, ay, bx, by):
            numerator = abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax)
            denominator = math.sqrt((by - ay) ** 2 + (bx - ax) ** 2)
            return numerator / denominator if denominator != 0 else float('inf')

        cornerA = wall["cornerA"]
        cornerB = wall["cornerB"]

        # Obtener coordenadas
        x1, y1 = cornerA[0], cornerA[1]
        x2, y2 = cornerB[0], cornerB[1]

        # Calcular la distancia desde la puerta a la línea
        distance = distance_point_to_line(door_x, door_y, x1, y1, x2, y2)

        # Comprobar si la distancia es menor o igual a la mitad del ancho del muro
        return distance <= width / 2

    # Get the current room node from the working memory graph
    def get_current_room_node(self):
        current_edge = self.g.get_edges_by_type("current")
        if len(current_edge) > 0:
            room_node_id = current_edge[0].origin
            if (room_node := self.g.get_node(room_node_id)) is not None:
                return room_node

        return

    def update_robot_pose(self):
        if (robot_node := self.g.get_node("Shadow")) is not None:
            if (room_node := self.get_current_room_node()) is not None:
                if (transform := self.inner_api.transform_axis(room_node.name, robot_node.name)) is not None:
                    print(transform)
                    robot_pose_x, robot_pose_y, _, angle_x, angle_y, angle_z = transform
                    robot_pose_x /= 1000
                    robot_pose_y /= 1000
                    orientation_quaternion = p.getQuaternionFromEuler([0, 0, angle_z])
                    p.resetBasePositionAndOrientation(self.pybullet_robot_id, [robot_pose_x, robot_pose_y, self.robot_height / 2], orientation_quaternion)  # No rotation, so orientation is [0,0,0,1]

                    return robot_pose_x, robot_pose_y

        return None, None

    # Create the walls using the nominal corners of the room as input
    def create_wall(self, corner1, corner2, color):
        x1, y1 = corner1[:2]
        x2, y2 = corner2[:2]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Calcular el ángulo para la rotación
        angle = math.atan2(y2 - y1, x2 - x1)
        # Crear la pared
        wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=[distance / 2, 0.1, self.walls_height])
        wall_id = p.createMultiBody(0, wall, basePosition=[(x1 + x2) / 2, (y1 + y2) / 2, self.walls_height],
                                    baseOrientation=p.getQuaternionFromEuler([0, 0, angle]))  # Rotar la pared
        p.changeVisualShape(wall_id, -1, rgbaColor=color)  # Cambiar color de la pared
        return wall_id

    def create_door(self,door):
        if (wall_id := door.attrs["parent"].value) is not None:
            if (wall_node := self.g.get_node(wall_id)) is not None:
                room_node = self.get_current_room_node()
                door_width = door.attrs["width"].value
                door_x, door_y, _ = self.inner_api.transform(wall_node.name, door.name)
                room_door_x, room_door_y, _ = self.inner_api.transform(room_node.name, door.name)

                room_door_x /= 1000.
                room_door_y /= 1000.

                door_x_left = door_x - door_width / 2
                door_x_right = door_x + door_width / 2

                world_door_x_left, world_door_y_left, _ = self.inner_api.transform( room_node.name, [door_x_left, 0., 0.], wall_node.name)
                world_door_x_right, world_door_y_right, _ = self.inner_api.transform( room_node.name, [door_x_right, 0., 0.], wall_node.name)

                world_door_x_left /= 1000.
                world_door_y_left /= 1000.
                world_door_x_right /= 1000.
                world_door_y_right /= 1000.

                wall_index = -1
                for index, wall in enumerate(self.py_walls_ids):
                    if self.is_door_in_wall_area(room_door_x, room_door_y, wall, 0.5):
                        wall_index = index
                        break

                x1, y1 = self.py_walls_ids[wall_index]["cornerB"][:2]
                distance = math.sqrt((world_door_x_left - x1) ** 2 + (world_door_y_left - y1) ** 2)

                # Get rotation angle from corresponding wall
                _ , orientation = p.getBasePositionAndOrientation(self.py_walls_ids[wall_index]["wall_id"][0])
                # Get color from corresponding wall
                color = p.getVisualShapeData(self.py_walls_ids[wall_index]["wall_id"][0])[0][7]
                # Remove wall that contains the door
                for wall in self.py_walls_ids[wall_index]["wall_id"]:
                    p.removeBody(wall)

                self.py_walls_ids[wall_index]["wall_id"] = []

                wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=[distance / 2, 0.1, self.walls_height])
                wall_id = p.createMultiBody(0, wall, basePosition=[(x1 + world_door_x_left) / 2, (y1 + world_door_y_left) / 2, self.walls_height],
                                            baseOrientation=orientation)  # Rotar la pared
                p.changeVisualShape(wall_id, -1, rgbaColor=color)  # Cambiar color de la pared
                self.py_walls_ids[wall_index]["wall_id"].append(wall_id)

                x1, y1 = self.py_walls_ids[wall_index]["cornerA"][:2]
                distance = math.sqrt((world_door_x_right - x1) ** 2 + (world_door_y_right - y1) ** 2)

                wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=[distance / 2, 0.1, self.walls_height])
                wall_id = p.createMultiBody(0, wall,
                                            basePosition=[(x1 + world_door_x_right) / 2, (y1 + world_door_y_right) / 2,
                                                          self.walls_height],
                                            baseOrientation=orientation)  # Rotar la pared
                p.changeVisualShape(wall_id, -1, rgbaColor=color)  # Cambiar color de la pared
                self.py_walls_ids[wall_index]["wall_id"].append(wall_id)

                wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=[door_width /2 / 1000., 0.1, self.walls_height / 2])
                wall_id = p.createMultiBody(0, wall,
                                            basePosition=[(world_door_x_right + world_door_x_left) / 2, (world_door_y_right + world_door_y_left) / 2, self.doors_height + self.walls_height / 2],
                                            baseOrientation=orientation)  # Rotar la pared
                p.changeVisualShape(wall_id, -1, rgbaColor=color)  # Cambiar color de la pared
                self.py_walls_ids[wall_index]["wall_id"].append(wall_id)

                print(self.py_walls_ids)

                # CREATES THE DOOR FRAME IN THE WALL
                # left_cilinder = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.15, height=self.walls_height)
                # left_cilinder_id = p.createMultiBody(0, left_cilinder,
                #                                     basePosition=[world_door_x_left, world_door_y_left, self.walls_height / 2])
                # p.changeVisualShape(left_cilinder_id, -1, rgbaColor=[1, 1, 1, 1])
                #
                # right_cilinder = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.15, height=self.walls_height)
                # right_cilinder_id = p.createMultiBody(0, right_cilinder,
                #                                     basePosition=[world_door_x_right, world_door_y_right, self.walls_height / 2])
                # p.changeVisualShape(right_cilinder_id, -1, rgbaColor=[1, 1, 1, 1])
                #
                # top_cilinder = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.15, height=door_width / 1000.)
                # top_cilinder_id = p.createMultiBody(0, top_cilinder, basePosition=[room_door_x, room_door_y, self.walls_height],
                #                                 baseOrientation=orientation)
                # p.changeVisualShape(top_cilinder_id, -1, rgbaColor=[1, 1, 1, 1])

    def clear_pybullet(self):
        for wall in self.py_walls_ids:
            for wall_id in wall["wall_id"]:
                p.removeBody(wall_id)
        self.py_walls_ids = []

    def create_room(self, nominal_corners):
        corners_arr = {}

        current_edge = self.g.get_edges_by_type("current")

        if len(current_edge) > 0 and len(nominal_corners) > 0:
            room_node = self.g.get_node(current_edge[0].origin)
            if room_node is None:
                return False
            self.current_room_id = room_node.attrs["room_id"].value
            for nominal_corner in nominal_corners:
                corner_edge_rt = self.inner_api.transform(room_node.name, nominal_corner.name)
                corner_edge_rt /= 1000.0
                if nominal_corner.attrs["corner_id"].value is not None and len(corner_edge_rt) > 0:
                    corners_arr["corner" + str(nominal_corner.attrs["corner_id"].value)] = [corner_edge_rt[0], corner_edge_rt[1], -1.]

        # Pared 1: entre corner0 y corner1 (horizontal inferior)
        p.addUserDebugLine(corners_arr["corner0"], corners_arr["corner1"], [1, 0, 0], 3)
        wall_id = self.create_wall(corners_arr["corner0"], corners_arr["corner1"], [1, 0, 0, 1])  # Rojo
        self.py_walls_ids.append({"wall_id":[wall_id], "cornerA":corners_arr["corner0"], "cornerB":corners_arr["corner1"]})

        # Pared 2: entre corner1 y corner2 (vertical derecha)
        p.addUserDebugLine(corners_arr["corner1"], corners_arr["corner2"], [0, 1, 0], 3)
        wall_id = self.create_wall(corners_arr["corner1"], corners_arr["corner2"], [0, 1, 0, 1])  # Verde
        self.py_walls_ids.append({"wall_id":[wall_id], "cornerA":corners_arr["corner1"], "cornerB":corners_arr["corner2"]})

        # Pared 3: entre corner2 y corner3 (horizontal superior)
        p.addUserDebugLine(corners_arr["corner2"], corners_arr["corner3"], [0, 0, 1], 3)
        wall_id = self.create_wall(corners_arr["corner2"], corners_arr["corner3"], [0, 0, 1, 1])  # Azul
        self.py_walls_ids.append({"wall_id":[wall_id], "cornerA":corners_arr["corner2"], "cornerB":corners_arr["corner3"]})

        # Pared 4: entre corner3 y corner0 (vertical izquierda)
        p.addUserDebugLine(corners_arr["corner3"], corners_arr["corner0"], [1, 1, 0], 3)
        wall_id = self.create_wall(corners_arr["corner3"], corners_arr["corner0"], [1, 1, 0, 1])  # Amarillo
        self.py_walls_ids.append({"wall_id":[wall_id], "cornerA":corners_arr["corner3"], "cornerB":corners_arr["corner0"]})

        return True

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)


    # =============== DSR SLOTS  ================
    # =============================================
    def update_node_att(self, id: int, attribute_names: [str]):
        console.print(f"UPDATE NODE ATT: {id} {attribute_names}", style='green')

    def update_node(self, id: int, type: str):
        pass

    def delete_node(self, id: int):
        # console.print(f"DELETE NODE:: {id} ", style='green')
        pass

    def update_edge(self, fr: int, to: int, type: str):
        console.print(f"UPDATE EDGE: {fr} to {type}", type, style='green')

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')

    def delete_edge(self, fr: int, to: int, type: str):
        if type == "current":
            console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
            self.clear_pybullet()
            self.room_created = False
            self.current_room_id = -1
            self.created_doors = []

# =============== Methods for Component Implements ==================
    # ===================================================================

    #
    # IMPLEMENTATION of getLidarData method from Lidar3D interface
    #
    def Lidar3D_getLidarData(self, name, start, len, decimationDegreeFactor):
        #print("Calling getLidarData")
        if name == "bpearl":
            return self.bpearl_interface_lidar_data
        elif name == "helios":
            return self.helios_interface_lidar_data

    #
    # IMPLEMENTATION of getLidarDataArrayProyectedInImage method from Lidar3D interface
    #
    def Lidar3D_getLidarDataArrayProyectedInImage(self, name):
        ret = ifaces.RoboCompLidar3D.TDataImage()
        #
        # write your CODE here
        #
        return ret
    #
    # IMPLEMENTATION of getLidarDataProyectedInImage method from Lidar3D interface
    #
    def Lidar3D_getLidarDataProyectedInImage(self, name):
        ret = ifaces.RoboCompLidar3D.TData()
        #
        # write your CODE here
        #
        return ret
    #
    # IMPLEMENTATION of getLidarDataWithThreshold2d method from Lidar3D interface
    #
    def Lidar3D_getLidarDataWithThreshold2d(self, name, distance, decimationDegreeFactor):
        ret = ifaces.RoboCompLidar3D.TData()
        #
        # write your CODE here
        #
        return ret
    # ===================================================================
    # ===================================================================


    ######################
    # From the RoboCompLidar3D you can use this types:
    # RoboCompLidar3D.TPoint
    # RoboCompLidar3D.TDataImage
    # RoboCompLidar3D.TData

