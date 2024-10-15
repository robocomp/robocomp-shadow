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

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 500

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
        planeId = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)

        self.pybullet_robot_id = -1
        self.run = False
        self.room_created = False
        self.created_door = False
        self.py_walls_ids = []
        self.current_room_id = -1
        self.created_doors = []
        self.walls_height = 2.
        self.doors_height = 2.
        
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

        self.update_robot_pose()
        print("ROOM_CREATED",self.room_created)
        if not self.room_created:
            nominal_corners = self.get_room_nominal_corners()
            print("NOMINAL CORNERS LEN",len(nominal_corners))
            if len(nominal_corners) == 4:
                if self.create_room(nominal_corners):
                    self.room_created = True
            else:
                print("Nominal corners no initialized yet")
        else:
            for door in self.g.get_nodes_by_type("door"):
                print("DOOR", door.name)
                print(door.attrs["room_id"].value)
                print(self.current_room_id)
                if "pre" not in door.name and door.attrs["room_id"].value == self.current_room_id and door.name not in self.created_doors:
                    print("DOOR", door.name)
                    self.create_door(door)
                    self.created_doors.append(door.name)

        p.stepSimulation()
        time.sleep(1./240.)  # 240Hz de simulación

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
                robot_pose_x, robot_pose_y, _ = self.inner_api.transform(room_node.name, robot_node.name)
                robot_pose_x /= 1000
                robot_pose_y /= 1000

                if not self.run:
                    self.run = True

                    # Code that should execute only once
                    print("ROBOT POSE", robot_pose_x, robot_pose_y)
                    robot = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=1.4)
                    self.pybullet_robot_id = p.createMultiBody(0, robot,
                                                         basePosition=[robot_pose_x, robot_pose_y, 0.])
                    p.changeVisualShape(self.pybullet_robot_id, -1, rgbaColor=[0, 0, 0, 1])
                else:
                    p.resetBasePositionAndOrientation(self.pybullet_robot_id, [robot_pose_x, robot_pose_y, 0.], [0., 0., 0., 1])  # No rotation, so orientation is [0,0,0,1]

    # Crear las paredes usando cajas
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

                print("WORLD DOOR X,Y LEFT: ", world_door_x_left, world_door_y_left)
                print("WORLD DOOR X,Y RIGHT: ", world_door_x_right, world_door_y_right)
                # p.removeBody(2)  # Asegúrate de que el ID sea correcto

                wall_index = -1
                for index, wall in enumerate(self.py_walls_ids):
                    print("WALL", wall)
                    if self.is_door_in_wall_area(room_door_x, room_door_y, wall, 0.5):
                        print("WALL THAT CONTAINS DOOR", wall)
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
        print('create_room')
        corners_arr = {}

        current_edge = self.g.get_edges_by_type("current")
        print(len(current_edge))
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

            print(corners_arr)

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
        # console.print(f"UPDATE NODE: {id} {type}", style='green')
        # if type == "door" and self.room_created is True:
        #     door_node = self.g.get_node(id)
        #     if door_node is not None:
        #         if "pre" not in door_node.name:
        #             self.create_door(door_node)
        pass

    def delete_node(self, id: int):
        # console.print(f"DELETE NODE:: {id} ", style='green')
        pass

    def update_edge(self, fr: int, to: int, type: str):
        console.print(f"UPDATE EDGE: {fr} to {type}", type, style='green')

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')

    def delete_edge(self, fr: int, to: int, type: str):
        console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
        if type == "current":
            self.clear_pybullet()
            self.room_created = False
            self.current_room_id = -1
            self.created_doors = []