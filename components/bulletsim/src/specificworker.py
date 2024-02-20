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

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import pybullet as p
import pybullet_data
import time
import numpy as np
import pkgutil
egl = pkgutil.get_loader('eglRenderer')
sys.path.append('/home/robocomp/robocomp/lib')
console = Console(highlight=False)
import cv2
import pyautogui

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 1000
        if startup_check:
            self.startup_check()
        else:

            # Start PyBullet in GUI mode
            self.physicsClient = p.connect(p.DIRECT) # p.GUI to see the graphio user interface, p.DIRECT to hide it
            # Set the path to PyBullet data
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf", [0, 0, -1])

            # Set gravity
            # p.setGravity(0, 0, -9.81)

            # Create the cylinder
            # Especificar las dimensiones del cilindro
            # cylinder_radius = 0.5
            # cylinder_height = 1.0
            #
            # # Especificar la posición en metros y orientación del cilindro
            # cylinder_position = [0, 0, 0]
            # cylinder_orientation = p.getQuaternionFromEuler([0, 0, 0])  # Sin rotación
            #
            # # Crear la forma del cilindro (collision shape)
            # cylinder_shape_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=cylinder_radius, height=cylinder_height)
            #
            # # Crear el cilindro como un cuerpo rígido
            # self.cylinder_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=cylinder_shape_id,
            #                                 basePosition=cylinder_position, baseOrientation=cylinder_orientation)

            # # Especificar la posición en metros y orientación de la caja
            # box_orientation = p.getQuaternionFromEuler([0, 0, 0])  # Sin rotación
            #
            # # Crear la forma de la caja (collision shape)
            # box_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_extents)
            #
            # # Crear la caja como un cuerpo rígido
            # self.box_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=box_shape_id,
            #                            basePosition=box_position, baseOrientation=box_orientation)
            self.obstacle_created = False
            self.singleshot_simple = False

            distance_between_points = 0.2

            self.data_dict = { "speed": [1, 0, 0],
                               "box_dimensions": [0.5, 0.5, 0.5],
                               "box_position": [-3.5, 0, 0],
                               "path": self.generate_equidistant_path(5, 0, 0, distance_between_points)
                               }

            #plugin = p.load    Plugin(egl.get_filename(), "_eglRendererPlugin")
            #print("plugin=", plugin)
            #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            #p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

            # self.personUid = p.loadURDF("girl.urdf", basePosition=[0, 0, 0])
                                   #baseOrientation=p.getQuaternionFromEuler([np.pi / 2, 0, 0]))
            #self.personUid = p.loadURDF("r2d2.urdf", baseOrientation=p.getQuaternionFromEuler([0, np.pi / 2, 0]), basePosition=[0, 0, 0])

            # Simulation parameters
            self.simulationTime = 30  # in seconds
            self.timeStep = 0.15 # time step for the simulation self.data_dict["speed"].norm

            # Define a steady forward speed
            self.forwardSpeed = 0.05  # Adjust this value as needed

            self.camDistance = 2
            self.yaw = 10

            # Parámetros de la elipse
            a = 5  # Semieje mayor
            b = 3  # Semieje menor
            d_points = 0.15  # Distancia entre puntos
            num_points = int((2 * np.pi * np.sqrt((a ** 2 + b ** 2) / 2)) / d_points)  # Estimación del número de puntos

            self.ellipse_path = self.generate_ellipse_path(a, b, d_points, num_points)
            #cv2.namedWindow('image')
            #cv2.setMouseCallback('image', self.mouse_click)
            p.setTimeStep(self.timeStep, 0)
            # Obtener el tiempo inicial de la simulación
            # self.start_time = p.getPhysicsEngineParameters()['fixedTimeStep']
            # get current time

            self.start = time.time()
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        return True

    @QtCore.Slot()
    def compute(self):
        # print simulation time
        pass
        # if not self.singleshot_simple:
        #     self.mission(self.data_dict)
        #     self.singleshot_simple = True

        # pos, ori = p.getBasePositionAndOrientation(self.personUid)
        #
        # # Calculate the forward direction based on the orientation
        # pos, ori = p.getBasePositionAndOrientation(self.personUid)
        # forwardDir = p.getMatrixFromQuaternion(ori)[0:3]  # Forward direction in world coordinates
        #
        # # Calculate the new position
        # newPos = [pos[0] + self.forwardSpeed * forwardDir[0] * self.timeStep,
        #           pos[1] + self.forwardSpeed * forwardDir[1] * self.timeStep,
        #           pos[2] + self.forwardSpeed * forwardDir[2] * self.timeStep]
        #
        # # Set the new position of the person
        # p.resetBasePositionAndOrientation(self.personUid, newPos, ori)
        # max_force = 2
        # joint_index = 0
        # current_position, _ = p.getBasePositionAndOrientation(self.cylinder_id)
        # force = max_force * (self.target_position[0] - current_position[0])
        #
        # # Aplicar fuerza a la articulación para mover el cilindro hacia la posición deseada
        # p.applyExternalForce(self.cylinder_id, -1, forceObj=[force, 0, 0], posObj=current_position, flags=p.LINK_FRAME)
        # linear_velocity = [1, 0, 0]  # Velocidad lineal en el eje x
        # p.resetBaseVelocity(self.cylinder_id, linearVelocity=linear_velocity)
        #
        # # Obtener información sobre el impacto entre el cilindro y la caja
        # contact_points = p.getContactPoints(self.cylinder_id) # , self.box_id
        #
        # # Verificar si hay puntos de contacto (impacto)
        # if contact_points:
        #     print("Timestamp del impacto time.time():", time.time() - self.start)
        #
        # # Step the simulation
        # p.stepSimulation()

        # pixelWidth = 640
        # pixelHeight = 480
        # camTargetPos = [0, 0, 0]
        # pitch = -10.0
        # roll = 0
        # upAxisIndex = 2
        # viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, self.camDistance, self.yaw, pitch, roll,
        #                                                  upAxisIndex)
        # projectionMatrix = [
        #     1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0, 0.0, 0.0, 0.0,
        #     -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0
        # ]
        #
        # _, _, rgba, _, _ = p.getCameraImage(pixelWidth,
        #                            pixelHeight,
        #                            viewMatrix=viewMatrix,
        #                            projectionMatrix=projectionMatrix,
        #                            shadow=1,
        #                            lightDirection=[1, 1, 1])
        #
        # # Convert PyBullet's RGBA to OpenCV's BGR format
        # rgba_array = np.array(rgba, dtype=np.uint8)
        # rgba_array = np.reshape(rgba_array, (pixelHeight, pixelWidth, 4))
        # bgr_array = cv2.cvtColor(rgba_array, cv2.COLOR_RGBA2BGR)
        #
        # # Display the image using OpenCV
        # #cv2.imshow('image', bgr_array)
        # #cv2.waitKey(2)

    def generate_equidistant_path(self, x, y, z, d):
        # Calcular la distancia total al punto dado
        total_distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        # Calcular el número de puntos necesario para lograr la distancia deseada
        num_points = int(total_distance / d) + 1

        # Generar el camino equiespaciado
        path_points = np.linspace([0, 0, 0], [x, y, z], num_points)

        return path_points
    def mission(self, data):

        t1 = time.time()
        # Especificar las dimensiones del cilindro
        cylinder_radius = 0.3
        cylinder_height = 1.3

        # Especificar la posición en metros y orientación del cilindro
        cylinder_position = [data["path"][0].x, data["path"][0].y, 0]
        cylinder_orientation = p.getQuaternionFromEuler([0, 0, 0])  # Sin rotación

        # Crear la forma del cilindro (collision shape)
        person_shape_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=cylinder_radius, height=cylinder_height)
        person_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=person_shape_id,
                                             basePosition=cylinder_position, baseOrientation=cylinder_orientation)

        # Crear la forma del cilindro (collision shape)
        obstacle_shape_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=data["box_radius"], height=cylinder_height)
        obstacle_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=obstacle_shape_id,
                                             basePosition=data["box_position"], baseOrientation=cylinder_orientation)

        if len(data["path"]) > 1:
            distance_between_p_points = np.linalg.norm(np.array([data["path"][1].x, data["path"][1].y]) - np.array([data["path"][0].x, data["path"][0].y]))

        sim_time = 0
        self.timeStep = distance_between_p_points/(data["speed"] * 8)
        self.timeStep = 1. / 240

        # print("Distance between points", distance_between_p_points, "timeStep", self.timeStep)

        # Verificar si hay puntos de contacto (impacto)
        while True:
            # print("Sim_time:", sim_time)
            person_position, _ = p.getBasePositionAndOrientation(person_id)
            person_position = np.array(person_position)
            # print("T proceso", time.time()-t1)
            if np.linalg.norm([data["path"][-1].x - person_position[0], data["path"][-1].y - person_position[1]]) < 0.1:
                print("-----------------Path completed, target point reached---------------")
                sim_time = sim_time + self.timeStep
                break

            if len(p.getContactPoints(person_id, obstacle_id)) > 0:
                print("-----------------Collision--------------------")
                person_collision_pose , _ = p.getBasePositionAndOrientation(person_id)
                p.removeBody(person_id)
                p.removeBody(obstacle_id)
                sim_time = sim_time + self.timeStep
                return True, sim_time, person_collision_pose

            speed_command = self.generate_velocity_command(data["path"], person_position, data["speed"], distance_between_p_points)

            if speed_command == [0.0, 0.0, 0.0]:
                person_collision_pose, _ = p.getBasePositionAndOrientation(person_id)
                p.removeBody(person_id)
                p.removeBody(obstacle_id)
                print("Robot reached the target")
                return False, sim_time, person_collision_pose

            p.resetBaseVelocity(person_id, linearVelocity=speed_command)
            # print("object position: ", person_position,"speed: ", speed_command ,"Speed norm", np.linalg.norm(speed_command))


            # if np.all([data["path"][-1].x-0.5, data["path"][-1].y-0.5,0] <= object_position) and np.all([data["path"][-1].x+0.5, data["path"][-1].y+0.5,0] >= object_position):
            sim_time = sim_time + self.timeStep
            p.stepSimulation()

            # time.sleep(self.timeStep)

        print("T proceso simulación completa", time.time()-t1, "sim-time", sim_time, "self time step", self.timeStep)

        person_collision_pose , _ = p.getBasePositionAndOrientation(person_id)
        p.removeBody(person_id)
        p.removeBody(obstacle_id)
        return False, sim_time, person_collision_pose

    def mouse_click(self, event, x, y, flags, param):

        # to check if left mouse  button was clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            self.yaw += 1
        if event == cv2.EVENT_RBUTTONDOWN:
            self.yaw -= 1

    def startup_check(self):
        print(f"Testing RoboCompBulletSim.TPoint from ifaces.RoboCompBulletSim")
        test = ifaces.RoboCompBulletSim.TPoint()
        print(f"Testing RoboCompBulletSim.Result from ifaces.RoboCompBulletSim")
        test = ifaces.RoboCompBulletSim.Result()
        QTimer.singleShot(200, QApplication.instance().quit)

    def find_target_point(self, path, position, d):
        """
        Encuentra el punto en el path a una distancia d por delante del robot, comenzando desde el punto más cercano en el path.

        :param path: Lista de puntos (x, y) del path.
        :param position: Posición actual (x, y) del robot.
        :param d: Distancia al punto objetivo.
        :return: Punto objetivo (x, y) en el path.
        """



        robot_pos = np.array(position)
        # print(robot_pos)

        closest_distance = np.inf
        closest_point = None
        closest_point_index = -1

        # Encuentra el punto más cercano al robot en el path
        for i, point in enumerate(path):
            # print(point)
            # print(np.array(point))
            aux = [point.x, point.y, 0]
            distance = np.linalg.norm(robot_pos - aux)
            if distance < closest_distance:
                closest_distance = distance
                closest_point = point
                closest_point_index = i

        if closest_point == path[-1]:
            return [closest_point.x, closest_point.y, 0]

        # Calcula el punto objetivo desde el punto más cercano
        target_point = None
        accumulated_distance = 0
        for i in range(closest_point_index, len(path) - 1):
            start_point = path[i]
            end_point = path[i + 1]
            segment_length = np.linalg.norm(np.array([end_point.x,end_point.y,0]) - np.array([start_point.x,start_point.y,0]))

            if accumulated_distance + segment_length > d:
                fraction = (d - accumulated_distance) / segment_length
                target_point = np.array([start_point.x,start_point.y,0]) + fraction * (np.array([end_point.x,end_point.y,0]) - np.array([start_point.x,start_point.y,0]))
                break
            accumulated_distance += segment_length

        # Si el punto objetivo no se encuentra dentro del rango, usa el último punto del path
        if target_point is None:
            target_point = np.array(path[-1])

        return target_point.tolist()

    def generate_velocity_command(self, path, robot_position, v, d):
        """
        Genera un comando de velocidad para el robot hacia un punto objetivo en el path.

        :param path: Lista de puntos (x, y) del path.
        :param robot_position: Posición actual (x, y) del robot.
        :param v: Velocidad en m/s.
        :param d: Distancia al punto objetivo.
        :return: Comando de velocidad (vx, vy).
        """

        target_point = self.find_target_point(path, robot_position, d)

        if target_point == path[-1]:
            return [0.0 , 0.0, 0.0]

        direction_vector = np.array(target_point) - np.array(robot_position)
        direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)
        velocity_command = direction_vector_normalized * v
        return velocity_command.tolist()

    def generate_ellipse_path(self, a, b, d_points, num_points):
        """
        Genera un path en forma de elipse con ejes especificados y puntos separados por una distancia aproximada.

        :param a: Semieje mayor de la elipse.
        :param b: Semieje menor de la elipse.
        :param d_points: Distancia aproximada entre puntos consecutivos en el path.
        :param num_points: Número de puntos a generar en el path.
        :return: Lista de puntos (x, y) que forman el path en elipse.
        """
        path = []
        theta = 0  # Ángulo inicial
        delta_theta = 2 * np.pi / num_points  # Incremento inicial de theta basado en el número de puntos

        for _ in range(num_points):
            x = a * np.cos(theta)  # Coordenada x
            y = b * np.sin(theta)  # Coordenada y

            path.append((x, y, 0))

            # Ajusta theta para el siguiente punto. Esta es una simplificación para mantener la separación aproximada.
            theta += delta_theta

        return path



# =============== Methods for Component Implements ==================
    # ===================================================================

    #
    # IMPLEMENTATION of simulatePath method from BulletSim interface
    #
    def BulletSim_simulatePath(self, path, speed, obstacle):
        #
        # write your CODE here
        #

        for point in path:
            point.x, point.y = point.x / 1000, point.y / 1000

        # BE CAREFUL (PYBULLET -> m   ROBOCOMP -> mm)

        collision, collision_time, collision_pose = self.mission(
            {"speed": speed,
             "box_radius": obstacle.radius / 1000,
             "box_position": [obstacle.x / 1000, obstacle.y / 1000, 0],
             "path": path
             })

        ret = ifaces.RoboCompBulletSim.Result()
        ret.collision = collision
        ret.collisionTime = collision_time
        ret.collisionPose.x = collision_pose[0]
        ret.collisionPose.y = collision_pose[1]
        ret.collisionPose.z = collision_pose[2]

        return ret

    # ===================================================================
    # ===================================================================



    ######################
    # From the RoboCompBulletSim you can use this types:
    # RoboCompBulletSim.TPoint
    # RoboCompBulletSim.Result





