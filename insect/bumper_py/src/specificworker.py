#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2025 by YOUR NAME HERE
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
from codecs import replace_errors
from warnings import catch_warnings

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from pyautogui import sleep
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import open3d as o3d
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

class SpecificWorker(GenericWorker):

    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.pcd = None
        self.VOXEL_SIZE = None
        self.Period = 15

        # Control
        self.adv = 0.0
        self.side = 0.0
        self.rot = 0.0

        self.adv_j = 0.0
        self.side_j = 0.0
        self.rot_j = 0.0

        # Animation
        self.fig, self.ax = None, None
        self.scatter = None
        self.robot_rect = None
        self.arrows = []

        # Datos inicializados
        self.pcd_np = np.array([])
        self.labels = np.array([])
        self.clusters = []
        self.repulsion = (0, 0)
        self.speed_line = np.array([])
        self.speed_r = (0, 0, 0)
        self.speed = (0, 0, 0)

        # Sigmoide parameters
        self.lateral_th = 1
        self.advance_th = 1.5
        self.angle_th = np.radians(90)
        self.brake_dist = 0.3

        # Configurar la interfaz gráfica
        self.init_plot()

        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        # Send command to robot
        try:
            self.omnirobot_proxy.setSpeedBase(0.0, 0.0, 0.0)
            print("BUMPER __DEL__: Stopping robot")
        except Ice.Exception as e:
            print("Error sending speed to robot", e)
        """Destructor"""

    def setParams(self, params):
        self.display = True
        return True

    @QtCore.Slot()
    def compute(self):
        print("---------------------------------------------------------------")
        # Initialize repulsion
        self.repulsion = np.array([0.0, 0.0])
        self.speed_line = np.array([[0, 0], [0, 0]])

        #TODO: Implement to override omnirobot with joystick
        # Check if joystick data is available
        # if self.adv_j != 0.0 or self.side_j != 0.0 or self.rot_j != 0.0:
        self.adv = self.adv_j
        self.side = self.side_j
        self.rot = self.rot_j
        # else:
            # Send command to robot
            # try:
            #     self.omnirobot_proxy.setSpeedBase(self.side, self.adv, self.rot)
            # except Ice.Exception as e:
            #     print("Error sending speed to robot", e)
            # return

        t1 = time.time()
        ldata_ = None
        try:
            ldata_ = self.lidar3d_proxy.getLidarDataWithThreshold2d("bpearl", 1800, 2)
        except Ice.Exception as e:
            print("Error reading Lidar, stopping robot", e)
            # Send stop command to robot
            try:
                self.omnirobot_proxy.setSpeedBase(0.0, 0.0, 0.0)
            except Ice.Exception as e:
                print("Error sending speed to robot", e)
            return


        # TODO: Control ldata values
        if len(ldata_.points) == 0:  # If ldata is not None
            return
        else:
            # print("Lidar data received")
            try:
                # Speed vector (adv, rot)
                dt = 1
                dy = (self.adv / 1000 * np.cos(self.rot) - self.side / 1000 * np.sin(self.rot)) * dt
                dx = (self.adv / 1000 * np.sin(self.rot) + self.side / 1000 * np.cos(self.rot)) * dt
                self.speed = np.array([dx, dy])

                # TODO: Move to params
                # build vector line
                d = 3
                resolution = 0.2
                n_points = int(d / resolution)

                # Build point cloud
                self.pcd = o3d.geometry.PointCloud()
                for ldata in ldata_.points:
                    self.pcd.points.append([ldata.x / 1000, ldata.y / 1000, ldata.z / 1000])

                # Voxelize the point cloud
                self.VOXEL_SIZE = 0.1
                self.pcd = self.pcd.voxel_down_sample(voxel_size=self.VOXEL_SIZE)


                # DBSCAN clustering of point cloud
                self.pcd_np = np.asarray(self.pcd.points)
                clustering = DBSCAN(eps=0.7, min_samples=1).fit(self.pcd_np)
                self.labels = clustering.labels_

                self.clusters = []
                k_brake = 1
                for i in range(max(self.labels) + 1):
                    cluster = self.pcd_np[self.labels == i]
                    centroid = np.mean(cluster, axis=0)
                    # compute radius
                    radius = np.max(np.linalg.norm(cluster - centroid, axis=1))


                    if np.linalg.norm(self.speed) > 0 :
                        v_dir = self.speed / (np.linalg.norm(self.speed))* resolution
                        speed_line = np.array([np.array([0, 0]), v_dir * n_points])

                        # Compute closest point from cluster to speed line
                        closest_point = cluster[np.argmin(
                            [self.point_to_line_distance(point, speed_line[-1], speed_line[0])[0] for point in cluster])]
                        dist, nearest = self.point_to_line_distance(closest_point, speed_line[1], speed_line[0])
                        dist_to_nearest = np.linalg.norm(nearest[:2])

                        # If distance is too large, ignore cluster
                        if dist > self.lateral_th * 1.5 or np.linalg.norm(nearest) > self.advance_th * 1.5:
                            cluster_repulsion = np.array([0, 0])
                            self.clusters.append((centroid, radius, closest_point[:2], nearest[:2], cluster_repulsion))
                            continue

                        # Calcular los factores k1 y k2 usando funciones sigmoides
                        k1 = self.sigmoid(dist, self.lateral_th, 2) # Lateral threshold
                        k2 = self.sigmoid(dist_to_nearest, self.advance_th,2) # Advance threshold
                        k3 = self.calculate_reduction_factor(closest_point,self.speed, self.angle_th, 2) # Angle threshold

                        # Distance to obstacles brake
                        if dist_to_nearest > 0:
                            k_ = (1 - (self.sigmoid(dist_to_nearest, self.brake_dist, 2)) * pow(k3,2))
                            if k_ < k_brake :
                                k_brake = k_
                        print("dist_to nearest", dist_to_nearest)
                        print("kS lateral, avance, angulo", k1, k2, k3, k_brake)

                        # Calcular la repulsión optimizada
                        cluster_repulsion = (nearest[:2] - closest_point[:2])
                        cluster_repulsion = k1 * k2 * k3 * cluster_repulsion / (
                                    dist + 1e-5)


                        direction = closest_point[:2] - centroid[
                                                        :2]  # Dirección desde el centroide hacia el closest_point
                        sign = np.sign(np.dot(cluster_repulsion, direction))  # Verificar si están alineados
                        cluster_repulsion *= sign  # Ajustar el signo si es necesario

                        # Scale
                        cluster_repulsion =  k1 * k2 * k3 * cluster_repulsion   # Avoid zero division

                        # Clip
                        MAX_REPULSION = 1
                        cluster_repulsion = np.clip(cluster_repulsion, -MAX_REPULSION, MAX_REPULSION)

                        # Print repulsion vector
                        print("Repulsion vector", cluster_repulsion)

                        self.clusters.append((centroid, radius, closest_point[:2], nearest[:2], cluster_repulsion))

                # Compute speed vector given repulsion vectors
                self.speed_r = [0.0,0.0]
                if self.clusters:
                    # # Compute sum of repulsion vectors
                    self.repulsion = np.sum([cluster[4] for cluster in self.clusters], axis=0)
                    self.speed_r += self.repulsion

                    # Print self.adv, self.side, self.rot
                    print("Speed", self.adv, self.side, self.rot)

                    # TODO: FIX
                    print("k_brake", k_brake)
                    self.adv = self.adv * k_brake
                    self.rot = self.rot # np.arctan2(speed_r[1],speed_r[0])
                    self.side = self.side + self.speed_r[0] * 1000
                    print("Speed + Repulsion", self.adv, self.side, self.rot)

                # Send command to robot
                try:
                    self.omnirobot_proxy.setSpeedBase(self.side, self.adv, self.rot)
                except Ice.Exception as e:
                    print("Error sending speed to robot", e)

                # print("Time to compute", time.time() - t1)

            except Ice.Exception as e:
              print("Error reading Lidar",e)

        # print("TIEMPO", time.time() - t1)
        # # ------------------- Plotting -------------------
        if(self.display):
            self.update_plot()

        # Clear values
        self.pcd = None
        self.pcd_np = np.array([])
        self.labels = np.array([])
        self.clusters = []

        return True



    # ===================================================================
    # ===================================================================
    # --------------------- AUXILIARY FUNCTIONS ---------------------


    def point_to_line_distance(self, point, line_start, line_end):
        """Calcula la distancia de un punto a una línea definida por dos puntos."""
        if point.shape[0] == 3 and line_start.shape[0] == 2:
            line_start = np.append(line_start, 0)
            line_end = np.append(line_end, 0)

        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)

        if line_len == 0:
            return None, None

        line_unitvec = line_vec / line_len
        point_vec_scaled = point_vec / line_len
        t = np.dot(line_unitvec, point_vec_scaled)
        t = np.clip(t, 0, 1)
        nearest = line_start + t * line_vec
        distance = np.linalg.norm(nearest - point)

        return distance, nearest

    def calculate_reduction_factor(self, closest_point, speed, angulo_corte=np.radians(45), k=2):
        """
        Calcula el factor de reducción k basado en el ángulo entre closest_point y speed usando una función sigmoide.

        :param closest_point: np.array, vector del punto más cercano
        :param speed: np.array, vector de velocidad
        :param angulo_corte: float, ángulo de corte en radianes para la sigmoide
        :return: float, factor de reducción k
        """
        # Asegurarse de que ambos vectores tengan la misma dimensión
        if closest_point.shape[0] == 3:
            closest_point = closest_point[:2]

        # Normalizar los vectores
        norm_closest_point = closest_point / np.linalg.norm(closest_point)
        norm_speed = speed / np.linalg.norm(speed)

        # Calcular el ángulo usando el producto punto
        cos_angle = np.dot(norm_closest_point, norm_speed)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        # Calcular el factor de reducción k usando la función sigmoide existente
        k = self.sigmoid(abs(angle), angulo_corte, k)

        return k

    def sigmoid(self, x, threshold, k=1):
        return 1 / (1 + np.exp(k * (x - threshold)))

    # ===================================================================
    # ===================================================================
    # --------------------- PLOTTING FUNCTIONS ---------------------

    def init_plot(self):
        """Inicializa la figura de matplotlib dentro del entorno de Qt."""
        self.plot_widget = QWidget()
        self.layout = QVBoxLayout(self.plot_widget)

        # Crear figura y canvas
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        # Crear el eje
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-4, 4)

        # Elementos gráficos iniciales
        self.scatter = self.ax.scatter([], [], cmap='tab20')
        self.robot_rect = self.ax.add_patch(
            plt.Rectangle((-0.25, -0.25), 0.5, 0.5, fill=False, color='red')
        )

        self.arrows = []

        # Mostrar el widget
        self.plot_widget.show()


    def update_plot(self):
        """Actualiza los elementos gráficos con los nuevos datos."""
        # Limpiar flechas previas y elementos adicionales
        for arrow in self.arrows:
            arrow.remove()
        self.arrows = []
        self.ax.clear()  # Limpiar completamente los ejes

        # Ajustar los límites del gráfico
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(-3, 3)

        # Comprobar si hay datos de puntos y etiquetas
        if self.pcd_np.size > 0 and len(self.labels) > 0:
            # Calcular colores para los puntos del Lidar
            max_label = max(self.labels) if max(self.labels) > 0 else 1  # Asegurarse de que max_label no sea 0
            colors = plt.get_cmap("tab20")(self.labels / max_label)  # Normalizar las etiquetas
            self.ax.scatter(self.pcd_np[:, 0], self.pcd_np[:, 1], c=colors[:, :3])
        else:
            print("Advertencia: No hay datos disponibles para los puntos del Lidar o etiquetas.")

        # Dibujar el robot como un rectángulo rojo
        robot_rect = plt.Rectangle((-0.25, -0.25), 0.5, 0.5, fill=False, color='red')
        self.ax.add_patch(robot_rect)

        # Dibujar flecha de repulsión
        if np.any(self.repulsion):
            # Clear repulsion arrow

            self.arrows.append(
                self.ax.arrow(0, 0, self.repulsion[0], self.repulsion[1],
                              head_width=0.05, head_length=0.1, fc='r', ec='r')
            )

        # Dibujar línea de velocidad
        if self.speed_line.size > 0:
            self.ax.plot(self.speed_line[:, 0], self.speed_line[:, 1], color='blue')

        # Dibujar clusters como círculos
        for cluster in self.clusters:
            centroid, radius, closest_point, nearest_point, _ = cluster
            circle = plt.Circle(centroid[:2], radius, fill=False, color='black')
            self.ax.add_patch(circle)

            # Dibujar puntos de interés en el cluster
            self.ax.scatter(closest_point[0], closest_point[1], color='red', label='Closest Point')
            self.ax.scatter(nearest_point[0], nearest_point[1], color='blue', label='Nearest Point')

        # Dibujar flecha de velocidad del robot en repulsión
        self.arrows.append(
            self.ax.arrow(0, 0, self.speed_r[0], self.speed_r[1],
                          head_width=0.2, head_length=0.2, fc='r', ec='r')
        )

        # Dibujar flecha de velocidad general del robot
        self.arrows.append(
            self.ax.arrow(0, 0, self.speed[0], self.speed[1],
                          head_width=0.1, head_length=0.1, fc='orange', ec='g')
        )

        # Redibujar el canvas
        self.canvas.draw()

    # ===================================================================
    # ===================================================================
    # --------------------- ROBOCOMP FUNCTIONS ---------------------
    # ===================================================================
    # ===================================================================

    def startup_check(self):
        print(f"Testing RoboCompLidar3D.TPoint from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TPoint()
        print(f"Testing RoboCompLidar3D.TDataImage from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TDataImage()
        print(f"Testing RoboCompLidar3D.TData from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TData()
        print(f"Testing RoboCompOmniRobot.TMechParams from ifaces.RoboCompOmniRobot")
        test = ifaces.RoboCompOmniRobot.TMechParams()
        print(f"Testing RoboCompGridPlanner.TPoint from ifaces.RoboCompGridPlanner")
        test = ifaces.RoboCompGridPlanner.TPoint()
        print(f"Testing RoboCompGridPlanner.TControl from ifaces.RoboCompGridPlanner")
        test = ifaces.RoboCompGridPlanner.TControl()
        print(f"Testing RoboCompGridPlanner.TPlan from ifaces.RoboCompGridPlanner")
        test = ifaces.RoboCompGridPlanner.TPlan()
        print(f"Testing RoboCompJoystickAdapter.AxisParams from ifaces.RoboCompJoystickAdapter")
        test = ifaces.RoboCompJoystickAdapter.AxisParams()
        print(f"Testing RoboCompJoystickAdapter.ButtonParams from ifaces.RoboCompJoystickAdapter")
        test = ifaces.RoboCompJoystickAdapter.ButtonParams()
        print(f"Testing RoboCompJoystickAdapter.TData from ifaces.RoboCompJoystickAdapter")
        test = ifaces.RoboCompJoystickAdapter.TData()
        QTimer.singleShot(200, QApplication.instance().quit)


    # =============== Methods for Component SubscribesTo ================
    # ===================================================================

    #
    # SUBSCRIPTION to sendData method from JoystickAdapter interface
    #
    def JoystickAdapter_sendData(self, data):

        # TODO: Implement to override
        self.side_j = 0.0
        self.adv_j = 0.0
        self.rot_j = 0.0

        # Take joystick data as an external. It comes in m/s, so we need to scale it to mm/s
        for axis in data.axes:
            if axis.name == "rotate":
                self.rot_j = axis.value
            elif axis.name == "advance":
                self.adv_j = axis.value
            elif axis.name == "side":
                self.side_j = axis.value
            else:
                print(f"[ JoystickAdapter ] Warning: Using a non-defined axes ({axis.name}).")
        pass


    # ===================================================================
    # ===================================================================

    # =============== Methods for Component Implements ==================
    # ===================================================================

    #
    # IMPLEMENTATION of modifyPlan method from GridPlanner interface
    #
    def GridPlanner_modifyPlan(self, plan):
        ret = ifaces.RoboCompGridPlanner.TPlan()
        #
        # write your CODE here
        #
        return ret
    #
    # IMPLEMENTATION of setPlan method from GridPlanner interface
    #
    def GridPlanner_setPlan(self, plan):

        #
        # write your CODE here
        #
        pass



    # IMPLEMENTATION of correctOdometer method from OmniRobot interface
    #
    def OmniRobot_correctOdometer(self, x, z, alpha):

        #
        # write your CODE here
        #
        pass


    #
    # IMPLEMENTATION of getBasePose method from OmniRobot interface
    #
    def OmniRobot_getBasePose(self):

        #
        # write your CODE here
        #
        return [x, z, alpha]
    #
    # IMPLEMENTATION of getBaseState method from OmniRobot interface
    #
    def OmniRobot_getBaseState(self):

        #
        # write your CODE here
        #
        state = RoboCompGenericBase.TBaseState()
        return state
    #
    # IMPLEMENTATION of resetOdometer method from OmniRobot interface
    #
    def OmniRobot_resetOdometer(self):

        #
        # write your CODE here
        #
        pass


    #
    # IMPLEMENTATION of setOdometer method from OmniRobot interface
    #
    def OmniRobot_setOdometer(self, state):

        #
        # write your CODE here
        #
        pass


    #
    # IMPLEMENTATION of setOdometerPose method from OmniRobot interface
    #
    def OmniRobot_setOdometerPose(self, x, z, alpha):

        #
        # write your CODE here
        #
        pass


    #
    # IMPLEMENTATION of setSpeedBase method from OmniRobot interface
    #
    def OmniRobot_setSpeedBase(self, advx, advz, rot):

        self.side = 0.0
        self.adv = advz
        self.rot = rot
        print("setSpeedBase", advx, advz, rot)
        pass


    #
    # IMPLEMENTATION of stopBase method from OmniRobot interface
    #
    def OmniRobot_stopBase(self):

        #
        # write your CODE here
        #
        pass



