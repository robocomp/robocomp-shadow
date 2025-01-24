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
        self.Period = 30

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

        # Configurar la interfaz gráfica
        self.init_plot()

        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):

        return True

    @QtCore.Slot()
    def compute(self):
        # Initialize repulsion
        self.repulsion = np.array([0.0, 0.0])
        self.speed_line = np.array([[0, 0], [0, 0]])

        # Check if joystick data is available
        if self.adv_j != 0.0 or self.side_j != 0.0 or self.rot_j != 0.0:
            self.adv = self.adv_j
            self.side = self.side_j
            self.rot = self.rot_j
            print("joystick")
            # Send command to robot
            # try:
            #     self.omnirobot_proxy.setSpeedBase(self.side, self.adv, self.rot)
            # except Ice.Exception as e:
            #     print("Error sending speed to robot", e)
            # return

        t1 = time.time()

        try:
            # Speed vector (adv, rot)
            dt = 1
            dy = (self.adv / 1000 * np.cos(self.rot) - self.side / 1000 * np.sin(self.rot)) * dt
            dx = (self.adv / 1000 * np.sin(self.rot) + self.side / 1000 * np.cos(self.rot)) * dt
            self.speed = np.array([dx, dy])

            # build vector line
            d = 3
            resolution = 0.1
            N_points = int(d / resolution)

            self.VOXEL_SIZE = 0.1

            # Get Lidar data
            ldata_ = self.lidar3d_proxy.getLidarDataWithThreshold2d("bpearl", 3000, 2)
            # for ldata in ldata_: build a point cloud
            self.pcd = o3d.geometry.PointCloud()
            # Print Ldata.points size
            for ldata in ldata_.points:
                self.pcd.points.append([ldata.x / 1000, ldata.y / 1000, ldata.z / 1000])

            # Voxelize the point cloud
            self.pcd = self.pcd.voxel_down_sample(voxel_size=self.VOXEL_SIZE)

            # DBSCAN clustering of point cloud

            self.pcd_np = np.asarray(self.pcd.points)
            clustering = DBSCAN(eps=0.7, min_samples=20).fit(self.pcd_np)
            self.labels = clustering.labels_

            self.clusters = []

            for i in range(max(self.labels) + 1):
                cluster = self.pcd_np[self.labels == i]
                centroid = np.mean(cluster, axis=0)
                # compute radius
                radius = np.max(np.linalg.norm(cluster - centroid, axis=1))

                # If speed != 0
                # if np.linalg.norm(self.speed) > 0:
                if(np.linalg.norm(self.speed)>0):
                    v_dir = self.speed / (np.linalg.norm(self.speed))* resolution
                    speed_line = np.array([np.array([0, 0]), v_dir * N_points])



                    # TODO: One call to point_to_line
                    # Compute closest point from cluster to speed line
                    closest_point = cluster[np.argmin([self.point_to_line_distance(point, speed_line[-1], speed_line[0])[0] for point in cluster])]
                    dist, nearest = self.point_to_line_distance(closest_point, speed_line[1], speed_line[0])

                    #Sigmoide parameters
                    lateral_th = 0.7
                    advance_th = 0.5

                    #Security distances TODO: Tune this values
                    side_th = 1 + 1 * abs(self.rot / 0.78)
                    front_th = 1.5 + abs(self.adv / 750)


                    # if dist < side_th and np.linalg.norm(nearest[:2]) < front_th:
                    # Calcular los factores k1 y k2 usando funciones sigmoides
                    k1 = self.sigmoid(dist, lateral_th)
                    k2 = self.sigmoid(np.linalg.norm(nearest[:2]), advance_th)

                    # Calcular la repulsión optimizada
                    repulsion = (nearest[:2] - closest_point[:2])
                    repulsion = repulsion / np.linalg.norm(repulsion)

                    # Escalar por el inverso de la distancia
                    repulsion =  k1 * k2 * repulsion / (
                                dist + 1e-6)  # Avoid zero division

                    # Clip repulsion vector to MAX_REPULSION using clip function
                    MAX_REPULSION = 1
                    repulsion = np.clip(repulsion, -MAX_REPULSION, MAX_REPULSION)

                    # Print repulsion vector
                    print("Repulsion vector", repulsion)

                    self.clusters.append((centroid, radius, closest_point[:2], nearest[:2], repulsion))


            speed_r = [0.0,0.0]
            if self.clusters:
                # Compute sum of repulsion vectors
                self.repulsion = np.sum([cluster[4] for cluster in self.clusters], axis=0)
                speed_r += repulsion

            print("Time to compute", time.time() - t1)
            # Send command to robot
            # self.omnirobot_proxy.setSpeedBase(self.adv, self.side, self.rot)
            # Get adv, side and rot from speed_r

            self.adv = self.adv - self.speed[1]
            self.rot = self.rot
            self.side = speed_r[0] * 300

            try:
                self.omnirobot_proxy.setSpeedBase(self.side, self.adv, self.rot)
            except Ice.Exception as e:
                print("Error sending speed to robot", e)

        except Ice.Exception as e:
          print("Error reading Lidar",e)

        # # ------------------- Plotting -------------------
        self.update_plot()

        return True

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

    def sigmoid(self, x, threshold):
        return 1 / (1 + np.exp(x - threshold))

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

        self.side_j = 0.0
        self.adv_j = 0.0
        self.rot_j = 0.0

        self.adv = 0.0
        self.side = 0.0
        self.rot = 0.0
        # Take joystick data as an external. It comes in m/s, so we need to scale it to mm/s
        for axis in data.axes:
            if axis.name == "rotate":
                self.rot_j = axis.value
            elif axis.name == "advance":
                self.adv_j = axis.value
            elif axis.name == "side":
                self.side_j = 0.0
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


    #
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


    # ===================================================================
    # ===================================================================


    ######################
    # From the RoboCompLidar3D you can call this methods:
    # self.lidar3d_proxy.getLidarData(...)
    # self.lidar3d_proxy.getLidarDataArrayProyectedInImage(...)
    # self.lidar3d_proxy.getLidarDataProyectedInImage(...)
    # self.lidar3d_proxy.getLidarDataWithThreshold2d(...)

    ######################
    # From the RoboCompLidar3D you can use this types:
    # RoboCompLidar3D.TPoint
    # RoboCompLidar3D.TDataImage
    # RoboCompLidar3D.TData

    ######################
    # From the RoboCompOmniRobot you can call this methods:
    # self.omnirobot_proxy.correctOdometer(...)
    # self.omnirobot_proxy.getBasePose(...)
    # self.omnirobot_proxy.getBaseState(...)
    # self.omnirobot_proxy.resetOdometer(...)
    # self.omnirobot_proxy.setOdometer(...)
    # self.omnirobot_proxy.setOdometerPose(...)
    # self.omnirobot_proxy.setSpeedBase(...)
    # self.omnirobot_proxy.stopBase(...)

    ######################
    # From the RoboCompOmniRobot you can use this types:
    # RoboCompOmniRobot.TMechParams

    ######################
    # From the RoboCompGridPlanner you can use this types:
    # RoboCompGridPlanner.TPoint
    # RoboCompGridPlanner.TControl
    # RoboCompGridPlanner.TPlan

    ######################
    # From the RoboCompJoystickAdapter you can use this types:
    # RoboCompJoystickAdapter.AxisParams
    # RoboCompJoystickAdapter.ButtonParams
    # RoboCompJoystickAdapter.TData


