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
import time

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import queue
from collections import deque
import cv2
import open3d as o3d
import copy
import math
import itertools

import interfaces as ifaces

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

from pydsr import *



class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 50

        # YOU MUST SET AN UNIQUE ID FOR THIS AGENT IN YOUR DEPLOYMENT. "_CHANGE_THIS_ID_" for a valid unique integer
        self.agent_id = 404
        self.g = DSRGraph(0, "pythonAgent", self.agent_id)

        # --------------- ODOMETRY ----------------
        self.odometry_node_id = "200"
        self.odometry_queue_len = 400
        self.odometry_queue = deque([0.0, 0.0, 0.0, 0.0], maxlen=self.odometry_queue_len)
        self.last_sim_timestamp = (0.0)
        self.odometry_node_id = 200

        # --------------- PROCESSING --------------
        self.voxel_size = 0.1
        self.max_height = 3
        self.min_height = 0.5
        # ----- remove_outliers -----
        self.nb_neighbors=500
        self.std_ratio=1

        # ----- get_hough_lines -----
        self.rho_min = -10000.0
        self.rho_max = 10000.0
        self.rho_step = 10.0
        self.theta_min = 0
        self.theta_max = np.pi
        self.theta_step = np.pi / 180
        self.min_votes = 50 #50
        self.lines_max = 150
        self.line_threshold = 1

        # ----- filter_parallel_hough_lines
        self.theta_thresh=np.pi / 4
        self.rho_thresh=2000

        # ------ get corners ------
        self.votes = 5
        self.CORNER_SUPPORT_THRESHOLD = 0.5
        self.CORNERS_PERP_ANGLE_THRESHOLD = 0.3
        self.NMS_MIN_DIST_AMONG_CORNERS = 0.5


        try:
            signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.update_node_att)
            signals.connect(self.g, signals.UPDATE_NODE, self.update_node)
            signals.connect(self.g, signals.DELETE_NODE, self.delete_node)
            signals.connect(self.g, signals.UPDATE_EDGE, self.update_edge)
            signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.update_edge_att)
            signals.connect(self.g, signals.DELETE_EDGE, self.delete_edge)
            console.print("signals connected")
        except RuntimeError as e:
            print(e)

        if startup_check:
            self.startup_check()
        else:
            self.color_palette = [
                [255, 0, 0],  # wall
                [128, 0, 0],  # building
                [0, 128, 0],  # sky
                [128, 128, 0],  # floor
                [0, 0, 128],  # tree
                [128, 0, 128],  # ceiling
                [0, 128, 128],  # road
                [128, 128, 128],  # bed
                [64, 0, 0],  # windowpane
                [192, 0, 0],  # grass
                [64, 128, 0],  # cabinet
                [192, 128, 0],  # sidewalk
                [64, 0, 128],  # person
                [192, 0, 128],  # earth
                [64, 128, 128],  # door
                [192, 128, 128],  # table
                [0, 64, 0],  # mountain
                [128, 64, 0],  # plant
                [0, 192, 0],  # curtain
                [128, 192, 0],  # chair
                [0, 64, 128],  # car
                [128, 64, 128],  # water
                [0, 192, 128],  # painting
                [128, 192, 128],  # sofa
                [64, 64, 0],  # shelf
                [192, 64, 0],  # house
                [64, 192, 0],  # sea
                [192, 192, 0],  # mirror
                [64, 64, 128],  # rug
                [192, 64, 128],  # field
                [64, 192, 128],  # armchair
                [192, 192, 128],  # seat
                [0, 0, 64],  # fence
                [128, 0, 64],  # desk
                [0, 128, 64],  # rock
                [128, 128, 64],  # wardrobe
                [0, 0, 192],  # lamp
                [128, 0, 192],  # bathtub
                [0, 128, 192],  # railing
                [128, 128, 192],  # cushion
                [64, 0, 64],  # base
                [192, 0, 64],  # box
                [64, 128, 64],  # column
                [192, 128, 64],  # signboard
                [0, 64, 64],  # chest of drawers
                [128, 64, 64],  # counter
                [0, 192, 64],  # sand
                [128, 192, 64],  # sink
                [64, 64, 128],  # skyscraper
                [192, 64, 128],  # fireplace
                [0, 192, 128],  # refrigerator
                [128, 192, 128],  # grandstand
                [64, 0, 192],  # path
                [192, 0, 192],  # stairs
                [0, 128, 192],  # runway
                [128, 128, 192],  # case
                [0, 64, 192],  # pool table
                [128, 64, 192],  # pillow
                [0, 192, 192],  # screen door
                [128, 192, 192],  # stairway
                [64, 64, 0],  # river
                [192, 64, 0],  # bridge
                [0, 192, 0],  # bookcase
                [128, 192, 0],  # blind
                [0, 64, 128],  # coffee table
                [128, 64, 128],  # toilet
                [0, 192, 128],  # flower
                [128, 192, 128],  # book
                [64, 64, 128],  # hill
                [192, 64, 128],  # bench
                [0, 192, 128],  # countertop
                [128, 192, 128],  # stove
                [64, 64, 192],  # palm
                [192, 64, 192],  # kitchen island
                [0, 192, 192],  # computer
                [128, 192, 192],  # swivel chair
                [64, 0, 64],  # boat
                [192, 0, 64],  # bar
                [0, 128, 64],  # arcade machine
                [128, 128, 64],  # hovel
                [0, 64, 192],  # bus
                [128, 64, 192],  # towel
                [0, 192, 192],  # light
                [128, 192, 192],  # truck
                [64, 64, 0],  # tower
                [192, 64, 0],  # chandelier
                [0, 192, 0],  # awning
                [128, 192, 0],  # streetlight
                [0, 64, 128],  # booth
                [128, 64, 128],  # television receiver
                [0, 192, 128],  # airplane
                [128, 192, 128],  # dirt track
                [64, 64, 192],  # apparel
                [192, 64, 192],  # pole
                [0, 192, 192],  # land
                [128, 192, 192],  # bannister
                [64, 64, 0],  # escalator
                [192, 64, 0],  # ottoman
                [0, 192, 0],  # bottle
                [128, 192, 0],  # buffet
                [0, 64, 128],  # poster
                [128, 64, 128],  # stage
                [0, 192, 128],  # van
                [128, 192, 128],  # ship
                [64, 64, 192],  # fountain
                [192, 64, 192],  # conveyer belt
                [0, 192, 192],  # canopy
                [128, 192, 192],  # washer
                [64, 64, 0],  # plaything
                [192, 64, 0],  # swimming pool
                [0, 192, 0],  # stool
                [128, 192, 0],  # barrel
                [0, 64, 128],  # basket
                [128, 64, 128],  # waterfall
                [0, 192, 128],  # tent
                [128, 192, 128],  # bag
                [64, 64, 192],  # minibike
                [192, 64, 192],  # cradle
                [0, 192, 192],  # oven
                [128, 192, 192],  # ball
                [64, 64, 0],  # food
                [192, 64, 0],  # step
                [0, 192, 0],  # tank
                [128, 192, 0],  # trade name
                [0, 64, 128],  # microwave
                [128, 64, 128],  # pot
                [0, 192, 128],  # animal
                [128, 192, 128],  # bicycle
                [64, 64, 192],  # lake
                [192, 64, 192],  # dishwasher
                [0, 192, 192],  # screen
                [128, 192, 192],  # blanket
                [64, 64, 0],  # sculpture
                [192, 64, 0],  # hood
                [0, 192, 0],  # sconce
                [128, 192, 0],  # vase
                [0, 64, 128],  # traffic light
                [128, 64, 128],  # tray
                [0, 192, 128],  # ashcan
                [128, 192, 128],  # fan
                [64, 64, 192],  # pier
                [192, 64, 192],  # crt screen
                [0, 192, 192],  # plate
                [128, 192, 192],  # monitor
                [64, 64, 0],  # bulletin board
                [192, 64, 0],  # shower
                [0, 192, 0],  # radiator
                [128, 192, 0],  # glass
                [0, 64, 128],  # clock
                [128, 64, 128],  # flag
            ]

            # --------------- VISUALIZACION ----------------
            self.w , self.scatter, self.lines, self.hough_lines, self.corners_plot = self.initialize_application()
            self.voxel_visualization = []

            self.read_deque = deque(maxlen=5)

            # ----------------

            # self.read_queue = deque(maxlen=12)
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):

        # Kill scatter plot
        self.scatter.clear()
        # Kill QGLWidget
        self.ui.viewer_3D.clear()
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
        # Get pointcloud form queue
        if self.read_deque:
            segmented_pointcloud = self.read_deque.pop()

            #

            # build points fromg segmented_pointcloud XArray, YArray, ZArray
            points = np.array([segmented_pointcloud.XArray , segmented_pointcloud.YArray, segmented_pointcloud.ZArray]).T / 1000
            # Build labels from segmented_pointcloud CategoriesArray
            labels = np.array(segmented_pointcloud.CategoriesArray)


            # Filter points categories categories_filter = [0, 1, 22, 8, 14] in labels
            categories_filter = [0, 1, 22, 8, 14, 114]

            # points = points[np.isin(labels, categories_filter)]

            points = points[ (points[:, 2] < self.max_height) & (points[:, 2] > self.min_height) & np.isin(labels, categories_filter)]
            # Filter z
            # Get room lines from pointcloud
            pcd_2D, hough_lines = self.get_room_lines_from_pointcloud(points)

            self.corners, corners_to_display = self.get_corners(hough_lines, votes=[self.votes]*len(hough_lines),
                                                                pc2d=pcd_2D,
                                                                CORNER_SUPPORT_THRESHOLD=self.CORNER_SUPPORT_THRESHOLD,
                                                                CORNERS_PERP_ANGLE_THRESHOLD=self.CORNERS_PERP_ANGLE_THRESHOLD,
                                                                NMS_MIN_DIST_AMONG_CORNERS=self.NMS_MIN_DIST_AMONG_CORNERS)
            # self.update_visualization(self.scatter, points, labels, self.lines, self.hough_lines, hough_lines, self.corners_plot, corners_to_display)
            self.update_visualization(self.scatter, self.voxel_visualization, labels, self.lines, self.hough_lines, hough_lines,
                                      self.corners_plot, corners_to_display)


    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    def get_room_lines_from_pointcloud(self, pointcloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)

        # Voxelización
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        self.voxel_visualization = np.asarray(pcd.points)

        # Filtrado de outliers

        pcd_filtered = self.remove_outliers(pcd, self.nb_neighbors, self.std_ratio)

        self.voxel_visualization = np.asarray(pcd_filtered.points)

        # Build 2D point cloud from 3D
        pcd_2D = o3d.geometry.PointCloud()
        points_2D = np.asarray(pcd_filtered.points)[:, :2]
        points_2D = np.hstack((points_2D, np.zeros((points_2D.shape[0], 1))))
        pcd_2D.points = o3d.utility.Vector3dVector(points_2D)

        # compute hough lines from 2D point cloud using opencv points hough method
        hough_lines = self.get_hough_lines(pcd_2D)

        return pcd_2D, hough_lines

    # =============== Methods for Component SubscribesTo ================
    # ===================================================================

    #
    # SUBSCRIPTION to pushLidarData method from Lidar3DPub interface
    #
    def Lidar3DPub_pushLidarData(self, lidarData):
        self.read_deque.append(lidarData)
        # # Knowing lidar_data = ifaces.RoboCompLidar3D.TDataCategory(XArray=x_array, YArray=y_array, ZArray=z_array, CategoriesArray=seg_array, period=100, timestamp=timestamp)
        # # Build the list of points and categories
        # points = []
        # categories = []
        # for i in range(len(lidarData.XArray)):
        #     # If category is 0 ansssssssd z > 0.5 append the point
        #     # if lidarData.CategoriesArray[i] == 0 and lidarData.ZArray[i] > 1000:
        #     points.append([lidarData.XArray[i], lidarData.YArray[i], lidarData.ZArray[i]])
        #     categories.append(lidarData.CategoriesArray[i])
        #
        # # update the visualization
        # self.update_visualization(self.scatter, [(points, categories)])
        # pass

    # ===================================================================
    # ===================================================================

    def initialize_application(self):
        """Inicializa la aplicación PyQt y los widgets."""

        w = gl.GLViewWidget()  # Crear una ventana de visualización
        w.show()
        w.setWindowTitle('Visualización de Puntos Aleatorios en 3D')
        w.setCameraPosition(distance=20)

        scatter = gl.GLScatterPlotItem()  # Configurar la nube de puntos
        w.addItem(scatter)

        lines = gl.GLLinePlotItem()
        w.addItem(lines)

        hough_lines = gl.GLLinePlotItem()
        w.addItem(hough_lines)

        corners = gl.GLScatterPlotItem()  # Configurar la nube de puntos
        w.addItem(corners)

    # Crear y agregar ejes
        x_axis, y_axis, z_axis = self.create_axes()
        w.addItem(x_axis)
        w.addItem(y_axis)
        w.addItem(z_axis)

        return w, scatter, lines, hough_lines, corners

    def create_axes(self):
        """Crea los ejes X, Y, Z."""
        axis_length = 1
        x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [axis_length, 0, 0]]), color=pg.glColor('r'), width=2)
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, axis_length, 0]]), color=pg.glColor('g'), width=2)
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, axis_length]]), color=pg.glColor('b'), width=2)
        return x_axis, y_axis, z_axis

    def update_visualization(self, scatter, points, labels, lines, hough_lines_plot, hough_lines, corners_plot, corners):

        colors = np.array([self.color_palette[0] for item in labels])
        print("corners", self.corners)
        # if self.corners:
        #     line_segments = []
        # #     Given list of corners, update the visualization, drawing line segments between them.
        #
        #     for votes, target in connections:
        #
        #     for i in range(len(self.corners) - 1):
        #         p1 = self.corners[i][1]
        #         p2 = self.corners[i + 1][1]
        #
        #         lines.setData(pos=np.array(p1, p2), color=(0, 1, 0, 1), width=2)
        # # else:
        # #      self.lines.setData(pos=np.zeros((0, 3)))  # Clear lines

        if self.corners:
            corner_coords = np.array([np.append(corner[1], 0) for corner in corners])

            # Plot as large red points
            corners_plot.setData(
                pos=corner_coords,
                color=(1, 1, 0, 1),  # Red
                size=10,  # Larger size
                pxMode=True  # Size in screen pixels
            )

            # Given list of corners, update the visualization, drawing line segments between them.
            line_segments = []
            for i in range(len(self.corners) - 1):
                p1 = self.corners[i][1]
                p1 = np.append(p1, 0)
                p2 = self.corners[i + 1][1]
                p2 = np.append(p2, 0)

                line_segments.append([p1, p2])
            line_segments.append([np.append(self.corners[-1][1], 0), np.append(self.corners[0][1], 0)])
            
            line_segments = np.array(line_segments).reshape(-1, 3)
            lines.setData(pos=line_segments, color=(0, 1, 0, 1), width=2)
        else:
            lines.setData(pos=np.zeros((0, 3)))  # Clear lines if no corners

        # Visualización de líneas de Hough (líneas amarillas)
        if hough_lines:
            line_length = 10.0  # Longitud de las líneas en metros
            hough_segments = []

            for rho, theta in hough_lines:
                # Convertir coordenadas polares a cartesianas
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho

                # Calcular puntos extremos de la línea
                x1 = x0 - line_length * (-b)
                y1 = y0 - line_length * a
                x2 = x0 + line_length * (-b)
                y2 = y0 + line_length * a

                # Añadir segmento como un par de puntos independiente
                # Insertamos un punto NaN para separar las líneas
                hough_segments.append([[x1, y1, -0.5], [x2, y2, -0.5], [np.nan, np.nan, np.nan]])

            # Convertir a array y aplanar, manteniendo los separadores NaN
            hough_segments = np.array(hough_segments).reshape(-1, 3)
            hough_lines_plot.setData(pos=hough_segments, color=(1, 1, 0, 1), width=1)  # Amarillo
        else:
            hough_lines_plot.setData(pos=np.zeros((0, 3)))  # Limpiar líneas si no hay resultados

        scatter.setData(pos=points, color=colors, size=3)

    def get_corners(self, lines, votes, pc2d, CORNER_SUPPORT_THRESHOLD, CORNERS_PERP_ANGLE_THRESHOLD,
                    NMS_MIN_DIST_AMONG_CORNERS):
        """
        Detects corners from lines and applies non-maximum suppression.

        :param lines: List of lines in the format [(rho, theta), ...].
        :param votes: List of votes corresponding to each line.
        :param CORNERS_PERP_ANGLE_THRESHOLD: Angle threshold for detecting perpendicular lines.
        :param NMS_MIN_DIST_AMONG_CORNERS: Minimum distance among corners for non-maximum suppression.
        :return: List of detected corners.
        """

        corners = []
        for (line1, votes_1), (line2, votes_2) in itertools.combinations(zip(lines, votes), 2):
            rho1, theta1 = line1
            rho2, theta2 = line2

            # Calculate the angle between the lines
            angle = abs(theta1 - theta2)
            if angle > np.pi:
                angle -= np.pi
            if angle < -np.pi:
                angle += np.pi

            # Check if the lines are perpendicular within the threshold
            if (np.pi / 2 - CORNERS_PERP_ANGLE_THRESHOLD < angle < np.pi / 2 + CORNERS_PERP_ANGLE_THRESHOLD):
                # Find the intersection point of the lines
                A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
                b = np.array([rho1, rho2])
                if np.linalg.det(A) != 0:  # Check if lines are not parallel
                    intersection = np.linalg.solve(A, b)
                    corners.append((min(votes_1, votes_2), intersection))

        corners_to_display = corners.copy()

        # Sort corners by votes
        corners.sort(key=lambda x: x[0], reverse=True)
        # Non-maximum suppression
        filtered_corners = []
        points_2D = np.asarray(pc2d.points)

        # Remove from corners those that are not supported by points_2D
        for corner in corners:
            if np.any(np.linalg.norm(points_2D[:, :2] - corner[1], axis=1) < CORNER_SUPPORT_THRESHOLD):
                filtered_corners.append(corner)

        print(f"Filtered corners: {len(filtered_corners)}")
        print(corners)

        # Re-order corners TODO: Fix, not working on complex rooms
        sorted_corners = self.sort_corners(filtered_corners)


        return sorted_corners, corners_to_display

    def sort_corners(self, corners):
        """
        Sorts the given points (corners) to form a closed polyline that outlines the perimeter of a shape.
        The function returns the corners in the same format as input (score, np.array([x, y])).

        Parameters:
          corners: list of tuples, where each tuple is of the form (score, np.array([x, y])).
                   The score is ignored during the sorting.

        Returns:
          A list of tuples in the same format as the input, sorted to form the perimeter polyline.
        """
        # Extract only the coordinate parts for computing the centroid
        points = [c[1] for c in corners]

        # Calculate the centroid (average of the coordinates)
        centroid = np.mean(np.array(points), axis=0)

        # Sort the corners based on the angle from the centroid,
        # using each corner's coordinate to compute the angle.
        sorted_corners = sorted(corners, key=lambda c: math.atan2(c[1][1] - centroid[1], c[1][0] - centroid[0]))

        return sorted_corners

    def remove_outliers(self,pcd, nb_neighbors=200, std_ratio=0.3):
        """
        Filtra outliers usando un criterio estadístico basado en la desviación estándar.

        :param pcd: Nube de puntos Open3D
        :param nb_neighbors: Número de vecinos a considerar por punto
        :param std_ratio: Umbral de desviación estándar
        :return: Nube de puntos filtrada
        """
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return cl

    def get_hough_lines(self, pcd):
        """
        Calcula las líneas de Hough a partir de un Point Cloud 2D usando cv2.HoughLinesPointSet.

        :param pcd: open3d.geometry.PointCloud con puntos 2D en metros.
        :return: Lista de líneas detectadas en formato [(rho, theta), ...] en metros.
        """
        # Obtener puntos del point cloud
        points = np.asarray(pcd.points)

        # print(points)
        # Asegurar que el point cloud es 2D (descartar tercera coordenada si existe)
        if points.shape[1] > 2:
            points = points[:, :2]

        # Convertir los puntos de metros a milímetros
        points_mm = points * 1000.0

        # Normalizar puntos a flotantes para OpenCV (requerido por HoughLinesPointSet)
        points_float = np.round(points_mm).astype(np.float32).reshape(-1, 1, 2)

        # print(points_float)
        # Parámetros del algoritmo Hough en milímetros


        lines = cv2.HoughLinesPointSet(
            points_float, self.lines_max,  self.line_threshold,  self.rho_min,  self.rho_max,  self.rho_step,  self.theta_min,  self.theta_max,  self.theta_step
        )

        # For each line, find points within threshold distance
        results = []
        line_threshold_m = self.line_threshold / 1000.0  # Convert threshold to meters
        print(lines)

        filtered_voted_lines = [line for line in lines if line[0][0] >= self.min_votes]

        print(filtered_voted_lines)
        # for line in lines[:, 0]:
        #     votes, rho_mm, theta = line
        #     rho = rho_mm / 1000.0  # Convert to meters
        #
        #     # print votes & min_votes
        #     print("votes", votes, "min_votes", self.min_votes)
        #
        #     # Calculate distance from each point to the line
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     distances = np.abs(points[:, 0] * a + points[:, 1] * b - rho)
        #     voting_indices = np.where(distances <= line_threshold_m)[0]
        #     voting_points = points[voting_indices]
        #
        #     if len(voting_points) < 2:
        #         continue  # Skip lines with insufficient points
        #
        #     # Project points onto the line to find extremes
        #     # Using parametric projection: t = x*a + y*b
        #     t_values = voting_points[:, 0] * a + voting_points[:, 1] * b
        #
        #     # Find points with min and max projection values
        #     min_idx = np.argmin(t_values)
        #     max_idx = np.argmax(t_values)
        #
        #     endpoint1 = voting_points[min_idx]
        #     endpoint2 = voting_points[max_idx]
        #
        #     # Calculate segment length
        #     segment_length = np.linalg.norm(endpoint2 - endpoint1)
        #
        #     # Append to resul
        #
        #     results.append((
        #         np.array([votes, rho, theta])
        #         # ,
        #         # {
        #         #     'voting_points': voting_points,
        #         #     'endpoint1': endpoint1,
        #         #     'endpoint2': endpoint2,
        #         #     'length': segment_length
        #         # }
        #     ))

        # print("RESULTS", results)



        # TODO: Perform filtering of parallel lines
        # Print number of lines detected prefiltering
        if lines is not None:
            print(f"Se detectaron {len(lines)} líneas antes del filtrado.")

        filtered_lines = self.filter_parallel_hough_lines(filtered_voted_lines, self.theta_thresh, self.rho_thresh)

        # Print number of lines detected postfiltering
        if filtered_lines is not None:
            print(f"Se detectaron {len(filtered_lines)} líneas.")
            return [(rho / 1000.0, theta) for _, rho, theta in filtered_lines[:, 0]]  # Convertir rho de mm a metros
        else:
            return []

    def filter_parallel_hough_lines(self, lines, theta_thresh=np.pi / 2, rho_thresh=2000):
        """
        Filtra las líneas de Hough eliminando las paralelas, agrupándolas por theta y rho,
        y devolviendo la más votada de cada grupo.

        :param lines: Lista de líneas en formato [(votos, rho, theta)] (rho en metros).
                      Las líneas pueden estar anidadas (por ejemplo, arrays con forma (1, 3)).
        :param theta_thresh: Umbral de diferencia angular en radianes para considerar líneas paralelas.
        :param rho_thresh: Umbral de diferencia de distancia para considerar líneas redundantes.
        :return: Lista con las líneas más votadas por grupo [(rho, theta, votos)].
        """
        groups = []  # Cada grupo es una lista de líneas similares

        # Sort lines by votes in descending order

        # Sort lines by votes in descending order
        lines = sorted(lines, key=lambda x: x[0] if np.isscalar(x[0]) else x[0].item(0), reverse=True)

        for line in lines:
            # Aplanar la línea en caso de que esté anidada
            if isinstance(line, np.ndarray):
                line = line.flatten()
            elif isinstance(line, list) and len(line) == 1:
                line = line[0]

            # Verifica que la línea tenga tres elementos
            if len(line) != 3:
                raise ValueError(f"La línea no tiene tres elementos: {line}")

            votes, rho, theta = line
            added = False

            # Revisamos si la línea actual se agrupa con alguna ya existente
            for group in groups:

                # Tomamos la primera línea del grupo como referencia
                ref_votes, ref_rho, ref_theta = group[0]

                # Convert angles from [0, 2pi] to [-pi, pi] for comparison
                theta = (theta + np.pi) % (2 * np.pi) - np.pi
                ref_theta = (ref_theta + np.pi) % (2 * np.pi) - np.pi

                if abs(theta - ref_theta) < theta_thresh or abs(abs(theta - ref_theta) - np.pi) < theta_thresh:
                    if abs(rho - ref_rho) < rho_thresh:
                        group.append((votes, rho, theta))
                        added = True
                        break


                # if abs(theta - ref_theta) < theta_thresh and abs(rho - ref_rho) < rho_thresh:
                #     # print Group added
                #     group.append((votes, rho, theta))
                #     added = True
                #     break

            # Si no se agrupa con ninguno, se crea un nuevo grupo
            if not added:
                groups.append([[votes, rho, theta]])

        # De cada grupo, se selecciona la línea con mayor número de votos
        best_lines = []
        for group in groups:
            # Print number of lines in group
            print(f"Grupo con {len(group)} líneas.")
            best_line = max(group, key=lambda l: l[0])
            # Pring best line
            print(f"Mejor línea: {best_line}")

            # Append best line to best_lines in tuple format (votes, rho, theta)
            best_lines.append([best_line])

        return np.array(best_lines)


    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        # console.print(f"UPDATE NODE ATT: {id} {attribute_names}", style='green')
        if id == self.odometry_node_id:
            odom_node = self.g.get_node("Shadow")
            odom_attrs = odom_node.attrs
            self.odometry_queue.append([odom_attrs["robot_current_advance_speed"].value,
                                        odom_attrs["robot_current_side_speed"].value,
                                        odom_attrs["robot_current_angular_speed"].value,
                                        odom_attrs["timestamp_alivetime"].value])
        pass

    def update_node(self, id: int, type: str):
        # console.print(f"UPDATE NODE: {id} {type}", style='green')
        pass
    def delete_node(self, id: int):
        # console.print(f"DELETE NODE:: {id} ", style='green')
        pass
    def update_edge(self, fr: int, to: int, type: str):

        # console.print(f"UPDATE EDGE: {fr} to {type}", type, style='green')
        pass
    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        # console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')
        pass
    def delete_edge(self, fr: int, to: int, type: str):
        # console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
        pass


