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
from numpy.polynomial.polynomial import polyline
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
import cupy as cp
from room_model import RoomModel
import interfaces as ifaces
import torch

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

from pydsr import *



class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 100

        # YOU MUST SET AN UNIQUE ID FOR THIS AGENT IN YOUR DEPLOYMENT. "_CHANGE_THIS_ID_" for a valid unique integer
        self.agent_id = 404
        self.g = DSRGraph(0, "pythonAgent", self.agent_id)

        # --------------- ODOMETRY ----------------
        self.odometry_node_id = "200"
        self.odometry_queue_len = 10000
        self.odometry_queue = deque(maxlen=self.odometry_queue_len)
        self.last_sim_timestamp = (0.0)
        self.odometry_node_id = 200

        # --------------- PROCESSING --------------
        self.voxel_size = 0.1
        self.max_height = 3.2
        self.min_height = 1.2
        # ----- remove_outliers -----
        self.nb_neighbors=400
        self.std_ratio=1

        # ----- get_hough_lines -----
        self.rho_min = -10000.0
        self.rho_max = 10000.0
        self.rho_step = 5.0
        self.theta_min = 0
        self.theta_max = np.pi
        self.theta_step = np.pi / 180
        self.min_votes = 1 / self.voxel_size
        self.lines_max = 150
        self.line_threshold = 1

        # ----- filter_parallel_hough_lines
        self.theta_thresh=np.pi / 4
        self.rho_thresh=500

        # ------ get corners ------
        self.votes = 5
        self.CORNER_SUPPORT_THRESHOLD = 0.5
        self.CORNERS_PERP_ANGLE_THRESHOLD = np.deg2rad(10)
        self.NMS_MIN_DIST_AMONG_CORNERS = 0.5
        self.SUPPORT_DISTANCE_THRESHOLD = self.voxel_size
        self.SUPPORT_PERCENTAGE_THRESHOLD = 0.55
        self.MAX_WALL_HOLE_WIDTH = 1.2
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
            o3d.core.Device("CUDA:0")
            self.hide()
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

            # ----------------
            self.read_deque = deque(maxlen=1)
            # Filter points categories self.categories_filter = [0, 1, 22, 8, 14] in labels
            self.categories_filter = [0, 1, 22, 8, 14, 114]
            self.accumulated_pcs = np.empty((0, 3), dtype=np.float32)
            self.accumulated_labels = np.empty((0, 1), dtype=np.uint8)
            self.last_pointcloud = None
            self.last_pointcloud_exists = False
            self.last_robot_pose_timestamp = 0.0

            # # Initialize the visualizer
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name='Real-Time 3D Point Cloud', height=480, width=640)
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            self.vis.add_geometry(axes)

            # Add a floor
            floor = o3d.geometry.TriangleMesh.create_box(width=10, height=10, depth=0.1)
            floor.translate([-5, -5, -0.1])  # Adjust position
            floor.paint_uniform_color([0.8, 0.8, 0.8])  # Set color to light gray
            self.vis.add_geometry(floor)

            self.pcd = o3d.geometry.PointCloud()
            points = np.random.rand(3, 3)
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.vis.add_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()

            # Set up the camera
            view_control = self.vis.get_view_control()
            view_control.set_zoom(35)  # Adjust zoom level

            # Add robot geometry
            # Load the Shadow .obj mesh
            self.shadow_mesh = o3d.io.read_triangle_mesh("../../etc/meshes/shadow.obj", print_progress=True)
            self.shadow_mesh.paint_uniform_color([1, 0, 1])
            self.vis.add_geometry(self.shadow_mesh)

            self.exists_room = False
            self.room = None

            # self.read_queue = deque(maxlen=12)
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    #       ------------------- POSE-------------------------
            self.robot_pose = np.array([0.0, 0.0, 0.0])

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
        if self.read_deque:

            segmented_pointcloud = self.read_deque.pop()
            category = np.array(segmented_pointcloud.CategoriesArray).flatten()
            # Generate np.array from new_pc arrayX, arrayY, arrayZ
            new_pc = np.column_stack(
                [np.array(segmented_pointcloud.XArray), np.array(segmented_pointcloud.YArray), np.array(segmented_pointcloud.ZArray)]) / 1000.0

            # PC filter Category & height
            height_mask = (new_pc[:, 2] < self.max_height) & (new_pc[:, 2] > self.min_height)
            category_mask = np.isin(category, self.categories_filter)
            new_pc = new_pc[category_mask & height_mask]

            # PC alignment and integration


            transformed_points = self.integrate_odometry_to_pointcloud(new_pc, segmented_pointcloud.timestamp)

            pcd = o3d.geometry.PointCloud()
            # print("Integrated odom point cloud size", len(self.accumulated_pcs), "new_pc size", len(transformed_points))
            self.accumulated_pcs = np.vstack((self.accumulated_pcs, transformed_points))
            pcd.points = o3d.utility.Vector3dVector(self.accumulated_pcs)
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

            self.accumulated_pcs = np.asarray(pcd.points).copy()
            # print("LEN", len(self.accumulated_pcs))
            # Get Hough lines
            pcd_2D, hough_lines = self.get_room_lines_from_pointcloud(pcd)

            print("Hough lines", hough_lines)

            # Compute corners
            polyline, corners_to_display, _, _, _ = self.get_corners(hough_lines, pc2d=pcd_2D)

            # Transform polyline to a np.array of 2d points
            # print(polyline)
            corner_poses = np.array([corner[1] for corner in polyline])
            #
            lidar_points = torch.tensor(np.array(self.accumulated_pcs, dtype=np.float32), dtype=torch.float32,
                                        device="cuda")  # Convert to meters
            if len(corner_poses) >= 4:
                if not self.exists_room:
                # Generate room model
                    self.room = RoomModel(corner_poses, height=2.5, device="cuda")
                    unexplained_points, mesh_colored, face_counts = self.room.project_points_with_pytorch3d(lidar_points)
                    self.draw_room(self.room.forward() , unexplained_points, face_counts)
                    self.exists_room = True
            elif self.exists_room:
                unexplained_points, mesh_colored, face_counts = self.room.project_points_with_pytorch3d(lidar_points)
                self.draw_room(self.room.forward() , unexplained_points, face_counts)
                print("Face counts", face_counts)
                self.draw_point_cloud(unexplained_points)
            # Update visualization
            self.vis.poll_events()
            self.vis.update_renderer()
            self.update_visualization(self.scatter, self.accumulated_pcs, self.lines, self.hough_lines, hough_lines,
                                      self.corners_plot, polyline)


    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    def plot_mesh_open3d(self, mesh):
        verts = mesh.verts_packed().cpu().numpy()
        faces = mesh.faces_packed().cpu().numpy()

        # Create Open3D TriangleMesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

        # Compute normals for better visualization
        o3d_mesh.compute_vertex_normals()

        # Visualize
        o3d.visualization.draw_geometries([o3d_mesh])

    def get_room_lines_from_pointcloud(self, pcd):
        """
        Obtains lines representing room walls from a 3D point cloud using the Hough transform.

        :param pcd: Open3D 3D point cloud
        :return: Tuple (2D_point_cloud, hough_lines)
        """

        # Outlier filtering code (disabled)
        # pcd_filtered = self.remove_outliers(pcd, self.nb_neighbors, self.std_ratio)
        # self.voxel_visualization = np.asarray(pcd_filtered.points)

        # Create a 2D point cloud from the original 3D cloud
        pcd_2D = o3d.geometry.PointCloud()

        # Extract only X and Y coordinates (top-down projection)
        points_2D = np.asarray(pcd.points)[:, :2]

        # Add Z=0 coordinates to maintain the three-dimensional format required by Open3D
        points_2D = np.hstack((points_2D, np.zeros((points_2D.shape[0], 1))))

        # Assign the points to the Open3D PointCloud object
        pcd_2D.points = o3d.utility.Vector3dVector(points_2D)

        # print("PCD 2D points", len(pcd_2D.points), pcd_2D.points)

        # Calculate lines using the Hough transform method implemented in get_hough_lines
        # which internally uses OpenCV
        hough_lines = self.get_hough_lines(pcd_2D)

        return pcd_2D, hough_lines

    # =============== Methods for Component SubscribesTo ================
    # ===================================================================

    #
    # SUBSCRIPTION to pushLidarData method from Lidar3DPub interface
    #
    def Lidar3DPub_pushLidarData(self, lidarData):
        self.read_deque.append(lidarData)



    # ===================================================================
    # ===================================================================

    def initialize_application(self):
        """Inicializa la aplicación PyQt y los widgets."""

        w = gl.GLViewWidget()  # Crear una ventana de visualización
        w.show()
        w.setWindowTitle('Visualización de Puntos Aleatorios en 3D')

        # Set camera zenital at 5m
        w.setCameraPosition(distance=15, elevation=90, azimuth=270)


        empty_array = np.zeros((1, 3), dtype=np.float32)
        scatter = gl.GLScatterPlotItem(pos=empty_array, size=5)  # Configurar la nube de puntos
        w.addItem(scatter)


        # Create empty line plots for lines and hough_lines
        lines = gl.GLLinePlotItem(pos=empty_array, color=(1, 0, 0, 1), width=2)
        # lines = gl.GLLinePlotItem()
        w.addItem(lines)

        hough_lines = gl.GLLinePlotItem(pos=empty_array, color=(1, 0, 0, 1), width=2)
        # w.addItem(hough_lines)


        corners = gl.GLScatterPlotItem(pos=empty_array, size=5)  # Configurar la nube de puntos
        # w.addItem(corners)

        # Create robot-aligned axes (initially at origin)
        self.robot_axes = self.create_robot_axes()
        for axis in self.robot_axes:
            w.addItem(axis)

    # Crear y agregar ejes
        x_axis, y_axis, z_axis = self.create_axes()
        w.addItem(x_axis)
        w.addItem(y_axis)
        w.addItem(z_axis)

        return w, scatter, lines, hough_lines, corners

    def create_robot_axes(self):
        """Create XYZ axes that will follow robot pose."""
        # X axis (red)
        x_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [1, 0, 0]]),
            color=(1, 0, 0, 1),  # Red
            width=3,
            antialias=True
        )

        # Y axis (green)
        y_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 1, 0]]),
            color=(0, 1, 0, 1),  # Green
            width=3,
            antialias=True
        )

        # Z axis (blue)
        z_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 1]]),
            color=(0, 0, 1, 1),  # Blue
            width=3,
            antialias=True
        )

        return [x_axis, y_axis, z_axis]

    def create_axes(self):
        """Crea los ejes X, Y, Z."""
        axis_length = 1
        x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [axis_length, 0, 0]]), color=pg.glColor('r'), width=2)
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, axis_length, 0]]), color=pg.glColor('g'), width=2)
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, axis_length]]), color=pg.glColor('b'), width=2)
        return x_axis, y_axis, z_axis

    def update_visualization(self, scatter, points, lines, hough_lines_plot, hough_lines, corners_plot, corners):

        # If self.corners len > 0, plot the corners
        if len(corners) > 0:
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
            for i in range(len(corners) - 1):
                p1 = corners[i][1]
                p1 = np.append(p1, 0)
                p2 = corners[i + 1][1]
                p2 = np.append(p2, 0)

                line_segments.append([p1, p2])
            # line_segments.append([np.append(self.corners[-1][1], 0), np.append(self.corners[0][1], 0)])
            #
            line_segments = np.array(line_segments).reshape(-1, 3)
            lines.setData(pos=line_segments, color=(0, 1, 0, 1), width=2)
        else:
            lines.setData(pos=np.zeros((0, 3)))  # Clear lines if no corners

        # Visualización de líneas de Hough (líneas amarillas)
        if hough_lines:
            line_length = 10.0  # Longitud de las líneas en metros
            hough_segments = []

            for _, rho, theta in hough_lines:
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
        scatter.setData(pos=np.asarray(points), color=(1, 1, 1, 1), size=3)


    def get_corners(self, lines, pc2d):
        """
        Detects corners from lines and applies non-maximum suppression.
        Also creates dictionaries that relate corners with lines and vice versa for later search
        and extracts the polyline that delimits the contour from these dictionaries.
        :param lines: List of lines in format [(votes, rho, theta), ...].
        :param pc2d: 2D point cloud (e.g., o3d.geometry.PointCloud object) of the lidar scene.
        :return: tuple containing:
                 - polyline: List of corners forming the contour.
                 - corners: List of all detected corners.
                 - corner_to_lines: Dictionary relating each corner (key: intersection as tuple) with lines.
                 - line_to_corners: Dictionary relating each line with the corners it contains.
                 - close_loop: Boolean indicating if the polyline forms a closed loop.
        """
        # Initialize collections
        corners = []
        corner_to_lines = {}  # Dictionary relating corners to their lines
        line_to_corners = {}  # Dictionary relating lines to their corners

        # Extract corners from line combinations
        for line1, line2 in itertools.combinations(lines, 2):
            # Each line is given as (votes, rho, theta)
            votes_1, rho1, theta1 = line1
            votes_2, rho2, theta2 = line2

            # Calculate the angle difference between the two lines
            angle = abs(theta1 - theta2)
            if angle > np.pi:
                angle -= np.pi
            if angle < -np.pi:
                angle += np.pi

            # Check if the lines are approximately perpendicular
            if np.pi / 2 - self.CORNERS_PERP_ANGLE_THRESHOLD < angle < np.pi / 2 + self.CORNERS_PERP_ANGLE_THRESHOLD:
                # Solve the equation system to get the intersection
                A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
                b = np.array([rho1, rho2])

                # Check that lines are not parallel (determinant != 0)
                if np.linalg.det(A) != 0:
                    intersection = np.linalg.solve(A, b)
                    # Convert intersection to tuple to use as key in dictionaries
                    intersection_tuple = (float(intersection[0]), float(intersection[1]))
                    # Save the corner with its minimum vote and the two lines that generated it
                    corner_entry = (min(votes_1, votes_2), intersection, (line1, line2))
                    corners.append(corner_entry)

                    # Update corner_to_lines: key is intersection, value is a set of lines
                    if intersection_tuple not in corner_to_lines:
                        corner_to_lines[intersection_tuple] = set()
                    corner_to_lines[intersection_tuple].update([line1, line2])

                    # Update line_to_corners: each line relates to the corner
                    for ln in [line1, line2]:
                        if ln not in line_to_corners:
                            line_to_corners[ln] = []
                        line_to_corners[ln].append(intersection_tuple)

        # Sort corners by votes in descending order
        corners.sort(key=lambda x: x[0], reverse=True)

        # Convert 2D point cloud to numpy array for faster calculations
        pc2d_points = np.asarray(pc2d.points)

        # Extract the polyline that delimits the contour using the created dictionaries
        polyline = []
        if len(corners) == 0:
            return [], corners, corner_to_lines, line_to_corners, False

        # Select the highest-voted corner as starting point
        start_corner = corners[0]
        polyline.append(start_corner)
        used_corners = set()  # Track used corners

        current_corner = start_corner
        finished = False
        max_iterations = len(corners)  # To avoid infinite loops
        iteration = 0
        close_loop = False

        # Main loop to build the polyline
        while not finished and iteration < max_iterations:
            iteration += 1
            found_next = False

            # Get the tuple representation of the current corner
            current_key = (float(current_corner[1][0]), float(current_corner[1][1]))

            # Get all lines that belong to the current corner
            lines_of_current = corner_to_lines.get(current_key, set())

            # For each of these lines, find associated corners as candidates
            candidates = set()
            for ln in lines_of_current:
                for candidate_key in line_to_corners.get(ln, []):
                    if candidate_key != current_key:
                        candidates.add(candidate_key)

            # Sort candidates by distance to the current corner
            candidates = sorted(candidates, key=lambda x: np.linalg.norm(np.array(x) - np.array(current_key)))

            # Review candidates that haven't been used yet and have support in the segment
            for candidate_key in candidates:
                # Find the corner in the list with that intersection
                candidate = next((c for c in corners if (float(c[1][0]), float(c[1][1])) == candidate_key), None)
                if candidate is None or candidate_key in used_corners:
                    continue

                p_current = current_corner[1]
                p_candidate = candidate[1]

                # Check if the segment has support in the point cloud
                if self.has_support(p_current, p_candidate, pc2d_points):
                    if len(polyline) > 1:
                        # Calculate angle between current segment and new segment
                        p_prev = polyline[-2][1]
                        vec_current = p_candidate - p_current
                        vec_prev = p_current - p_prev
                        angle = np.arctan2(vec_current[1], vec_current[0]) - np.arctan2(vec_prev[1], vec_prev[0])

                        # Check if angle is within tolerance of expected angles (0°, ±90°, ±270°) avoiding 180° (returning to previous)
                        tolerance = 10
                        angle_deg = np.rad2deg(angle)
                        if any(abs(angle_deg - target) <= tolerance for target in [-90, 0, 90, 270, -270]):
                            polyline.append(candidate)
                            used_corners.add(candidate_key)
                            current_corner = candidate
                            found_next = True

                            # Check if we've closed the loop by returning to the first corner
                            if len(polyline) > 2 and candidate_key == (float(polyline[0][1][0]), float(polyline[0][1][1])):
                                finished = True
                                close_loop = True
                            break
                    else:
                        # First segment, just add the candidate
                        polyline.append(candidate)
                        used_corners.add(candidate_key)
                        current_corner = candidate
                        found_next = True
                        break

            # If no valid candidate is found, exit the current iteration
            if not found_next:
                finished = True

        return polyline, corners, corner_to_lines, line_to_corners, close_loop

    def share_line(self, corner_a, corner_b):
        """
        Helper function: checks if two corners share at least one line.

        :param corner_a: First corner in format (score, intersection, (line1, line2))
        :param corner_b: Second corner in format (score, intersection, (line1, line2))
        :return: Boolean indicating if corners share at least one line
        """
        _, _, lines_a = corner_a
        _, _, lines_b = corner_b
        # Check the intersection of line tuples from both corners
        return (lines_a[0] in lines_b) or (lines_a[1] in lines_b)

    def has_support(self, p1, p2, pc2d_points):
        """
        Helper function: verifies that a segment between two points has support in the point cloud.
        Divides the p1-p2 segment into N samples and checks if enough points are close to the cloud.

        :param p1: First point of the segment
        :param p2: Second point of the segment
        :param pc2d_points: Point cloud in which to check for support
        :return: Boolean indicating if the segment has enough support points
        """
        # Calculate distance between p1 and p2
        distance = np.linalg.norm(p2 - p1)

        # Calculate number of samples needed
        N = int(distance / self.voxel_size)

        if N == 0:
            return False

        # Generate points along the segment
        samples = np.linspace(0, 1, N)[:, None] * (p2 - p1) + p1
        support_count = 0

        # Maximum allowed consecutive samples without support (to detect holes in walls)
        continuous_samples = self.MAX_WALL_HOLE_WIDTH / self.voxel_size
        samples_counter = continuous_samples

        # Use Euclidean distance for each sample against all pc2d points
        for s in samples:
            if samples_counter == 0:
                # Maximum number of consecutive samples without support has been reached
                return False

            # Calculate distances between the sample and pc2d points
            distances = np.linalg.norm(pc2d_points[:, :2] - s, axis=1)

            # If any point is close enough, it's considered support
            if np.any(distances < self.SUPPORT_DISTANCE_THRESHOLD):
                support_count += 1
                samples_counter = continuous_samples
            else:
                samples_counter -= 1

        # Return True if the support percentage exceeds the threshold
        return support_count / N >= self.SUPPORT_PERCENTAGE_THRESHOLD


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
        Calculates Hough lines from a 2D Point Cloud using cv2.HoughLinesPointSet.

        :param pcd: open3d.geometry.PointCloud with 2D points in meters.
        :return: List of detected lines in format [(votes, rho, theta), ...] in meters.
        """
        # Get points from the point cloud
        points = np.asarray(pcd.points)

        # Ensure the point cloud is 2D (discard third coordinate if it exists)
        if points.shape[1] > 2:
            points = points[:, :2]

        # Convert points from meters to millimeters
        points_mm = points * 1000.0

        # Normalize points to float for OpenCV (required by HoughLinesPointSet)
        points_float = np.round(points_mm).astype(np.float32).reshape(-1, 1, 2)

        # Apply HoughLinesPointSet
        lines = cv2.HoughLinesPointSet(
            points_float, self.lines_max, self.line_threshold, self.rho_min, self.rho_max, self.rho_step,
            self.theta_min, self.theta_max, self.theta_step
        )

        # Votes filter
        filtered_voted_lines = [line for line in lines if line[0][0] >= self.min_votes]
        # Parallel filter
        filtered_lines = self.filter_parallel_hough_lines(filtered_voted_lines, self.theta_thresh, self.rho_thresh)

        if filtered_lines is not None:
            return [(votes, rho / 1000.0, theta) for votes, rho, theta in
                    filtered_lines[:, 0]]  # Convert rho from mm to meters
        else:
            return []

    def filter_parallel_hough_lines(self, lines, theta_thresh=np.pi / 2, rho_thresh=2000):
        """
        Filters Hough lines by removing parallel ones, grouping them by theta and rho,
        and returning the most voted line from each group.

        :param lines: List of lines in format [(votes, rho, theta)] (rho in meters).
                     Lines may be nested (e.g., arrays with shape (1, 3)).
        :param theta_thresh: Angular difference threshold in radians to consider lines as parallel.
        :param rho_thresh: Distance difference threshold to consider lines as redundant.
        :return: List with the most voted lines per group [(votes, rho, theta)].
        """
        groups = []  # Each group is a list of similar lines

        # Sort lines by votes in descending order
        lines = sorted(lines, key=lambda x: x[0] if np.isscalar(x[0]) else x[0].item(0), reverse=True)

        for line in lines:
            if isinstance(line, np.ndarray):
                line = line.flatten()
            elif isinstance(line, list) and len(line) == 1:
                line = line[0]

            if len(line) != 3:
                raise ValueError(f"Line does not have three elements: {line}")

            votes, rho, theta = line
            added = False

            # Check if current line belongs to any existing group
            for group in groups:
                # Take the first line of the group as reference
                ref_votes, ref_rho, ref_theta = group[0]

                # Normalize angles to range [-pi, pi]
                theta = (theta + np.pi) % (2 * np.pi) - np.pi
                ref_theta = (ref_theta + np.pi) % (2 * np.pi) - np.pi

                # Check if lines are parallel (considering lines with opposite direction)
                parallel = abs(theta - ref_theta) < theta_thresh or abs(abs(theta - ref_theta) - np.pi) < theta_thresh

                # Check if they are close enough in rho (to avoid separate groups)
                same_rho = abs(rho - ref_rho) < rho_thresh or abs(abs(rho) - abs(ref_rho)) < rho_thresh

                if parallel and same_rho:
                    group.append((votes, rho, theta))
                    added = True
                    break

            # If not grouped with any existing one, create a new group
            if not added:
                groups.append([[votes, rho, theta]])

        # From each group, select the line with the highest number of votes
        best_lines = []
        for group in groups:
            best_line = max(group, key=lambda l: l[0])
            best_lines.append([best_line])

        return np.array(best_lines)

    def integrate_odometry_to_pointcloud(self, new_pc, timestamp):
        # # copy odometry_list to avoid modifying the original
        queue_copy = self.get_odometry_simple(np.array(self.odometry_queue.copy()), self.last_robot_pose_timestamp, timestamp)

        if not queue_copy:
            print("No odometry values or invalid format")
            return None

        # # Print first and last timestamp of the queue_copy, self.last_robot_pose_timestamp and timestamp and length of queue_copy
        # print("Queue copy timestamps", queue_copy[0][3], queue_copy[-1][3], self.last_robot_pose_timestamp, timestamp, len(queue_copy))
        # # Print all values of queue_copy
        # for i in range(len(queue_copy)):
        #     print(queue_copy[i][0], queue_copy[i][1], queue_copy[i][2], queue_copy[i][3])
        # Construir accum 4x4 desde self.robot_pose
        # get adv, side, angular from self.robot_pose
        x = self.robot_pose[0]
        y = self.robot_pose[1]
        angle = self.robot_pose[2]

        # Build 4x4 transformation matrix using adv, side and rot
        cos_rot = np.cos(angle)
        sin_rot = np.sin(angle)

        orig_robot_pose = np.array([
            [cos_rot.item(), -sin_rot.item(), 0, x.item()],
            [sin_rot.item(), cos_rot.item(), 0, y.item()],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        prev_time = queue_copy[0][3]
        accum = np.eye(4)

        # Odometry integration
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
            ])
            # Accumulate the transformation
            accum = np.dot(accum, T)
            prev_time = curr_time
        self.update_robot_axes(accum)

        # # Transform every three first elements of the new_pc and buil homogeneus matrix
        homogeneous_points = np.hstack([new_pc, np.ones((new_pc.shape[0], 1))])
        #
        point_cloud = np.dot(accum, homogeneous_points.T).T
        #
        # # Normalize homogenous component to get new_pc 3xN (divide by last element)
        transformed_points = point_cloud[:, :3] / point_cloud[:, 3][:, None]

        return transformed_points # Return as numpy array if needed

    def get_odometry_simple(self, queue, last_ts, pc_ts):
        lower = min(float(last_ts), float(pc_ts))
        upper = max(float(last_ts), float(pc_ts))
        return [odom for odom in queue if lower <= float(odom[3]) <= upper]

    def add_unique_points(self, total, actual, radius=0.01):
        """
        Add points from actual to total that don't exist within radius in total using Open3D.

        Args:
            total: numpy array of shape (n, 3) containing existing points
            actual: numpy array of shape (m, 3) containing new candidate points
            radius: maximum distance to consider a point as existing (default 0.01)

        Returns:
            A new numpy array combining total with unique points from actual
        """
        if len(total) == 0:
            return actual.copy()

        if len(actual) == 0:
            return total.copy()

        # Convert arrays to Open3D point clouds
        total_pcd = o3d.geometry.PointCloud()
        total_pcd.points = o3d.utility.Vector3dVector(total)

        actual_pcd = o3d.geometry.PointCloud()
        actual_pcd.points = o3d.utility.Vector3dVector(actual)

        # Build KDTree for fast radius search
        kdtree = o3d.geometry.KDTreeFlann(total_pcd)

        # Find points in actual that don't have neighbors in total within radius
        unique_mask = []
        for point in actual:
            _, idx, _ = kdtree.search_radius_vector_3d(point, radius)
            unique_mask.append(len(idx) == 0)  # True if no neighbors found

        unique_mask = np.array(unique_mask)

        # Combine the arrays
        return np.vstack([total, actual[unique_mask]])

    ############### DRAW ##################
    def draw_point_cloud(self, points):
        """
        Update the point cloud with new data.
        :param points: Nx3 numpy array of 3D points.
        """
        # Concatenate all clusters
        #all_clusters = np.concatenate(clusters)
        points = points.cpu().numpy()
        self.pcd.points = o3d.utility.Vector3dVector(points)
        colors = np.tile([0.0, 1.0, 0.0], (points.shape[0], 1))  # Green color for each point
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def draw_room(self, mesh_colored, unexplained_points: torch.Tensor, face_counts: torch.Tensor):
        """
        Visualiza la malla con colores por cara y los puntos no explicados como nube de puntos.

        Args:
            mesh_colored: pytorch3d.structures.Meshes con vértices y caras definidos.
            unexplained_points: (N, 3) tensor con puntos LiDAR no explicados.
            face_counts: (F,) tensor con la cantidad de puntos explicados por cada cara.
        """
        device = unexplained_points.device
        verts = mesh_colored.verts_packed()
        faces = mesh_colored.faces_packed()
        F = faces.shape[0]

        # Normalize counts to [0,1] range
        max_count = face_counts.max() if face_counts.max() > 0 else 1
        normalized_counts = face_counts.float() / max_count

        # Create a clear color gradient from blue (low) to red (high)
        # face_colors = torch.stack([
        #     normalized_counts,  # Red component increases with count
        #     torch.zeros_like(normalized_counts),  # No green
        #     1.0 - normalized_counts  # Blue component decreases with count
        # ], dim=1)

        # Alternative: stepped colors for better distinction
        num_steps = 5
        step = torch.floor(normalized_counts * num_steps) / num_steps
        face_colors = torch.stack([
            step,                           # Red increases in steps
            (1.0 - step) * 0.5,             # Some green for mid-range
            1.0 - step                      # Blue decreases in steps
        ], dim=1)

        # Assign colors to vertices (average of adjacent faces)
        vert_colors = torch.zeros_like(verts)
        counts_per_vertex = torch.zeros(verts.shape[0], device=device)

        for i in range(F):
            for j in range(3):
                vidx = faces[i, j]
                vert_colors[vidx] += face_colors[i]
                counts_per_vertex[vidx] += 1

        vert_colors = vert_colors / (counts_per_vertex.unsqueeze(-1) + 1e-8)

        # Convertir a Open3D
        verts_np = verts.detach().cpu().numpy()
        faces_np = faces.detach().cpu().numpy()
        colors_np = vert_colors.detach().cpu().numpy()
        unexpl_np = unexplained_points.detach().cpu().numpy()

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts_np)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces_np)
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors_np)
        o3d_mesh.compute_vertex_normals()

        # Nube de puntos de los puntos no explicados
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(unexpl_np)
        pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Rojo

        # Visualizar
        self.vis.clear_geometries()
        self.vis.get_render_option().mesh_show_back_face = True
        self.vis.add_geometry(o3d_mesh, reset_bounding_box=True)
        self.vis.add_geometry(pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    # def draw_room(self, room):
    #     """
    #     Convert the PyTorch3D mesh from the fridge model to an Open3D mesh and visualize it.
    #     """
    #     # self.vis.clear_geometries()
    #
    #     # Get the PyTorch3D mesh
    #     mesh = room.forward()
    #     verts = mesh.verts_list()[0].detach().cpu().numpy()
    #     faces = mesh.faces_list()[0].detach().cpu().numpy()
    #
    #     # Create Open3D mesh
    #     o3d_mesh = o3d.geometry.TriangleMesh()
    #     o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    #     o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    #
    #     # Compute normals and apply color
    #     o3d_mesh.compute_vertex_normals()
    #     o3d_mesh.paint_uniform_color([0.0, 0.5, 1.0])  # Light blue fridge
    #
    #     # Añadir geometría y actualizar
    #     self.vis.get_render_option().mesh_show_back_face = True
    #     self.vis.add_geometry(o3d_mesh, reset_bounding_box=True)
    #     self.vis.poll_events()
    #     self.vis.update_renderer()

    def update_robot_axes(self, pose):
        """
        Update robot axes position and orientation based on current pose.

        Args:
            pose: (x, y, z, roll, pitch, yaw) or 4x4 transformation matrix
        """
        if isinstance(pose, (tuple, list)) and len(pose) == 6:
            # Convert Euler angles to rotation matrix
            x, y, z, roll, pitch, yaw = pose
            transform = self.euler_to_matrix(roll, pitch, yaw)
            transform[:3, 3] = [x, y, z]  # Set translation
        else:
            transform = pose  # Assume it's already a 4x4 matrix

        # Base axis endpoints (before transformation)
        base_points = [
            np.array([[0, 0, 0], [1, 0, 0]]),  # X
            np.array([[0, 0, 0], [0, 1, 0]]),  # Y
            np.array([[0, 0, 0], [0, 0, 1]])  # Z
        ]

        # Apply transformation to each axis
        for i, axis in enumerate(self.robot_axes):
            # Transform points
            homogeneous = np.column_stack([base_points[i], np.ones(2)])
            transformed = (transform @ homogeneous.T).T[:, :3]
            axis.setData(pos=transformed)

    def apply_transformation(self, points, transformation_matrix):
        """
        Aplica una matriz de transformación 4x4 a una nube de puntos.

        :param points: Nube de puntos como matriz Nx3 (coordenadas X, Y, Z).
        :param transformation_matrix: Matriz de transformación 4x4 que incluye rotación y traslación.
        :return: Nube de puntos transformada.
        """
        # points = np.asarray(points)

        # Create homogeneous coordinates (X, Y, Z, 1)
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

        # Apply transformation (all operations stay on GPU)
        transformed_xyz = np.dot(transformation_matrix, points_homogeneous.T).T

        # Normalize homogeneous component to get new_pc 3xN (divide by last element)
        transformed_xyz_normalize = transformed_xyz[:, :3] / transformed_xyz[:, 3][:, None]

        # Combine with original Z coordinates
        return transformed_xyz_normalize

    # def apply_transformation(self, points, transformation_matrix):
    #     """
    #     Aplica una matriz de transformación 3x3 a una nube de puntos.
    #
    #     :param points: Nube de puntos como matriz Nx3 (coordenadas X, Y, Z).
    #     :param transformation_matrix: Matriz de transformación 3x3 que incluye rotación y traslación.
    #     :return: Nube de puntos transformada.
    #     """
    #     # Print first point on points, points shape, and points size
    #     # print("First point", points[0])
    #     # print("Points shape", points.shape)
    #     # print("Points size", points.size)
    #
    #     # Agregar una columna de 1s a los puntos para que sea una matriz Nx3 homogénea (X, Y, Z, 1)
    #     points = cp.asarray(points)
    #     transformation_matrix = cp.asarray(transformation_matrix)
    #
    #     # Create homogeneous coordinates (X, Y, 1)
    #     ones = cp.ones((points.shape[0], 1), dtype=cp.float32)
    #     points_homogeneous = cp.hstack([points[:, :2], ones])
    #
    #     # Apply transformation (all operations stay on GPU)
    #     transformed_xy = cp.dot(transformation_matrix, points_homogeneous.T).T
    #
    #     # Combine with original Z coordinates
    #     return cp.hstack([transformed_xy[:, :2], points[:, 2:]])

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


