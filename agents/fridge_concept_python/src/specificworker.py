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

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import numpy as np
import time
from shapely.geometry import Polygon, Point
import open3d as o3d
import torch
import torch.optim as optim

console = Console(highlight=False)

from pydsr import *
from fridge_model import FridgeModel

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.room_drawn = None  # flag to draw room only once
        self.f = None   # just one fridge
        self.Period = 100

        # YOU MUST SET AN UNIQUE ID FOR THIS AGENT IN YOUR DEPLOYMENT. "_CHANGE_THIS_ID_" for a valid unique integer
        self.agent_id = 600
        self.g = DSRGraph(0, "pythonAgent", self.agent_id)

        try:
            # signals.connect(self.g, signals.UPDATE_NODE_ATTR, self.update_node_att)
            # signals.connect(self.g, signals.UPDATE_NODE, self.update_node)
            # signals.connect(self.g, signals.DELETE_NODE, self.delete_node)
            # signals.connect(self.g, signals.UPDATE_EDGE, self.update_edge)
            # signals.connect(self.g, signals.UPDATE_EDGE_ATTR, self.update_edge_att)
            # signals.connect(self.g, signals.DELETE_EDGE, self.delete_edge)
            console.print("signals connected")
        except RuntimeError as e:
            print(e)

        self.rt_api = rt_api(self.g)
        self.inner_api = inner_api(self.g)

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

        room_node, robot_node, robot_pose = self.check_robot_and_room_exist()

        # draw room mesh
        self.draw_room(room_node)

        # Get lidar data
        timestamp, lidar_data = self.read_lidar_helios()

        # filter points by room polygon
        residuals = self.filter_points_by_room(room_node, robot_pose, lidar_data)
        #print("residuals", residuals.shape, "number of points:", residuals.shape[0])

        MIN_RESIDUALS = 100
        if residuals.size < MIN_RESIDUALS:
            console.print("Not enough residuals", style='red')
            return

        residuals = torch.tensor(np.array(residuals, dtype=np.float32), dtype=torch.float32, device="cuda") / 1000.0  # Convert to meters
        if self.f is not None:
            residuals = self.f.remove_explained_points(residuals, 0.1)

        print("residuals", residuals.shape, "number of points:", residuals.shape[0])
        # Get fridges from graph
        # fridge_nodes = self.g.get_nodes_by_type("fridge")
        # if fridge_nodes:
        #     for f in fridge_nodes:
        #         model = FridgeModel(f)
        #         residuals = model.project_points(residuals)
        #     if residuals.size < MIN_RESIDUALS:
        #         console.print("Not enough residuals after projecting fridges", style='red')
        #         return

        # generate clusters from residuals
        #clusters = self.get_clusters(residuals, 500, 25)
        #print("Cluster", clusters.shape)

        if self.f is None:
            self.f = self.initialize_fridge(room_node, residuals)

        self.draw_point_cloud(residuals)

    ##########################################################################################333
    def check_robot_and_room_exist(self):
        if self.g.get_nodes_by_type("room") is None or self.g.get_node("Shadow") is None:
            console.print("Shadow or room node not found", style='red')
            return False
        room_node = self.g.get_nodes_by_type("room")[0]
        robot_node = self.g.get_node("Shadow")
        return room_node, robot_node, self.read_robot_pose(room_node, robot_node)

    def read_lidar_helios(self):
        try:
            lidar_data = self.lidar3d_proxy.getLidarData("helios", 0, 2 * np.pi, 2)
            if lidar_data is None:
                console.print("Lidar data is None")
                return -1, np.array([])
            p_filter = np.array([np.array([point.x, point.y, point.z]) for point in lidar_data.points if point.z > 0.1 and point.distance2d > 200])
            return lidar_data.timestamp, p_filter
        except Ice.Exception as e:
            console.print_exception(e)
            console.log("Error reading lidar data")
            return -1, np.array([])

    def filter_points_by_room(self, room_node, robot_pose, lidar_data):
        # Get room polygon
        room_width, room_depth, room_polygon = self.generate_room_polygon(room_node)

        # Transform lidar data to room frame
        transformed_lidar_data = self.transform_to_room_frame(lidar_data, robot_pose)

        # Get residuals unexplained by the room polygon
        return self.project_points_to_polygon(transformed_lidar_data, room_polygon, 200)

    def read_robot_pose(self, room_node, robot_node):
        robot_edge_rt = self.rt_api.get_edge_RT(room_node, robot_node.id)
        robot_tx, robot_ty, robot_tz = robot_edge_rt.attrs['rt_translation'].value
        robot_rx, robot_ry, robot_rz = robot_edge_rt.attrs['rt_rotation_euler_xyz'].value

        # Convert Euler angles (in radians) to a 3D rotation matrix
        def euler_to_rotation_matrix(rx, ry, rz):
            # Rotation around the X axis
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)]
            ])
            # Rotation around the Y axis
            Ry = np.array([
                [np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)]
            ])
            # Rotation around the Z axis
            Rz = np.array([
                [np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz), np.cos(rz), 0],
                [0, 0, 1]
            ])
            # Combined rotation matrix: Rz * Ry * Rx
            return Rz @ Ry @ Rx

        # Create the 3D rotation matrix
        rotation = euler_to_rotation_matrix(robot_rx, robot_ry, robot_rz)

        # Create the 3D translation matrix
        translation = np.array([
            [1, 0, 0, robot_tx],
            [0, 1, 0, robot_ty],
            [0, 0, 1, robot_tz],
            [0, 0, 0, 1]
        ])

        # Create the 3D rotation matrix in homogeneous coordinates
        rotation_homogeneous = np.eye(4)
        rotation_homogeneous[:3, :3] = rotation

        # Combined transformation matrix (translation * rotation)
        transformation_matrix = translation @ rotation_homogeneous
        return transformation_matrix

    def generate_room_polygon(self, room_node):
        room_width = room_node.attrs["width"].value
        room_depth = room_node.attrs["depth"].value

        room_width_half = room_width / 2
        room_depth_half = room_depth / 2

        # Construct a polygon representing the room
        room_polygon = Polygon([
            (room_width_half, room_depth_half),
            (room_width_half, -room_depth_half),
            (-room_width_half, -room_depth_half),
            (-room_width_half, room_depth_half)
        ])
        return room_width, room_depth, room_polygon

    def transform_to_room_frame(self, points: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
        """
        Transform LiDAR points using the robot's transformation matrix.
        :param lidar_points: Nx3 array of LiDAR points.
        :param transformation_matrix: 4x4 transformation matrix.
        :return: Transformed LiDAR points.
        """
        # Convert LiDAR points to homogeneous coordinates
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))

        # Apply transformation
        transformed_points = (transformation_matrix @ homogeneous_points.T).T

        # Return only the x, y, z coordinates (discard the homogeneous coordinate)
        return transformed_points[:, :3]

    def project_points_to_polygon(self, points: np.ndarray, polygon: Polygon, distance_threshold: float) -> np.ndarray:
        """
        Projects LiDAR points onto a polygon and removes points that are close to the polygon.

        :param points: Nx2 array of LiDAR points (each row is a point [x, y]).
        :param polygon: Shapely Polygon representing the room.
        :param distance_threshold: Points closer than this distance to the polygon will be removed.
        :return: Filtered Nx2 array of LiDAR points.
        """

        return np.array([point for point in points if polygon.exterior.distance(Point(point)) > distance_threshold])

    def get_clusters(self, points: np.ndarray, distance_threshold: float, min_points: int) -> np.ndarray:
        # Step 3: Create an Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Step 4: Cluster the points using DBSCAN
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
            labels = np.array(pcd.cluster_dbscan(eps=distance_threshold, min_points=min_points))

        # Return np.array in which each position is a np.array of the points of each cluster
        return np.array([points[labels == i] for i in range(max(labels) + 1)], dtype=object)

    def initialize_fridge(self, room_node, points):
        init_params = [-0.3, 0.4, 0.9, 0.0, 0.0, 0.1, 0.5, 0.2, 1.8]  # Initial guess
        f = FridgeModel(init_params)
        f.print_params()

        optimizer = optim.Adam([
            {'params': [f.x, f.y, f.z], 'lr': 0.1, 'momentum': 0.6},  # Position
            {'params': [f.a, f.b, f.c], 'lr': 0.01, 'momentum': 0.6},  # Rotation (higher LR)
            {'params': [f.w, f.d, f.h], 'lr': 0.1, 'momentum': 0.6}  # Size
        ])
        num_iterations = 1000
        # convert points to tensor
        loss_ant = float('inf')
        now = time.time()
        print("Start optimization...")
        for i in range(num_iterations):
            optimizer.zero_grad()
            loss = f.loss_function(points)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss.item():6f}")  # Check gradients
                if abs(loss.item() - loss_ant) < 0.00001:  # Convergence criterion
                    break
                loss_ant = loss.item()

        print("Optimization complete.")
        f.print_params()
        f.print_grads()
        print(f"Elapsed time: {time.time() - now:.4f} seconds")

        # I need now to draw the resulting fridge in the room
        self.draw_fridge(f)

        # hess, cov = f.hessian_and_covariance(points)
        # np.set_printoptions(precision=2, suppress=True)
        # print("Covariance Matrix:\n", cov.cpu().numpy())
        # std_devs = torch.sqrt(torch.diag(cov))
        # print("Standard Deviations (Uncertainty) per Parameter:", std_devs.cpu().numpy())
        # eigvals, eigvecs = torch.linalg.eigh(hess)
        # print("Eigenvalues of Hessian:", eigvals.cpu().numpy())
        # unexplained = f.remove_explained_points(points, 0.05)
        # print("Unexplained points:", unexplained.shape[0], "out of", points.shape[0], "percentage:", unexplained.shape[0] / points.shape[0] * 100, "%")
        return f
    
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

    def draw_room(self, room_node):

        # only draw the first time
        if self.room_drawn:
            return

        room_width = room_node.attrs["width"].value / 1000
        room_depth = room_node.attrs["depth"].value / 1000
        room_height = 1

        # Create the four walls
        wall_thickness = 0.1

        # Front wall
        front_wall = o3d.geometry.TriangleMesh.create_box(width=room_width, height=wall_thickness, depth=room_height)
        front_wall.translate([-room_width / 2, -room_depth / 2, 0])
        front_wall.paint_uniform_color([1.0, 0.6, 0.2])  # Set color to light orange
        self.vis.add_geometry(front_wall)

        # Back wall
        back_wall = o3d.geometry.TriangleMesh.create_box(width=room_width, height=wall_thickness, depth=room_height)
        back_wall.translate([-room_width / 2, room_depth / 2 - wall_thickness, 0])
        back_wall.paint_uniform_color([1.0, 0.6, 0.2])  # Set color to light orange
        self.vis.add_geometry(back_wall)

        # Left wall
        left_wall = o3d.geometry.TriangleMesh.create_box(width=wall_thickness, height=room_depth, depth=room_height)
        left_wall.translate([-room_width / 2, -room_depth / 2, 0])
        left_wall.paint_uniform_color([1.0, 0.6, 0.2])  # Set color to light orange
        self.vis.add_geometry(left_wall)

        # Right wall
        right_wall = o3d.geometry.TriangleMesh.create_box(width=wall_thickness, height=room_depth, depth=room_height)
        right_wall.translate([room_width / 2 - wall_thickness, -room_depth / 2, 0])
        right_wall.paint_uniform_color([1.0, 0.6, 0.2])  # Set color to light orange
        self.vis.add_geometry(right_wall)

        self.room_drawn = True

    def draw_fridge(self, fridge):
        """
        Convert the PyTorch3D mesh from the fridge model to an Open3D mesh and visualize it.
        """
        # Get the PyTorch3D mesh
        mesh = fridge.forward()
        verts = mesh.verts_list()[0].detach().cpu().numpy()
        faces = mesh.faces_list()[0].detach().cpu().numpy()

        # Create Open3D mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

        # Compute normals and apply color
        o3d_mesh.compute_vertex_normals()
        o3d_mesh.paint_uniform_color([0.0, 0.5, 1.0])  # Light blue fridge

        # Add to visualizer
        self.vis.add_geometry(o3d_mesh)
        self.vis.poll_events()
        self.vis.update_renderer()

    #######################################

    def startup_check(self):
        print(f"Testing RoboCompLidar3D.TPoint from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TPoint()
        print(f"Testing RoboCompLidar3D.TDataImage from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TDataImage()
        print(f"Testing RoboCompLidar3D.TData from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TData()
        QTimer.singleShot(200, QApplication.instance().quit)

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



    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        console.print(f"UPDATE NODE ATT: {id} {attribute_names}", style='green')

    def update_node(self, id: int, type: str):
        console.print(f"UPDATE NODE: {id} {type}", style='green')

    def delete_node(self, id: int):
        console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):

        console.print(f"UPDATE EDGE: {fr} to {type}", type, style='green')

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')

    def delete_edge(self, fr: int, to: int, type: str):
        console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
