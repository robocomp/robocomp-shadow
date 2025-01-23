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


sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 30
        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):

        self.adv = 0.0
        self.side = 0.0
        self.rot = 0.0
	
        self.adv_j = 0.0
        self.side_j = 0.0
        self.rot_j = 0.0

        return True


    @QtCore.Slot()
    def compute(self):

        # Check if joystick data is available
        if self.adv_j != 0.0 or self.side_j != 0.0 or self.rot_j != 0.0:
            self.adv = self.adv_j
            self.side = self.side_j
            self.rot = self.rot_j
            print("joystick")
            # Send command to robot
            try:
                self.omnirobot_proxy.setSpeedBase(self.side, self.adv, self.rot)
            except Ice.Exception as e:
                print("Error sending speed to robot", e)
            return

        t1 = time.time()

        try:
            # Calculate de line given by the robot speed vector (adv, rot)
            # Robot position
            robot_position = np.array([0, 0])

            # Speed vector (adv, rot)
            dt = 1
            dy = (self.adv / 1000 * np.cos(self.rot) - self.side / 1000 * np.sin(self.rot)) * dt
            dx = (self.adv / 1000 * np.sin(self.rot) + self.side / 1000 * np.cos(self.rot)) * dt
            speed = np.array([dx, dy])

            # build vector line
            d = 3
            resolution = 0.1
            N_points = int(d / resolution)

            self.VOXEL_SIZE = 0.1

            # Get Lidar data
            ldata_ = self.lidar3d_proxy.getLidarDataWithThreshold2d("bpearl", 3000, 2)
            # for ldata in ldata_: build a point cloud
            pcd = o3d.geometry.PointCloud()
            for ldata in ldata_.points:
                pcd.points.append([ldata.x / 1000, ldata.y / 1000, ldata.z / 1000])

            # Voxelize the point cloud
            pcd = pcd.voxel_down_sample(voxel_size=self.VOXEL_SIZE)

            # DBSCAN clustering of point cloud
            pcd_np = np.asarray(pcd.points)
            clustering = DBSCAN(eps=0.7, min_samples=20).fit(pcd_np)
            labels = clustering.labels_

            clusters = []

            for i in range(max(labels)+1):
                cluster = pcd_np[labels == i]
                centroid = np.mean(cluster, axis=0)
                # compute radius
                radius = np.max(np.linalg.norm(cluster - centroid, axis=1))

                # If speed != 0
                if np.linalg.norm(speed) > 0:
                    v_dir = speed / np.linalg.norm(speed) * resolution
                    speed_line = np.array([robot_position, robot_position + v_dir * N_points])

                    # TODO: One call to point_to_line
                    # Compute closest point from cluster to speed line
                    closest_point = cluster[np.argmin([self.point_to_line_distance(point, speed_line[1], speed_line[0])[0] for point in cluster])]
                    dist, nearest = self.point_to_line_distance(closest_point, speed_line[1], speed_line[0])

                    #Sigmoide parameters
                    lateral_th = 0.7
                    advance_th = 0.5

                    #Security distances TODO: Tune this values
                    side_th = 1 + 1 * abs(self.rot / 0.78)
                    front_th = 1.5 + abs(self.adv / 750)


                    if dist < side_th and np.linalg.norm(nearest[:2]) < front_th:
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

                        clusters.append((centroid, radius, closest_point[:2], nearest[:2], repulsion))


            speed_r = [0.0,0.0]
            if clusters:
                # Compute sum of repulsion vectors
                repulsion = np.sum([cluster[4] for cluster in clusters], axis=0)
                speed_r += repulsion

            print("Time to compute", time.time() - t1)
            # Send command to robot
            # self.omnirobot_proxy.setSpeedBase(self.adv, self.side, self.rot)
            # Get adv, side and rot from speed_r

            self.adv = self.adv - speed[1]
            self.rot = self.rot
            self.side = speed_r[0] * 300

            print("Speed", self.adv, self.side, self.rot)
            try:
                self.omnirobot_proxy.setSpeedBase(self.side, self.adv, self.rot)
            except Ice.Exception as e:
                print("Error sending speed to robot", e)



        # ------------------- Plotting -------------------
            if True: # TODO: Plotting variable and move plot to window
                # Color the point cloud based on cluster labels
                colors = plt.get_cmap("tab20")(labels / (max(labels) if max(labels) > 0 else 1))
                pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

                #   draw in with matplotlib in a 2D plot window
                plt.figure()
                # fix axis to +- 5m
                plt.xlim(-2.5, 2.5)
                plt.ylim(-2, 2)

                plt.scatter(pcd_np[:, 0], pcd_np[:, 1], c=labels, cmap='tab20')
                # draw robot as red box with 0,5m side centered in the robot position
                plt.gca().add_artist(plt.Rectangle((-0.25, -0.25), 0.5, 0.5, fill=False, color='red'))

                if clusters:
                #     Compute sum of repulsion vectors
                    plt.arrow(0, 0, repulsion[0], repulsion[1], head_width=0.05, head_length=0.1, fc='r', ec='r')
                    # Draw speed line
                    plt.plot(speed_line[:, 0], speed_line[:, 1], color='blue')
                    # draw clusters as circles with radius equal to the radius of the cluster
                    for cluster in clusters:
                        plt.gca().add_artist(plt.Circle(cluster[0][:2], cluster[1], fill=False, color='black'))
                #         Draw closest point as a red dot
                        plt.scatter(cluster[2][0], cluster[2][1], color='red')
                #         Draw nearest point as a blue dot
                        plt.scatter(cluster[3][0], cluster[3][1], color='blue')

                #     draw speed_r as a red arrow
                    plt.arrow(0, 0, speed_r[0], speed_r[1], head_width=0.2, head_length=0.2, fc='r', ec='r')
                # draw robot speed using speed array
                plt.arrow(0, 0, speed[0], speed[1], head_width=0.1, head_length=0.1, fc='orange', ec='g')
            #     Draw plot in a window and close after 0.1s
                plt.show()

        except Ice.Exception as e:
          print("Error reading Lidar",e)

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
        # Set speed from joystick data to omnirobot

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


