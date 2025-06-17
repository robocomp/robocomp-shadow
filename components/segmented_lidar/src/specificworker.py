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
from src.segmentator import Segmentator
import cv2
import numpy as np
import cupy as cp

import queue
import traceback
from collections import deque
import time
import sys
from threading import Thread, Event

from mmseg.apis import inference_model, init_model, show_result_pyplot, MMSegInferencer
import shutil
import csv

sys.path.append('/opt/robocomp/lib')

console = Console(highlight=False)


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, configData, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map, configData)
        self.Period = configData["Period"]["Compute"]
        if startup_check:
            self.startup_check()
        else:
            #self.model_performance_testing()

            self.read_queue = deque(maxlen=1)
            self.odometry_queue = deque(maxlen=15)
            self.pointcloud_queue = deque(maxlen=5)

            # ============== SEGMENTATION ==================
            self.segmentator = Segmentator()

            self.timestamp = 0

            self.event = Event()
            self.thread_period = 1000
            
            self.integrate_odometry = False

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        return True


    @QtCore.Slot()
    def compute(self):

        try:
            image = self.camera360rgbd_proxy.getROI(-1,-1,-1,600,-1,600)
            if image.width / image.height > 4:
                print("Wrong image aspect ratio")
                # event.wait(self.thread_period / 1000)
                return
            if image.alivetime == self.timestamp:
                print("No new image")
                return
            rgb = np.frombuffer(image.rgb, dtype=np.uint8).reshape((image.height, image.width, 3))

            depth = np.frombuffer(image.depth, dtype=np.float32).reshape((image.height, image.width, 3))

            self.timestamp = image.alivetime
            points, mask = self.segmentator.process_frame(rgb, depth, self.timestamp)
            if self.integrate_odometry:
                # # From self.odometry_queue, create a copy and search odometry values between now and image timestamp. Iterate in a reverse way until finding a timestamp lower than the image timestamp
                odometry = cp.array(self.odometry_queue.copy())
                odometry_interval = odometry[odometry[:, 3] > self.timestamp]
                points = self.integrate_odometry_to_pointcloud(odometry_interval, points)

            lidar_data = self.to_lidar_data(points, self.timestamp)
            self.lidar3dpub_proxy.pushLidarData(lidar_data)
            #image = self.segmentator.mask_to_color(mask.get())
            #cv2.imshow("Segmentation", image)
            #cv2.waitKey(1)
            
        except Ice.Exception as e:
            print(e, "Error communicating with CameraRGBDSimple")


    def get_rgbd(self, camera_name: str, event: Event):
        while not event.is_set():
            try:
                start = time.time()
                print("try timestamp",start)
                image = self.camera360rgbd_proxy.getROI(-1,-1,-1,600,-1,600)
                if image.width / image.height > 4:
                    print("Wrong image aspect ratio")
                    # event.wait(self.thread_period / 1000)
                    expended = time.time() - start
                    print("expended", expended, "wait", (self.thread_period / 1000) - expended )
                    event.wait((self.thread_period / 1000) - expended)
                    continue
                if image.alivetime == self.last_rgbd_timestamp_thread:
                    print("No new image")
                    # event.wait(self.thread_period / 1000)
                    expended = time.time() - start
                    print("expended", expended, "wait", (self.thread_period / 1000) - expended )
                    event.wait((self.thread_period / 1000) - expended)
                    continue
                rgb_frame = np.frombuffer(image.rgb, dtype=np.uint8).reshape((image.height, image.width, 3))

                depth_frame = cp.frombuffer(image.depth, dtype=cp.float32).reshape((image.height, image.width, 3))
                self.read_queue.append([rgb_frame, depth_frame, image.alivetime])
                self.last_rgbd_timestamp_thread = image.alivetime
                expended = time.time() - start
                print("expended", expended, "wait", (self.thread_period / 1000) - expended )
                event.wait((self.thread_period / 1000) - expended)
            except Ice.Exception as e:
                print(e, "Error communicating with CameraRGBDSimple")
                expended = time.time() - start
                print("expended", expended, "wait", (self.thread_period / 1000) - expended )
                event.wait((self.thread_period / 1000) - expended)

    def integrate_odometry_to_pointcloud(self, odometry_list, pointcloud):

        """Accumulation with minimal memory usage (good for very large datasets)"""
        if not odometry_list.any():
            print("No odometry values")
            return cp.eye(3)

        prev_time = odometry_list[0][3]
        accum = cp.eye(4)

        for i in range(1, len(odometry_list)):
            curr_time = odometry_list[i][3]
            dt = curr_time - prev_time
            curr_speed = odometry_list[i][:3] * dt * 0.001

            # Construct the 3x3 transformation matrix for this step
            T = cp.array([
                [cp.cos(curr_speed[2]).get(), -cp.sin(curr_speed[2]).get(), 0, curr_speed[0].get()],
                [cp.sin(curr_speed[2]).get(), cp.cos(curr_speed[2]).get(), 0, curr_speed[1].get()],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            # Accumulate the transformation
            accum = cp.dot(accum, T)

            prev_time = curr_time


        points = pointcloud[:, :3]
        # Transform every three first elements of the pointcloud and buil homogeneus matrix
        homogeneous_points = cp.hstack([points, cp.ones((points.shape[0], 1))])

        point_cloud = cp.dot(accum, homogeneous_points.T).T

        # Normalize homogenous component to get pointcloud 3xN
        pointcloud[:, :3] = point_cloud[:, :3]


        # Stack a 1 as fourth element
        return pointcloud # Return as numpy array if needed


    def to_lidar_data(self, points, timestamp):
        x_array, y_array, z_array, seg_array = points.T
        seg_array = seg_array.astype(np.int32)
        lidar_data = ifaces.RoboCompLidar3D.TDataCategory(XArray=x_array, YArray=y_array, ZArray=z_array, CategoriesArray=seg_array, period=100, timestamp=timestamp)
        return lidar_data

    def model_performance_testing(self):
        models = MMSegInferencer.list_models('mmseg')
        # Write to CSV
        with open('model_results.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Model Name", "Inference Time (s)"])  # Header

            for i in models:
                if "ade20k" in i:
                    print(i)
                    # if not os.path.isdir("images/"+i):
                    mean_time = 0
                    try:
                        inferencer = MMSegInferencer(model=i, device="cuda")
                        for j in range(5):
                            start = time.time()
                            image = self.camera360rgbd_proxy.getROI(-1, -1, -1, 600, -1, 600)
                            rgb = np.frombuffer(image.rgb, dtype=np.uint8).reshape((image.height, image.width, 3))
                            result = inferencer(rgb, return_datasamples=True)
                            print(result)
                            print(i, "- Expended time", time.time() - start)
                            if j > 0:
                                mean_time += (time.time() - start)
                        print(i, "mean time:", mean_time / 4)
                        writer.writerow([i, mean_time / 4])
                        try:
                            shutil.rmtree("/home/robolab/.cache/torch/hub/checkpoints")
                        except:
                            print("Folder deleted yet")
                    except Exception as e:
                        print(e)
                        continue
        exit(0)

    def startup_check(self):
        print(f"Testing RoboCompCamera360RGBD.TRoi from ifaces.RoboCompCamera360RGBD")
        test = ifaces.RoboCompCamera360RGBD.TRoi()
        print(f"Testing RoboCompCamera360RGBD.TRGBD from ifaces.RoboCompCamera360RGBD")
        test = ifaces.RoboCompCamera360RGBD.TRGBD()
        print(f"Testing RoboCompLidar3D.TPoint from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TPoint()
        print(f"Testing RoboCompLidar3D.TDataImage from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TDataImage()
        print(f"Testing RoboCompLidar3D.TData from ifaces.RoboCompLidar3D")
        test = ifaces.RoboCompLidar3D.TData()
        QTimer.singleShot(200, QApplication.instance().quit)

    # =============== Methods for Component SubscribesTo ================
    # ===================================================================

    #
    # SUBSCRIPTION to newFullPose method from FullPoseEstimationPub interface
    #
    def FullPoseEstimationPub_newFullPose(self, pose):
        self.odometry_queue.append(np.array([pose.vx * 1000, pose.vy * 1000, pose.vrz, pose.timestamp]))


    # ===================================================================sss
    # ===================================================================

    # =============== Methods for Component Implements ==================
    # ===================================================================

    #
    # IMPLEMENTATION of getLidarData method from Lidar3D interface
    #
    def Lidar3D_getLidarData(self, name, start, len, decimationDegreeFactor):
        ret = ifaces.RoboCompLidar3D.TData()
        #
        # write your CODE here
        #
        return ret
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
    # IMPLEMENTATION of getLidarDataByCategory method from Lidar3D interface
    #
    def Lidar3D_getLidarDataByCategory(self, categories, timestamp):
        ret = ifaces.RoboCompLidar3D.TData()
        # copy last element from the queue
        if self.read_queue:
            ret = self.read_queue[-1]

        # filter elements by category (labels)
        if categories:
            points = []
            colors = []
            labels = []
            for i in range(len(ret.points)):
                if ret.points[i].labels in categories:
                    points.append([ret.points[i].x, ret.points[i].y, ret.points[i].z])
                    colors.append(ret.points[i].colors)
                    labels.append(ret.points[i].labels)
            ret.points = points
            ret.categories = colors
            ret.labels = labels

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

    ######################
    # From the RoboCompLidar3DPub you can publish calling this methods:
    # self.lidar3dpub_proxy.pushLidarData(...)

    ######################
    # From the RoboCompLidar3D you can use this types:
    # RoboCompLidar3D.TPoint
    # RoboCompLidar3D.TDataImage
    # RoboCompLidar3D.TData
    # RoboCompLidar3D.TDataCategory

    ######################
    # From the RoboCompCamera360RGBD you can call this methods:
    # self.camera360rgbd_proxy.getROI(...)

    ######################
    # From the RoboCompCamera360RGBD you can use this types:
    # RoboCompCamera360RGBD.TRoi
    # RoboCompCamera360RGBD.TRGBD

