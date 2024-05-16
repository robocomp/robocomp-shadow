#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2022 by YOUR NAME HERE
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

from PySide2.QtCore import Qt
# from PySide2.QtCore import QTimer
# from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import numpy as np
import time
import cv2
from threading import Thread, Event
import traceback
import queue
from collections import deque
import sys
import yaml
import itertools
import glob
# from PIL import Image
import copy
# import math

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/ByteTrack')
from byte_tracker import BYTETracker
import matching
from ultralytics import YOLO
import torch
import itertools
import math

from pydsr import *

sys.path.append('/home/robocomp/software/JointBDOE')

from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

console = Console(highlight=False)

_OBJECT_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                 'sheep',
                 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                 'scissors',
                 'teddy bear', 'hair drier', 'toothbrush']

_DETECTED_OBJECT_NAMES = ['person', 'sports ball', 'tv', 'couch']
# Get _OBJECT_NAMES indexes for the objects to detect
_DETECTED_OBJECTS = [i for i, obj in enumerate(_OBJECT_NAMES) if obj in _DETECTED_OBJECT_NAMES]
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, params, startup_check=False):
        """
        Initializes an instance of `SpecificWorker`, setting up various components
        such as the graph, camera, tracker, models, and event handling mechanisms.

        Args:
            proxy_map (`dsrgraph.Graph` object.): 2D environment map that this
                specific worker is associated with, which the function uses to
                access related objects and functions in the worker's scope.
                
                		- `super(SpecificWorker, self).__init__(proxy_map)`: This line
                initializes the base class `SpecificWorker` using the `proxy_map`
                argument.
                		- `self.agent_id = 274`: The `agent_id` property is set to a
                fixed value of 274, which is used for identification purposes in
                the deployment.
                		- `g = DSRGraph(0, "environment_object_perception_agent",
                self.agent_id)`: The `g` variable is assigned a `DSRGraph` object
                with a specific identifier (`0`), name ("environment_object_perception_agent"),
                and agent ID (`self.agent_id`).
                		- `signals.connect(self.g, signals.UPDATE_NODE_ATTR,
                self.update_node_att)`: This line connects the `update_node_attr`
                signal of the `g` graph to the method `update_node_att` in the
                class `SpecificWorker`.
                		- `signals.connect(self.g, signals.UPDATE_NODE_ATTR,
                self.update_node_att)`: This line connects the `update_node_attr`
                signal of the `g` graph to the method `update_node_att` in the
                class `SpecificWorker`.
                		- `ifaces.RoboCompVisualElementsPub.TData()`: The
                `ifaces.RoboCompVisualElementsPub.TData()` line initializes an
                instance of the `TData` class from the `ifaces` module, which is
                used for handling data from the Robocomp visual elements publisher.
                		- `lidar_odometry_data = self.lidarodometry_proxy.getPoseAndChange()`:
                This line retrieves the latest lidar odometry data from the
                `lidarodometry_proxy` instance using the `getPoseAndChange()` method.
                		- `self.rgb_read_queue = deque(maxlen=1)`: The `self.rgb_read_queue`
                property is set to an empty queue with a maximum length of 1, which
                is used for storing the read RGB images from the camera.
                		- `self.image_read_thread = Thread(target=self.get_rgb_thread,
                args=["camera_top", self.event], name="rgb_read_queue", daemon=True)`:
                This line creates a new thread for reading RGB images from the
                camera using the `get_rgb_thread()` method and storing them in the
                `rgb_read_queue` property.
                		- `self.inference_execution_thread = Thread(target=self.inference_thread,
                args=[self.event], name="inference_read_queue", daemon=True)`:
                This line creates a new thread for executing the YOLO and JointBDOE
                inference models using the `inference_thread()` method and storing
                the results in the `event` property.
                		- `self.timer.timeout.connect(self.compute)`: This line connects
                the `timeout` signal of the `timer` instance to the `compute()`
                method in the class `SpecificWorker`, which is called every `
                Period` milliseconds to perform the inference task.
            params (unknown value because there isn't enough context to determine
                its data type based on the provided code snippet.): 3rd argument
                passed to the constructor of the SpecificWorker class, which is
                used to specify additional configuration parameters for the worker's
                operation, such as startup checks and time intervals.
                
                	1/ `startup_check`: This is a boolean value that indicates whether
                to call the `startup_check()` method or not. It is optional and
                can be skipped if not provided.
                	2/ `proxy_map`: This is a dictionary containing the proxy maps
                for different environments. The function initializes the agent's
                proxy map using this parameter.
                	3/ `agent_id`: This is an integer value that represents the unique
                identifier of the agent in the deployment. It is set to
                `_CHANGE_THIS_ID_` by default, and can be modified if required.
                	4/ `g`: This is a DSRGraph object that represents the environment
                graph. The function initializes this object with the specified ID
                and connects signals to it for updates, deletes, and node attribute
                changes.
                	5/ `event`: This is an event object that is used to communicate
                between threads. It is created and passed as an argument to the
                thread that reads the RGB image.
                	6/ `rgb_original`: This is a boolean value that indicates whether
                the original RGB image should be read or not. By default, it is
                set to `False`, and can be modified if required.
                	7/ `refresh_sleep_time`: This is an integer value that represents
                the time interval between frame refreshments in milliseconds. It
                is set to 5 by default, and can be modified if required.
                	8/ `rgb_read_queue`: This is a dequeued list of RGB images that
                are read from the camera and passed to the inference thread for processing.
                	9/ `inference_read_queue`: This is a dequeued list of inference
                outputs that are produced by the inference thread and passed to
                the event loop for processing.
                
                	In summary, `params` is a dictionary containing properties for
                initializing the agent's proxy map, reading RGB images from the
                camera, setting up the environment graph, and configuring the timer
                interval for frame refreshments.
            startup_check (`Event`.): function that performs a check for any issues
                or errors during the startup process, and is executed if `True`,
                otherwise it is skipped.
                
                		- `startup_check`: This is an optional parameter that is used
                to run a check on the agent's startup. If it is set to `True`, the
                `startup_check()` method will be called, otherwise it will not be
                called. The method checks if the agent has been started correctly
                and performs any necessary initialization.
                
                	The properties of the `startup_check` parameter are:
                
                		- `startup_check`: A boolean value that determines whether to
                run the `startup_check()` method or not.
                		- `False`: The default value for `startup_check`. This means
                that the `startup_check()` method will not be called on startup.
                		- `True`: If set to `True`, the `startup_check()` method will
                be called on startup to check if the agent has been started correctly.
                
                	The `startup_check` parameter has no attributes or properties
                other than its default value and the ability to accept a boolean
                value.

        """
        super(SpecificWorker, self).__init__(proxy_map)

        # YOU MUST SET AN UNIQUE ID FOR THIS AGENT IN YOUR DEPLOYMENT. "_CHANGE_THIS_ID_" for a valid unique integer
        self.agent_id = 274
        self.g = DSRGraph(0, "environment_object_perception_agent", self.agent_id)

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

        if startup_check:
            self.startup_check()
        else:
            self.Period = 100
            self.thread_period = 100
            self.display = False

            # Hz
            self.cont = 0
            self.last_time = time.time()
            self.fps = 0
            
            # read test image to get sizes
            started_camera = False

            while not started_camera:
                try:
                    self.rgb_original = self.camera360rgbd_proxy.getROI(-1, -1, -1, -1, -1, -1)
            
                    print("Camera specs:")
                    print(" width:", self.rgb_original.width)
                    print(" height:", self.rgb_original.height)
                    print(" depth", self.rgb_original.rgbchannels)
                    print(" focalx", self.rgb_original.focalx)
                    print(" focaly", self.rgb_original.focaly)
                    print(" period", self.rgb_original.period)
                    print(" alive time", self.rgb_original.alivetime)
                    print(" ratio {:.2f}".format(self.rgb_original.width/self.rgb_original.height))
            
                    # Image ROI require parameters
                    self.roi_xsize = self.rgb_original.width // 2
                    self.roi_ysize = self.rgb_original.height
                    self.final_xsize = self.rgb_original.width // 2
                    self.final_ysize = self.rgb_original.height // 2
                    self.roi_xcenter = self.rgb_original.width // 2
                    self.roi_ycenter = self.rgb_original.height // 2
            
                    # Target ROI size
                    self.target_roi_xsize = self.roi_xsize
                    self.target_roi_ysize = self.roi_ysize
                    self.target_roi_xcenter = self.roi_xcenter
                    self.target_roi_ycenter = self.roi_ycenter
            
                    started_camera = True
                except Ice.Exception as e:
                    traceback.print_exc()
                    print(e, "Trying again CAMERA...")
                    time.sleep(2)
            
            ############## OBJECTS ##############
            #TRACKER
            self.tracker = BYTETracker(frame_rate=5, buffer_=1000)
            # self.tracker_back = BYTETracker(frame_rate=5)
            self.objects = ifaces.RoboCompVisualElementsPub.TData()
            # self.ui.pushButton.clicked.connect(self.reset_tracks)
            self.reset = False
            self.refresh_sleep_time = 5

            ############## MODELS ##############
            # Load YOLO model
            self.load_v8_model()
            # Load JointBDOE model
            self.load_jointbdoe_model()

            ############## FOVEA ##############
            self.tracked_id = -1

            lidar_odometry_data = self.lidarodometry_proxy.getPoseAndChange()
            # Generate numpy transform matrix with the lidar odometry data
            self.last_transform_matrix = np.array(
                [[lidar_odometry_data.pose.m00, lidar_odometry_data.pose.m01, lidar_odometry_data.pose.m02, lidar_odometry_data.pose.m03],
                 [lidar_odometry_data.pose.m10, lidar_odometry_data.pose.m11, lidar_odometry_data.pose.m12, lidar_odometry_data.pose.m13],
                 [lidar_odometry_data.pose.m20, lidar_odometry_data.pose.m21, lidar_odometry_data.pose.m22, lidar_odometry_data.pose.m23],
                 [lidar_odometry_data.pose.m30, lidar_odometry_data.pose.m31, lidar_odometry_data.pose.m32, lidar_odometry_data.pose.m33]])

            self.rgb_read_queue = deque(maxlen=1)
            self.inference_read_queue = deque(maxlen=1)

            self.event = Event()
            
            self.image_read_thread = Thread(target=self.get_rgb_thread, args=["camera_top", self.event],
                                      name="rgb_read_queue", daemon=True)
            self.image_read_thread.start()
                        
            self.inference_execution_thread = Thread(target=self.inference_thread, args=[self.event],
                                      name="inference_read_queue", daemon=True)
            self.inference_execution_thread.start()
            
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        """
        Reads configuration parameters from a file and sets instance variables
        based on the read parameters, then returns `True`.

        Args:
            params (dict): configuration parameters for the YOLO model, including
                the YOLO model file, display flag, depth flag, and class names,
                which are read from a file using the `open()` function.

        Returns:
            bool: a Boolean value indicating whether the parameters were successfully
            read and the process can proceed.

        """
        try:
            self.classes = [0]
            # self.yolo_model = params["yolo_model"]
            self.display = params["display"] == "true" or params["display"] == "True"
            self.depth_flag = params["depth"] == "true" or params["depth"] == "True"
            with open(params["classes-path-file"]) as f:
                [self.classes.append(_OBJECT_NAMES.index(line.strip())) for line in f.readlines()]
            print("Params read. Starting...", params)
        except:
            print("Error reading config params")
            traceback.print_exc()

        return True

    @QtCore.Slot()
    def compute(self):
        """
        This method carries out the main object detection pipeline.

        It retrieves the image and, if applicable, depth information from the read_queue.
        It then sets the region of interest (ROI) dimensions if there is a tracked object.
        Then, it makes predictions using YOLOv8 and stores this information in an interface format.
        If the display option is enabled, it will display the tracking results on the image.
        Finally, it shows the frames per second (FPS) of the pipeline.
        """
        if self.reset:
            print("RESETTING TRACKS") 
            self.tracker.tracked_stracks = []  # type: list[STrack]
            self.tracker.lost_stracks = []  # type: list[STrack]
            self.tracker.removed_stracks = []  # type: list[STrack]
            self.tracker.frame_id = 0
            time.sleep(2)
            stop_node = Node(agent_id=self.agent_id, type='intention', name="STOP")
            try:
                id_result = self.g.insert_node(stop_node)
                console.print('Person mind node created -- ', id_result, style='red')
                has_edge = Edge(id_result, self.g.get_node('Shadow').id, 'has', self.agent_id)
                self.g.insert_or_assign_edge(has_edge)

                print(' inserted new node  ', id_result)

            except:
                traceback.print_exc()
                print('cant update node or add edge RT')
            time.sleep(2)
            self.reset = False

        if self.inference_read_queue:
            start = time.time()
            out_v8_front, color_front, orientation_bboxes_front, orientations_front, out_v8_back, color_back, orientation_bboxes_back, orientations_back, depth_front, depth_back, delta, alive_time, period, front_roi, back_roi = self.inference_read_queue.pop()
            people_front, objects_front = self.get_segmentator_data(out_v8_front, color_front, depth_front)
            people_back, objects_back = self.get_segmentator_data(out_v8_back, color_back, depth_back)

            people_front = self.associate_orientation_with_segmentation(people_front, orientation_bboxes_front, orientations_front)
            people_back = self.associate_orientation_with_segmentation(people_back, orientation_bboxes_back, orientations_back)


            # Compare tracks_front and tracks_back to remove tracks with same ID
            # for i, front_data in enumerate(tracks_front):
            #     for track_back in tracks_back:
            #         # Calculate pose difference
            #         pose_distance = np.linalg.norm(track_front.pose - track_back.pose)
            #         if track_front.track_id == track_back.track_id:
            #             if track_front.score > track_back.score:
            #                 tracks_back.remove(track_back)
            #             else:
            #                 tracks_front.remove(track_front)
            #             break

            # Fuse people_front and people_back and equal it to self.people_write

            for key in objects_front:
                objects_front[key] += people_front[key]
                objects_back[key] += people_back[key]
            # Check if the object is in the same position in both images
            for i in range(len(objects_front["poses"])):
                for j in range(len(objects_back["poses"])):
                    if objects_front["classes"][i] == objects_back["classes"][j]:
                        # Calculate pose difference
                        pose_distance = math.sqrt((objects_front["poses"][i][0] - objects_back["poses"][j][0]) ** 2 + (objects_front["poses"][i][1] - objects_back["poses"][j][1]) ** 2)
                        if pose_distance < 1200:
                            if objects_front["confidences"][i] > objects_back["confidences"][j]:
                                objects_back["confidences"][j] = 0
                            else:
                                objects_front["confidences"][i] = 0

            elements = {}
            for key in objects_front:
                elements[key] = objects_front[key] + objects_back[key]

            lidar_odometry_data = self.lidarodometry_proxy.getPoseAndChange()
            # Generate numpy transform matrix with the lidar odometry data
            matrix_data = np.array(
                [[lidar_odometry_data.pose.m00, lidar_odometry_data.pose.m01, lidar_odometry_data.pose.m02, lidar_odometry_data.pose.m03],
                 [lidar_odometry_data.pose.m10, lidar_odometry_data.pose.m11, lidar_odometry_data.pose.m12, lidar_odometry_data.pose.m13],
                 [lidar_odometry_data.pose.m20, lidar_odometry_data.pose.m21, lidar_odometry_data.pose.m22, lidar_odometry_data.pose.m23],
                 [lidar_odometry_data.pose.m30, lidar_odometry_data.pose.m31, lidar_odometry_data.pose.m32, lidar_odometry_data.pose.m33]])

            # Get the difference between the last odometry matrix and the current one
            odometry_difference = self.calcular_matriz_diferencial(matrix_data, self.last_transform_matrix)
            # print("Odometry difference", odometry_difference)
            self.last_transform_matrix = np.copy(matrix_data)
            tracks = self.tracker.update(np.array(elements["confidences"]),
                                np.array(elements["bboxes"]),
                                np.array(elements["classes"]),
                                np.array(elements["poses"]),
                                np.array(elements["orientations"]), odometry_difference)

            # print("Tracks", tracks)

            front_objects = self.to_visualelements_interface(tracks, alive_time, front_roi)

            # Fuse front_objects and back_objects and equal it to self.objects_write
            # self.visualelementspub_proxy.setVisualObjects(front_objects)
            # If display is enabled, show the tracking results on the image

            if self.display:
                img_front = self.display_data_tracks(color_front, front_objects.objects)
                # img_int = img.astype('float32') / 255.0
                # image_comp = cv2.addWeighted(img_int, 0.5, depth, 0.5, 0)
                cv2.imshow("back", img_back)
                cv2.imshow("front", img_front)
                cv2.waitKey(1)

            # Show FPS and handle Keyboard Interruption
            try:
                self.show_fps(alive_time, period)
            except KeyboardInterrupt:
                self.event.set()

    def reset_tracks(self):
        self.reset = True

    def calcular_matriz_diferencial(self, transformacion1, transformacion2):
        # Calcular la inversa de la primera matriz de transformación
        """
        Calculates the differential matrix by multiplying the inverse of the first
        transformation matrix by the second transformation matrix and scaling the
        result by 1000.

        Args:
            transformacion1 (`np.array`.): 4x4 transformation matrix that describes
                the rotation and translation of an object in 3D space before
                applying the transformation.
                
                		- `np.linalg.inv(transformacion1)` calculates the inverse of the
                matrix `transformacion1`, which is a square matrix of size (n, n).
                		- The matrix `transformacion1` represents a linear transformation
                between two vectors in an n-dimensional space.
                		- The elements of `transformacion1` are the coefficients of the
                linear transformation, arranged in a matrix with dimensions (n x
                n).
            transformacion2 (ndarray or NumPy array.): 2nd matrix of transformation
                that the function will take and multiply by its inverse to obtain
                the differential matrix.
                
                		- `np.linalg.inv(transformacion1)`: This is an instance of the
                NumPy `InvitationMatrx` class, which represents the inverse of the
                first matrix of transformation `transformacion1`. The inverse is
                computed using the `inv()` method provided by the ` InvitationMatrx`
                class.
                		- `np.dot(inversa_transformacion1, transformacion2)`: This is
                an instance of the NumPy `Vector` class, which represents the dot
                product of two vectors. In this case, it computes the dot product
                of the inverse of the first matrix of transformation
                `inversa_transformacion1` and the second matrix of transformation
                `transformacion2`.
                		- `matriz_diferencial[:3, 3] *= 1000`: This line of code multiplies
                a specific sub-matrix of the output matrix (`[:3, 3]`) of the dot
                product by a scalar value `1000`. This is done to artificially
                inflate the diagonal elements of the resulting matrix.

        Returns:
            float: a transformed matrix with values multiplied by 1000.

        """
        inversa_transformacion1 = np.linalg.inv(transformacion1)

        # Calcular la matriz diferencial multiplicando la inversa de la primera
        # matriz de transformación por la segunda matriz de transformación
        matriz_diferencial = np.dot(inversa_transformacion1, transformacion2)
        # Trasponer la matriz diferencial
        matriz_diferencial[:3, 3] *= 1000
        return matriz_diferencial

    ###################### MODELS LOADING METHODS ######################
    def load_v8_model(self):
        """
        Retrieves and loads a YOLOv8 model from disk or downloads it if not found,
        then stores it in the instance's attribute `v8_model`. If the model is not
        found, the function asks for selecting one of the available models and
        downloads it accordingly.

        """
        pattern = os.path.join(os.getcwd(), "yolov8?-seg.engine")
        matching_files = glob.glob(pattern)
        model_path = matching_files[0] if matching_files else None
        if model_path:
            print(f"{model_path} found")
            try:
                self.v8_model = YOLO(model_path)
            except:
                print("Error loading model. Downloading and converting model:")
                filename_with_extension = model_path.split('/')[-1]
                filename_without_extension = filename_with_extension.split('.engine')[0]
                self.download_and_convert_v8_model(filename_without_extension)
        # If model not found, ask for choosing between the available models (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        else:
            print("Model not found. Select a model: n, s, m, l, x")
            model_name = input()
            self.download_and_convert_v8_model("yolov8" + model_name + "-seg")
    def download_and_convert_v8_model(self, model_name):
        # Download model
        """
        Downloads a YOLO model from a given name and exports it to a TRT format
        on the default device, creating a new model instance.

        Args:
            model_name (str): name of the YOLO model that needs to be downloaded
                and converted to the TRT format.

        """
        self.v8_model = YOLO(model_name)
        # Export the model to TRT
        self.v8_model.export(format='engine', device='0')
        self.v8_model = YOLO(model_name + '.engine')
    def load_jointbdoe_model(self):
        """
        Loads a trained TensorFlow model for JointBDOE, selects the best device
        for inference based on the available hardware, and retrieves the necessary
        data for the model's usage.

        """
        try:
            self.device = select_device("0", batch_size=1)
            self.model = attempt_load(
                "/home/robocomp/software/JointBDOE/runs/JointBDOE/coco_s_1024_e500_t020_w005/weights/best.pt",
                map_location=self.device)
            self.stride = int(self.model.stride.max())
            with open("/home/robocomp/software/JointBDOE/data/JointBDOE_weaklabel_coco.yaml") as f:
                self.data = yaml.safe_load(f)  # load data dict
        except:
            print("Error loading JointBDOE model")
            exit(1)
    ######################################################################################################3

    def get_rgb_thread(self, camera_name: str, event: Event):
        """
            A method that continuously gets RGB data from a specific camera until an Event is set.

            Args:
                camera_name (str): The name of the camera to get RGB data from.
                event (Event): The Event that stops the method from running when set.

            """
        while not event.is_set():
            try:
                # if self.lidar_read_queue:

                    start = time.time()
                    # Get ROIs from the camera.

                    image_front = self.camera360rgbd_proxy.getROI(960, 480, 960, 960, 640, 640)
                    image_back = self.camera360rgbd_proxy.getROI(0, 480, 960, 960, 640, 640)

                    roi_data_front = ifaces.RoboCompCamera360RGB.TRoi(xcenter=image_front.roi.xcenter, ycenter=image_front.roi.ycenter, xsize=image_front.roi.xsize, ysize=image_front.roi.ysize, finalxsize=image_front.roi.finalxsize, finalysize=image_front.roi.finalysize)
                    roi_data_back = ifaces.RoboCompCamera360RGB.TRoi(xcenter=image_back.roi.xcenter, ycenter=image_back.roi.ycenter, xsize=image_back.roi.xsize, ysize=image_back.roi.ysize, finalxsize=image_back.roi.finalxsize, finalysize=image_back.roi.finalysize)

                    color_front = np.frombuffer(image_front.rgb, dtype=np.uint8).reshape(image_front.height, image_front.width, 3)
                    depth_front = np.frombuffer(image_front.depth, dtype=np.float32).reshape(image_front.height, image_front.width, 3)
                    color_back = np.frombuffer(image_back.rgb, dtype=np.uint8).reshape(image_back.height, image_back.width, 3)
                    depth_back = np.frombuffer(image_back.depth, dtype=np.float32).reshape(image_back.height, image_back.width, 3)

                    # Process image for orientation DNN
                    front_img_tensor = self.convert_image_to_tensor(color_front)
                    back_img_tensor = self.convert_image_to_tensor(color_back)

                    # Calculate time difference.
                    delta = int(1000 * time.time() - image_front.alivetime)
                    data_package = [color_front, front_img_tensor, color_back, back_img_tensor, depth_front, depth_back, delta, image_front.period, image_front.alivetime, roi_data_front, roi_data_back]

                    # image = self.camera360rgbd_proxy.getROI(-1, -1, -1, -1, -1, -1)
                    # color = np.frombuffer(image.rgb, dtype=np.uint8).reshape(image.height, image.width, 3)
                    # depth = np.frombuffer(image.depth, dtype=np.float32).reshape(image.height, image.width, 3)
                    # color_front, color_back, depth_front, depth_back = self.get_back_and_front(color, depth)

                    # # Calculate time difference.
                    # delta = int(1000 * time.time() - image.alivetime)

                    # data_package = [color_front, color_back, depth_front, depth_back, delta, image.period, image.alivetime]

                    self.rgb_read_queue.append(data_package)
                    # if (time.time() - start) > 0.1:
                    #     print("Time exceded get image")
                    event.wait(self.thread_period / 1000)


            except Ice.Exception as e:
                traceback.print_exc()
                print(e, "Error communicating with Camera360RGBD")
                return

    def convert_image_to_tensor(self, image):
        """
        Converts a given image into a tensor format, transposing its dimensions
        and normalizing its values to be between 0 and 1. It does so by using
        PyTorch's `torch.from_numpy()` method and applying transformations on the
        numpy array representing the image.

        Args:
            image (ndarray object or tensor.): 2D image that will be converted to
                a tensor.
                
                		- `image`: The input image, which is an instance of `np.ndarray`.
                		- `stride`: The stride of the image, which is a scalar value
                that determines how much each dimension of the image is skipped
                during processing.
                		- `auto`: A boolean value that indicates whether the image should
                be resized using the nearest neighbor or bicubic interpolation methods.

        Returns:
            tensor, which is a contiguous array of numerical values in Python: a
            tensor representation of the input image, ready to be used in a deep
            learning model.
            
            		- `img_ori`: The original image tensor after letterboxing, transposing,
            and normalization. Its shape is (1, height, width, 3), where height
            and width are the image's dimensions, and 3 represents the color
            channels (BGR).
            		- `torch.device`: The device on which the tensor is stored, which
            is specified by the function argument `self.device`. This information
            is not explicitly mentioned in the output but is implicitly provided
            as part of the returned tensor object.
            		- `np.ascontiguousarray()`: This method is used to convert the image
            tensor from a numpy array to a contiguous NumPy array, which allows
            for efficient processing with NumPy functions.
            		- `/ 255.0`: This line normalizes the pixel values to the range [0.0,
            1.0].

        """
        img_ori = letterbox(image, 640, stride=self.stride, auto=True)[0]
        img_ori = img_ori.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_ori = np.ascontiguousarray(img_ori)
        img_ori = torch.from_numpy(img_ori).to(self.device)
        img_ori = img_ori / 255.0  # 0 - 255 to 0.0 - 1.0

        if len(img_ori.shape) == 3:
            img_ori = img_ori[None]  # expand for batch dim
        return img_ori

    def inference_thread(self, event: Event):
        """
        Reads RGB frames from a queue, performs computer vision inferencing over
        each frame using a TensorFlow model, and appends the processed output to
        a new queue for further processing.

        Args:
            event (Event): event object that is used to wait for a specified time
                period before continuing with the function's execution.

        """
        while not event.is_set():
            if self.rgb_read_queue:
                start = time.time()
                color_front, front_img_tensor, color_back, back_img_tensor, depth_front, depth_back, delta, period, alive_time, front_roi, back_roi = self.rgb_read_queue.pop()
                out_v8_front, orientation_bboxes_front, orientations_front = self.inference_over_image(color_front, front_img_tensor)
                out_v8_back, orientation_bboxes_back, orientations_back = self.inference_over_image(color_back, back_img_tensor)
                self.inference_read_queue.append([out_v8_front, color_front, orientation_bboxes_front, orientations_front, out_v8_back, color_back, orientation_bboxes_back, orientations_back, depth_front, depth_back, delta, alive_time, period, front_roi, back_roi])
                # if (time.time() - start) > 0.1:
                #     print("Time exceded inference")
            # event.wait(10 / 1000)

    def get_mask_with_modified_background(self, mask, image):
        """
        A method that removes the background of person ROI considering the segmentation mask and fill it with white color

        Args:
            mask: black and white mask obtained with the segmentation
            image: RGB image corresponding to person ROI
        """
        masked_image = mask * image
        is_black_pixel = np.logical_and(masked_image[:, :, 0] == 0, masked_image[:, :, 1] == 0, masked_image[:, :, 2] == 0)
        masked_image[is_black_pixel] = [255, 255, 255]

        return masked_image

    def inference_over_image(self, img0, img_ori):
        # Make inference with both models
        """
        Performs image-based object detection using two different models (V8 and
        original model) on a given input image `img0`. It returns the predicted
        bounding boxes and orientations for both models.

        Args:
            img0 (image/tensor/np array/numpy-like object.): 3D mesh model being
                used for inference.
                
                		- `img0`: The input image to be analyzed. It is an instance of
                the `Image` class, which represents a 2D image with possible padding
                or cropping.
                		- `img_ori`: The original input image before any transformation.
                It is also an instance of the `Image` class and serves as a reference
                for the inference.
            img_ori (ndarray ( NumPy array) as suggested by its declaration with
                `img_ori` as the variable name.): original image used for training
                the models and serving as input to the `get_orientation_data()` function.
                
                		- `img_ori`: The input image to perform inference on. It is
                assumed to have a numpy array format and various attributes such
                as shape (e.g., `(1024, 768)` for a single image).

        Returns:
            dict: a list of bounding boxes and their corresponding orientations
            for each detected object.

        """
        orientation_bboxes, orientations = self.get_orientation_data(img_ori, img0)
        out_v8 = self.v8_model.predict(img0, show_conf=True, classes=_DETECTED_OBJECTS)
        return out_v8, orientation_bboxes, orientations

    def to_visualelements_interface(self, tracks, image_timestamp, roi):
        """
        This method generates interface data for visual objects detected in an image.

        Args:
            boxes (numpy array): Array of bounding boxes for detected objects, each box is an array [x1, y1, x2, y2].
            scores (numpy array): Array of scores indicating the confidence of each detected object.
            cls_inds (numpy array): Array of class indices corresponding to the type of each detected object.

        The method filters out detected objects based on whether their class is in a pre-defined list of classes.
        It then creates an object (act_object) for each of the remaining detections and populates it with information
        including the bounding box, score, and type of the detected object.

        The method also includes the region of interest (ROI) in the RGB image where the object was detected.
        """
        objects = []
        for track in tracks:
            x_pose = round(track._pose[0], 2)
            y_pose = round(track._pose[1], 2)
            z_pose = round(track._pose[2], 2)
            generic_attrs = {
                "score": str(track.score),
                "bbox_left": str(int(track.bbox[0])),
                "bbox_top": str(int(track.bbox[1])),
                "bbox_right": str(int(track.bbox[2])),
                "bbox_bot": str(int(track.bbox[3])),
                "x_pos": str(x_pose),
                "y_pos": str(y_pose),
                "z_pos": str(z_pose),
                "orientation": str(round(float(track.orientation), 2))
            }
            # object_ = ifaces.RoboCompVisualElementsPub.TObject(id=int(track.track_id), type=track.clase, attributes=generic_attrs, image=self.mask_to_TImage(track.image, roi))
            object_ = ifaces.RoboCompVisualElementsPub.TObject(id=int(track.track_id), type=track.clase,
                                                               attributes=generic_attrs)
            objects.append(object_)
        # print("IMAGE TIMESTAMP", image_timestamp)
        return ifaces.RoboCompVisualElementsPub.TData(timestampimage=image_timestamp, timestampgenerated=int(time.time() * 1000), period=self.Period, objects=objects)

    def mask_to_TImage(self, mask, roi):
        """
        Takes a binary mask and a rectangular region of interest (ROI) as input
        and returns a `TImage` object representing the mask within the ROI.

        Args:
            mask (ndarray (or NumPy array).): 2D mask image that will be used to
                cropped the input `roi` image and return a new `TImage` instance.
                
                		- `mask`: A numpy array representing an image mask with shape
                `(y, x, 3)`. The dimensions `y`, `x`, and `3` represent the height,
                width, and color channels (red, green, and blue) of the mask.
            roi (ndarray.): 2D rectangle of interest within the image, which is
                used to crop and return only the portion of the original image
                that falls within the ROI.
                
                		- `y`, `x`: The height and width of the `roi`, respectively,
                which represent the spatial coordinates of the cropped region.

        Returns:
            ifaces.RoboCompCamera360RGB.TImage: a TImage object containing the
            masked image with the specified ROI.
            
            	1/ `image`: This is a bytes object representing the masked image.
            	2/ `height`: This is the height of the masked image in pixels.
            	3/ `width`: This is the width of the masked image in pixels.
            	4/ `roi`: This represents the region of interest (ROI) in the masked
            image, specified as a tuple of (x, y) coordinates.

        """
        y, x, _ = mask.shape
        return ifaces.RoboCompCamera360RGB.TImage(image=mask.tobytes(), height=y, width=x, roi=roi)

    def associate_orientation_with_segmentation(self, people, orientation_bboxes, orientations):
        """
        This method associates the orientation of each person with the corresponding segmentation mask.

        Args:
            people (dict): Dictionary containing the information of the detected people.
            orientation_bboxes (numpy array): Array of bounding boxes for detected objects, each box is an array [x1, y1, x2, y2].
            orientations (numpy array): Array of orientations for detected objects, each orientation is a float value.

        The method iterates over the detected people and calculates the distance between each person and the detected
        orientations. The orientation with the minimum distance is then associated with the person.
        """
        for i, person in enumerate(people["bboxes"]):
            min_distance = 100000
            for j, orientation_bbox in enumerate(orientation_bboxes):
                distance = self.get_distance(person, orientation_bbox)
                if distance < min_distance:
                    min_distance = distance
                    people["orientations"][i] = np.deg2rad(orientations[j][0])
        return people
    def get_distance(self, person, orientation_bbox):
        """
        This method calculates the distance between a person and an orientation.

        Args:
            person (numpy array): Array containing the bounding box of a person, in the format [x1, y1, x2, y2].
            orientation_bbox (numpy array): Array containing the bounding box of an orientation, in the format [x1, y1, x2, y2].

        The method calculates the distance between the center of the person and the center of the orientation.
        """
        person_center = self.get_center(person)
        orientation_center = self.get_center(orientation_bbox)
        distance = np.linalg.norm(person_center - orientation_center)
        return distance
    def get_center(self, bbox):
        """
        This method calculates the center of a bounding box.

        Args:
            bbox (numpy array): Array containing the bounding box of an object, in the format [x1, y1, x2, y2].

        The method calculates the center of the bounding box as the average of the top left and bottom right corners.
        """
        center = (bbox[0:2] + bbox[2:4]) / 2
        return center

    def get_pose_data(self, result):
        """
        Takes an input list of detection results and returns lists of bounding
        boxes, confidence values, and skeleton data for each person in the image.

        Args:
            result (ndarray of shape (N, 5), where N is the number of frames or
                images in the input sequence.): 2D or 3D bounding box coordinates
                and corresponding confidence scores for each person in the image,
                which are used to generate high-quality documentation for the code.
                
                		- `result.keypoints`: A numpy array containing the keypoints (x
                and y coordinates) of the people in the image, in format `(N, 2)`
                where N is the number of people in the image.
                		- `result.boxes`: A list of bounding boxes around each person
                in the image, in format `(N, 4)` where N is the number of people
                in the image. The four elements in each box are `(x1, y1, x2, y2)`,
                representing the top-left corner of the box and its size.
                		- `result.conf`: A numpy array containing the confidence scores
                of each person's pose estimation, in format `(N,)` where N is the
                number of people in the image.
                
                	Note: In the function signature, `self` refers to the instance
                of the class that `get_pose_data` belongs to. The other arguments
                provided to `get_pose_data` are:
                
                		- `result`: The deserialized input containing the pose estimation
                results.
                
                	The function then iterates over each person in the image, and for
                each person, it extracts their keypoints, bounding box, and
                confidence score, and stores them in lists called `pose_bboxes`,
                `pose_confidences`, and `skeletons`. Finally, it returns these
                lists as output.

        Returns:
            list: a list of 3D bounding boxes, confidence scores, and skeleton
            data for each person in the image.

        """
        pose_bboxes = []
        pose_confidences = []
        skeletons = []
        for result in result:
            if result.keypoints != None and result.boxes != None:
                boxes = result.boxes
                keypoints = result.keypoints.xy.cpu().numpy().astype(int)
                if len(keypoints) == len(boxes):
                    for i in range(len(keypoints)):
                        person_bbox = boxes[i].xyxy.cpu().numpy().astype(int)[0]
                        pose_bboxes.append(person_bbox)
                        pose_confidences.append(boxes[i].conf.cpu().numpy()[0])
                        skeletons.append(keypoints[i])
        return pose_bboxes, pose_confidences, skeletons, [0] * len(boxes)

    def get_segmentator_data(self, results, color_image, depth_image):
        """
        Takes a list of detection results, color image, and depth image as input
        and returns a tuple containing two lists: "people" and "objects". Each
        list contains a list of dictionaries containing information about the
        detected objects (or people) such as bounding boxes, class labels, confidence
        scores, masks, and orientations.

        Args:
            results (str): 3D object detections and segmentation results produced
                by another code module or application, which is passed to the
                `get_segmentator_data()` function for further processing and analysis.
            color_image (ndarray object.): 2D color image that is used to compute
                the masks and hashes for each person or object in the scene.
                
                		- `roi_ysize`: The height of the ROI (Region of Interest) in the
                color image.
                		- `roi_xsize`: The width of the ROI in the color image.
                		- `shape`: The shape of the color image, which is (ROI ysize,
                ROI xsize, 3) in this case.
                
                	The `color_image` is a NumPy array containing the ROI of the color
                image. It has dimensions (ROI ysize, ROI xsize, 3), where each
                pixel has a value in the RGB color space. The image may be captured
                from various sources such as a camera or an existing image.
            depth_image (ndarray or NumPy array, specifically an array of shape
                (Height, Width, Number of Channels) representing a depth image.):
                2D image of depth values for the scene, which is used to compute
                the 3D positions of objects in the scene along with their corresponding
                masks.
                
                		- `roi_ysize`: The size of the region of interest (ROI) in the
                depth image.
                		- `roi_xsize`: The size of the ROI in the x-axis direction in
                the depth image.
                		- `shape`: The shape of the depth image, which is a 3D NumPy
                array containing the pixel values of the image.
                
                	These properties are used in the function to extract specific
                information from the depth image, such as the coordinates of the
                object masks, the class labels, and the confidence scores.

        Returns:
            tuple: a list of dictionaries containing information about the objects
            and people detected in an image.

        """
        people = {"bboxes": [], "poses": [], "confidences": [], "masks": [], "classes": [], "orientations": [],
                  "hashes": []}
        objects = {"bboxes": [], "poses": [], "confidences": [], "masks": [], "classes": [], "orientations": [],
                   "hashes": []}
        roi_ysize, roi_xsize, _ = color_image.shape
        for result in results:
            if result.masks != None and result.boxes != None:
                masks = result.masks.xy
                boxes = result.boxes
                if len(masks) == len(boxes):
                    for i in range(len(boxes)):
                        element_confidence = boxes[i].conf.cpu().numpy()[0]
                        if element_confidence > 0.6:
                            element_class = boxes[i].cls.cpu().numpy().astype(int)[0]
                            element_bbox = boxes[i].xyxy.cpu().numpy().astype(int)[0]
                            image_mask = np.zeros((roi_ysize, roi_xsize, 1), dtype=np.uint8)
                            act_mask = masks[i].astype(np.int32)
                            cv2.fillConvexPoly(image_mask, act_mask, (1, 1, 1))
                            image_mask_element = image_mask[element_bbox[1]:element_bbox[3],
                                                 element_bbox[0]:element_bbox[2]]
                            color_image_mask = color_image[element_bbox[1]:element_bbox[3],
                                               element_bbox[0]:element_bbox[2]]
                            element_mask = self.get_mask_with_modified_background(image_mask_element, color_image_mask)
                            element_hash = self.get_color_histogram(element_mask)
                            height, width, _ = image_mask_element.shape
                            depth_image_mask = depth_image[element_bbox[1]:element_bbox[3],
                                               element_bbox[0]:element_bbox[2]]
                            element_pose = self.get_mask_distance(image_mask_element, depth_image_mask)
                            if element_pose != [0, 0, 0]:
                                if element_class == 0:
                                    people["poses"].append(element_pose)
                                    people["bboxes"].append(element_bbox)
                                    people["confidences"].append(element_confidence)
                                    people["masks"].append(color_image_mask)
                                    people["classes"].append(element_class)
                                    people["hashes"].append(element_hash)
                                else:
                                    objects["bboxes"].append(element_bbox)
                                    objects["poses"].append(element_pose)
                                    objects["confidences"].append(element_confidence)
                                    objects["masks"].append(color_image_mask)
                                    objects["classes"].append(element_class)
                                    objects["hashes"].append(element_hash)

        people["orientations"] = [-4] * len(people["bboxes"])
        objects["orientations"] = [-4] * len(objects["bboxes"])
        return people, objects

    def get_mask_distance(self, mask, depth_image):
        # Get bbox center point
        # Get depth image shape and calculate bbox center
        """
        Computes the distance of a person from a depth image based on the binary
        mask of the person's body parts. It retrieves the center point and width
        of the bbox, extracts valid points from the depth image, and calculates
        the distances to those points using the histogram method. Finally, it
        returns the median distance of the valid points.

        Args:
            mask (ndarray.): 2D segmentation of the depth image, which is used to
                extract the valid points (i.e., points with non-zero values) and
                calculate their distances from the bbox center.
                
                	1/ `mask`: A numpy array of shape `(depth_image.shape[0],
                depth_image.shape[1])`, where each element is either 0 or 1,
                indicating whether a pixel in the depth image belongs to the object
                of interest or not.
                	2/ `depth_image`: A numpy array representing the depth image,
                which is assumed to be a rectangular array with dimensions
                corresponding to the shape of the depth sensor's image.
                	3/ `bbox_center`: A numpy array of length 2, containing the center
                coordinates of the bounding box around the object of interest in
                pixels.
                	4/ `segmentation_points`: A numpy array of shape `(mask.shape[1],
                mask.shape[0])`, containing the indices of the pixels in the depth
                image that belong to the object of interest.
                	5/ `p`: A float value representing the bbox width, which is used
                to calculate the minimum distance between points in the segmentation
                mask and their corresponding depth values.
            depth_image (3D array, specifically an array with shape ( height,
                width, channels) representing a depth image.): 2D depth image that
                the function will analyze to determine the person's pose.
                
                		- `shape`: The shape of the depth image, which can be a 3D array
                with shape `( height, width, depth )`.
                		- `bbox_center`: The center point of the bounding box (bbox) for
                the object in the depth image. The coordinates are calculated as
                `(depth_image_shape[1] // 2, depth_image_shape[0] // 2)`.
                		- `segmentation_points`: An array containing the coordinates of
                the points on which the bbox is applied to the mask. The shape of
                this array is `NxM`, where `N` is the number of points used to
                calculate the bbox center, and `M` is the number of channels in
                the depth image (typically 3 for a RGB depth image).
                		- `p`: The distance from the bbox center at which points are
                considered inside the bbox. The value is calculated as
                `depth_image_shape[1] // 4.5`.
                		- `valid_points`: An array containing the coordinates of the
                points that pass the bbox filtering. The shape of this array is
                `Nx2`, where `N` is the number of points remaining after bbox filtering.
                		- `distances`: The distances between the valid points and the
                bbox center, calculated as `np.linalg.norm(valid_points - bbox_center,
                axis=1)`.
                		- `hist`: A 2D histogram array created from the `distances` array
                using `np.histogram`. The shape of this array is `(bins, height)`.
                		- `edges`: An array containing the edges of each bin in the
                histogram. The shape of this array is `(bins, width)`.
                		- `person_dist`: An array containing the distances between the
                bbox center and the points inside the mold interval. The shape of
                this array is `(1,)` or `()`.

        Returns:
            list: a list of three values representing the mean distance of a person
            from the camera at different positions.

        """
        depth_image_shape = depth_image.shape
        bbox_center = [depth_image_shape[1] // 2, depth_image_shape[0] // 2]
        segmentation_points = np.argwhere(np.all(mask == 1, axis=-1))[:, [1, 0]]
        # p = bbox width / 4
        p = depth_image_shape[1] // 4.5
        segmentation_points = segmentation_points[np.linalg.norm(segmentation_points - bbox_center, axis=1) < p]
        # Extraer directamente los puntos válidos (no ceros) y sus distancias en un solo paso.
        # Esto evitará tener que crear y almacenar matrices temporales y booleanas innecesarias.
        valid_points = depth_image[segmentation_points[:, 1], segmentation_points[:, 0]]
        valid_points = valid_points[np.any(valid_points != [0, 0, 0], axis=-1)]
        # print valid points
        if valid_points.size == 0:
            return [0, 0, 0]
        distances = np.linalg.norm(valid_points, axis=-1)
        # Parece que intentas generar bins para un histograma basado en un rango de distancias.
        # Crear los intervalos directamente con np.histogram puede ser más eficiente.
        hist, edges = np.histogram(distances, bins=np.arange(np.min(distances), np.max(distances), 300))
        if hist.size == 0:
            pose = np.mean(valid_points, axis=0).tolist()
            return pose
        # Identificar el intervalo de moda y extraer los índices de las distancias que caen dentro de este intervalo.
        max_index = np.argmax(hist)
        pos_filtered_distances = np.logical_and(distances >= edges[max_index], distances <= edges[max_index + 1])
        # Filtrar los puntos válidos y calcular la media en un solo paso.
        person_dist = np.mean(valid_points[pos_filtered_distances], axis=0).tolist()
        return person_dist

    # def get_color_histogram(self, color):
    #     color =cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    #     hist = cv2.calcHist([color], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    #     return self.normalize_histogram(hist)

    #GPT VERSION: white pixels deleted
    def get_color_histogram(self, color):
        # Convert the color image to HSV
        """
        Converts an image from RGB to HSV, creates a mask to exclude unwanted
        pixels, and calculates the histogram of the resulting HSV image using the
        `cv2.calcHist` function.

        Args:
            color (3-dimensional numpy array.): 3D RGB image to be converted to
                HSV and histogrammed.
                
                		- `color` is a NumPy array with 3 color channels (RGB) representing
                a single color image.
                		- The shape of `color` is `(height, width, 3)`, where `height`
                and `width` are the dimensions of the image.
                		- Each element in `color` has one of three possible values for
                each channel: 0 for black, 255 for white, and any other value for
                a colored pixel.
                		- The data type of `color` is `np.uint8`, indicating that the
                pixels are represented as 8-bit integers.

        Returns:
            array of normalized histogram values: a normalized histogram of the
            HSV color space of the input color image.
            
            	1/ `hist`: The output is a 3D array with shape `(num_bins, num_pixels,
            num_colors)`, where `num_bins` is the number of bins in the histogram,
            `num_pixels` is the total number of pixels in the input image, and
            `num_colors` is the number of distinct colors present in the image.
            	2/ `normalize_histogram`: The output of this function is a normalized
            histogram, where each bin represents the distribution of pixels in
            that color category across all images. This ensures that the histogram
            is comparable and meaningful across different images and datasets.

        """
        color_hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)

        # Create a mask to exclude pixels with Saturation = 0 and Value = 256
        mask = (color_hsv[:, :, 1] > 0) & (color_hsv[:, :, 2] < 256)

        # Calculate histogram using the mask to exclude unwanted pixels
        hist = cv2.calcHist([color_hsv], [0, 1, 2], mask.astype(np.uint8), [8, 8, 8], [0, 256, 0, 256, 0, 256])

        return self.normalize_histogram(hist)

    def normalize_histogram(self, hist):
        """
        Normalizes a given histogram by dividing it by the total number of pixels
        in the histogram, resulting in a normalized histogram with values between
        0 and 1.

        Args:
            hist (ndarray (i.e., an array-like object).): 2D histogram that needs
                to be normalized.
                
                		- `np.sum(hist)` represents the total number of pixels in the histogram.

        Returns:
            array of values: a normalized histogram of pixel values.
            
            	1/ Total Pixels: The variable `total_pixels` represents the total
            number of pixels in the histogram. It is calculated by summing up all
            the values in the histogram.
            	2/ Normalized Histogram: The variable `normalized_histogram` represents
            the normalized histogram, which is the histogram divided by the total
            number of pixels. This creates a probability distribution where the
            sum of all the probabilities is equal to 1.
            	3/ Properties of Normalized Histogram: The normalized histogram has
            several desirable properties, including:
            			- Probability values range from 0 to 1
            			- The sum of all the probability values is equal to 1
            			- Each bin represents a continuous range of pixel values
            
            	In summary, the `normalize_histogram` function transforms an input
            histogram into a normalized form, which simplifies further processing
            or analysis of the histogram.

        """
        total_pixels = np.sum(hist)
        normalized_hist = hist / total_pixels
        return normalized_hist

    def get_orientation_data(self, processed_image, original_image):
        """
        1/ takes a processed image and its original image as input, 2. applies a
        model to the processed image with augmentation enabled, 3. generates an
        output with non-maximum suppression, 4. extracts orientation bboxes from
        the output, and 5. returns both the orientation bboxes and orientations
        in native space.

        Args:
            processed_image (ndarray.): 2D image that has been processed by some
                previous step or transformation, and is provided to the
                `get_orientation_data` function as an argument for the purpose of
                estimating orientations from it.
                
                		- `roi_xsize`: The width of the ROI (Region of Interest) in pixels.
                		- `original_image`: The original image from which the ROI was extracted.
                		- `augment`: A boolean value indicating whether to apply
                augmentations during feature extraction.
                		- `scales`: An array of scales to apply to the feature extraction
                output, including the scale of the ROI.
                		- `num_angles`: The number of angles in the orientation heatmap.
            original_image (ndarray.): 2D image that was preprocessed and passed
                through the model as part of the multi-scale feature extraction process.
                
                		- `shape`: The shape of the original image, which is (height,
                width) or (depth, height, width), where depth refers to the number
                of color channels (e.g., RGB).
                		- `size`: The size of the original image in pixels, which can
                be used to calculate the scale factor for non-max suppression.

        Returns:
            int: a tuple of two numpy arrays: `orientation_bboxes` and `orientations`.

        """
        out_ori = self.model(processed_image, augment=True, scales=[self.roi_xsize / 640])[0]
        out = non_max_suppression(out_ori, 0.3, 0.5, num_angles=self.data['num_angles'])
        orientation_bboxes = scale_coords(processed_image.shape[2:], out[0][:, :4], original_image.shape[:2]).cpu().numpy().astype(int)  # native-space pred
        orientations = (out[0][:, 6:].cpu().numpy() * 360) - 180   # N*1, (0,1)*360 --> (0,360)
        return orientation_bboxes, orientations

    # Calculate image ROI for element centering
    def set_roi_dimensions(self, objects):
        """
            Set Region of Interest (ROI) based on objects.

            Args:
                objects (list): List of detected objects. Each object contains information about its position and size.

            The method goes through the list of objects and when it finds the object that matches the tracked_id,
            it calculates the desired ROI based on the object's position and size. The ROI is then stored in the class's attributes.
            """
        for object in objects:
            if object.id == self.tracked_id:
                roi = object.image.roi
                x_roi_offset = roi.xcenter - roi.xsize / 2
                y_roi_offset = roi.ycenter - roi.ysize / 2
                x_factor = roi.xsize / roi.finalxsize
                y_factor = roi.ysize / roi.finalysize

                left = int(object.left * x_factor + x_roi_offset)
                right = (object.right * x_factor + x_roi_offset)

                top = int(object.top * y_factor + y_roi_offset)
                bot = int(object.bot * y_factor + y_roi_offset)

                self.target_roi_xcenter = (left + (right - left)/2) % self.rgb_original.width
                self.target_roi_ycenter = (top + (bot - top)/2)
                self.target_roi_ysize = np.clip(int((bot - top)*2), 0, self.rgb_original.height)
                self.target_roi_xsize = self.target_roi_ysize
                return
        
        self.tracked_element = None
        self.tracked_id = None
        self.target_roi_xcenter = self.rgb_original.width // 2
        self.target_roi_ycenter = self.rgb_original.height // 2
        self.target_roi_xsize = self.rgb_original.width // 2
        self.target_roi_ysize = self.rgb_original.height

    ###############################################################

    def show_fps(self, alive_time, period):
        """
        Calculates and displays the frame rate, alive time, period, and update
        rate of an image processing thread. It updates the values every 1 second
        (or a user-defined period) and limits the thread's update rate to prevent
        excessive CPU usage.

        Args:
            alive_time (int): duration of time the object has been alive or running,
                and it is used to calculate the current period of the image
                processing task.
            period (int): interval between FPS measurements, and it is used to
                calculate the current frame rate based on the elapsed time since
                the last measurement.

        """
        if time.time() - self.last_time > 1:
            self.last_time = time.time()
            cur_period = int(1000./self.cont)
            delta = (-1 if (period - cur_period) < -1 else (1 if (period - cur_period) > 1 else 0))
            print("Freq:", self.cont, "Hz. Alive_time:", alive_time, "ms. Img period:", int(period),
              "ms. Curr period:", cur_period, "ms. Inc:", delta, "Timer:", self.thread_period)
            self.thread_period = np.clip(self.thread_period+delta, 0, 200)
            self.cont = 0
        else:
            self.cont += 1

    def display_data_tracks(self, img, elements): #Optimizado
        """
        This function overlays bounding boxes and object information on the image for tracked objects.

        Args:
            img (numpy array): The image to display object data on.
            elements (list): Tracked objects with bounding box coordinates, scores, and class indices.
            class_names (list, optional): Names of the classes.

        Returns:
            img (numpy array): The image with overlaid object data.
        """
        for element in elements:
            x0, y0, x1, y1 = map(int, [int(float(element.attributes["bbox_left"])), int(float(element.attributes["bbox_top"])), int(float(element.attributes["bbox_right"])), int(float(element.attributes["bbox_bot"]))])
            cls_ind = element.type
            color = (_COLORS[cls_ind] * 255).astype(np.uint8).tolist()
            # text = f'Class: {class_names[cls_ind]} - Score: {element.score * 100:.1f}% - ID: {element.id}'
            text = f'{float(element.attributes["x_pos"])} - {float(element.attributes["y_pos"])} - {element.id} - {float(element.attributes["orientation"])} - {_OBJECT_NAMES[element.type]}'
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_ind]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]

            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            txt_bk_color = (_COLORS[cls_ind] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
        return img

    ############################################################################################
    def startup_check(self):
        """
        Runs tests to verify the functionality of various components of a robot
        vision system, including TImage, TDepth, TRGBD, and TGroundTruth classes
        from ifaces.RoboCompCameraRGBDSimple and ifaces.RoboCompHumanCameraBody
        modules, as well as KeyPoint, Person, and PeopleData classes.

        """
        print(f"Testing RoboCompCameraRGBDSimple.TImage from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TImage()
        print(f"Testing RoboCompCameraRGBDSimple.TDepth from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TDepth()
        print(f"Testing RoboCompCameraRGBDSimple.TRGBD from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TRGBD()
        print(f"Testing RoboCompHumanCameraBody.TImage from ifaces.RoboCompHumanCameraBody")
        test = ifaces.RoboCompHumanCameraBody.TImage()
        print(f"Testing RoboCompHumanCameraBody.TGroundTruth from ifaces.RoboCompHumanCameraBody")
        test = ifaces.RoboCompHumanCameraBody.TGroundTruth()
        print(f"Testing RoboCompHumanCameraBody.KeyPoint from ifaces.RoboCompHumanCameraBody")
        test = ifaces.RoboCompHumanCameraBody.KeyPoint()
        print(f"Testing RoboCompHumanCameraBody.Person from ifaces.RoboCompHumanCameraBody")
        test = ifaces.RoboCompHumanCameraBody.Person()
        print(f"Testing RoboCompHumanCameraBody.PeopleData from ifaces.RoboCompHumanCameraBody")
        test = ifaces.RoboCompHumanCameraBody.PeopleData()
        QTimer.singleShot(200, QApplication.instance().quit)

    # =============== Ice required interfaces ==================
    #
    # IMPLEMENTATION of getVisualObjects method from VisualElements interface
    #
    def VisualElements_getVisualObjects(self, objects):
        return self.objects_read
    # ===================================================================
    # ===================================================================

    #
    # SUBSCRIPTION to setTrack method from SegmentatorTrackingPub interface
    #
    def SegmentatorTrackingPub_setTrack(self, track):
        """
        Sets chosen track ID and stores it in tracker. It also retrieves target
        ROI coordinates based on given track ID.

        Args:
            track (`Track` object.): 3D tracking object to be associated with the
                Segmentator, and when provided, sets the chosen track for the
                Segmentator's tracker.
                
                		- `id`: An integer value representing the unique identifier of
                the track to be followed.
                		- `chosen_track`: A reference to a track object that has been
                chosen by the user.
                		- `target_roi_xcenter`: The x-coordinate of the target region
                of interest (ROI) center.
                		- `target_roi_ycenter`: The y-coordinate of the target ROI center.
                		- `target_roi_xsize`: The x-size of the target ROI.
                		- `target_roi_ysize`: The y-size of the target ROI.
                
                	The function then processes the `track` object based on its
                properties, setting the `tracker.set_chosen_track()` and updating
                internal variables accordingly.

        """
        self.tracker.set_chosen_track(track.id)
        if track.id == -1:
            self.tracked_element = None
            self.tracked_id = None
            self.target_roi_xcenter = self.rgb_original.width // 2
            self.target_roi_ycenter = self.rgb_original.height // 2
            self.target_roi_xsize = self.rgb_original.width // 2
            self.target_roi_ysize = self.rgb_original.height
            return

        for track_obj in self.objects_read:
            if track_obj.id == track.id:
                self.target_roi_xcenter_list = queue.Queue(10)
                self.tracked_element = track_obj
                self.tracked_id = track.id
                
                return
    def update_plot(self,frame):
        """
        Updates a plot by clearing it and re-drawing it with a subset of data from
        a list of IDs.

        Args:
            frame (`object`.): 2D plot that is to be updated with new data.
                
                		- `frame`: A Pandas DataFrame object containing time series data
                with two columns - `xs` (with values of type `float`) representing
                the time series and `ys` (with values of type `float`) representing
                the corresponding values.

        """
        pass
        # self.ax.clear()  # Limpia el plot actual
         
        # # Limit x and y lists to 20 items
        # self.xs = self.xs[-20:]
        # for i in self.id_list:
        #     self.ys[i] = self.ys[i][-20:]
        #     self.ax.plot(self.xs,self.ys[i], marker='o', linestyle='-')
        # self.ax.set_xlabel('Tiempo')
        # self.ax.set_ylabel('Valor')

        # plt.xticks(rotation=45, ha='right')
        # plt.subplots_adjust(bottom=0.30)

        # return self.ax,
    ######################
    # From the RoboCompByteTrack you can call this methods:
    # self.bytetrack_proxy.allTargets(...)
    # self.bytetrack_proxy.getTargets(...)
    # self.bytetrack_proxy.getTargetswithdepth(...)
    # self.bytetrack_proxy.setTargets(...)

    ######################
    # From the RoboCompByteTrack you can use this types:
    # RoboCompByteTrack.Targets

    ######################
    # From the RoboCompCamera360RGBD you can call this methods:
    # self.camera360rgbd_proxy.getROI(...)

    ######################
    # From the RoboCompCamera360RGBD you can use this types:
    # RoboCompCamera360RGBD.TRoi
    # RoboCompCamera360RGBD.TRGBD

    ######################
    # From the RoboCompVisualElements you can use this types:
    # RoboCompVisualElements.TRoi
    # RoboCompVisualElements.TObject

    # From the RoboCompLidarOdometry you can call this methods:
    # self.lidarodometry_proxy.getFullPoseEuler(...)
    # self.lidarodometry_proxy.getFullPoseMatrix(...)
    # self.lidarodometry_proxy.getPoseAndChange(...)
    # self.lidarodometry_proxy.reset(...)

    # =============== DSR SLOTS  ================
    # =============================================

    def update_node_att(self, id: int, attribute_names: [str]):
        console.print(f"UPDATE NODE ATT: {id} {attribute_names}", style='green')

    def update_node(self, id: int, type: str):
        console.print(f"UPDATE NODE: {id} {type}", style='green')

    def delete_node(self, id: int):
        console.print(f"DELETE NODE:: {id} ", style='green')

    def update_edge(self, fr: int, to: int, type: str):

        """
        Prints a message and updates an edge in a graph based on its label.

        Args:
            fr (int): 0-based index of the first node in the range of nodes being
                updated.
            to (int): type of edge being updated in the code.
            type (str): edge being updated, and it is used to identify the specific
                edge being modified in the code.

        """
        console.print(f"UPDATE EDGE: {fr} to {type}", type, style='green')

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')

    def delete_edge(self, fr: int, to: int, type: str):
        console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
