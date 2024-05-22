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
        Sets up various components and connections for an object recognition system,
        including setting unique IDs, connecting signals, loading YOLO and jointBDOE
        models, and initializing timers.

        Args:
            proxy_map (int): 2D mapping of the camera's view to a fixed-size image,
                which is used to scale and transform the camera's ROI (Region Of
                Interest) into a pixel space for processing and analysis.
            params (object, possibly from an external module.): 360rgbd proxy
                object and provides access to its various methods and properties
                for use within the function.
                
                		- `self.agent_id`: A unique integer ID assigned to this agent
                in the deployment. This is an required field and has type `int`.
                		- `g`: A reference to a `DSRGraph` object, which is the environment
                object perception graph. This is an instance of the `DSRGraph`
                class and has no additional attributes.
                		- `startup_check`: A boolean indicating whether a startup check
                should be performed. This is an optional field and has type `bool`.
                		- ` Period`: The sampling period in milliseconds. This is an
                optional field and has type `int`.
                		- `thread_period`: The inference execution period in milliseconds.
                This is an optional field and has type `int`.
                		- `display`: A boolean indicating whether the display should be
                turned on or off. This is an optional field and has type `bool`.
                
                	The `params` dictionary contains various properties, including:
                
                		- `camera360rgbd_proxy`: A reference to a `Camera360RgbdProxy`
                object, which provides access to the 360-degree RGBD camera data.
                This is an instance of the `Camera360RgbdProxy` class and has no
                additional attributes.
                		- `lidarodometry_proxy`: A reference to a `LidarOdometryProxy`
                object, which provides access to the lidar odometry data. This is
                an instance of the `LidarOdometryProxy` class and has no additional
                attributes.
                		- `rgb_read_queue`: A queue of RGB image read threads. This is
                a Python queue instance and has no additional attributes.
                		- `inference_read_queue`: A queue of inference execution threads.
                This is a Python queue instance and has no additional attributes.
                
                	The `params` dictionary also contains the `timer` object, which
                is used to schedule the computation of the RGB images. The `
                timer.timeout.connect(self.compute)` line connects the `compute`
                method to the `timer.timeout` event.
            startup_check (int): startup check functionality, which is executed
                only if the value is `True`, and it checks the camera feed, lidar
                odometry data, and YOLO model for validation before proceeding
                with the main functionality of the agent.

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
        Parses configuration parameters from a specified file and sets class,
        display, and depth flags based on the parsed values.

        Args:
            params (dict): configuration parameters for the object detection system,
                including the YOLO model, display options, depth flag, and class
                path file contents.

        Returns:
            bool: a boolean value indicating whether the parameters were read
            successfully or not.

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
                # img_back = self.display_data_tracks(color_back, front_objects.objects)
                # img_int = img.astype('float32') / 255.0
                # image_comp = cv2.addWeighted(img_int, 0.5, depth, 0.5, 0)
                # cv2.imshow("back", img_back)
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
        transformation matrix by the second transformation matrix and then scaling
        the resulting matrix by a factor of 1000.

        Args:
            transformacion1 (2D matrix.): 4x4 matrix that describes the first
                transformation applied to the image data.
                
                		- `np.linalg.inv(transformacion1)` is the inverse of the matrix
                `transformacion1`, which is a 3x3 numpy array.
                		- The elements of `transformacion1` are integers between 0 and
                20, representing the stretching or shrinking factors for each
                dimension (row, column, and element-wise) in the first transformation
                matrix.
            transformacion2 (3x3 (three-by-three) numpy array.): 3x4 transformation
                matrix that, when multiplied by the inverse of the first transformation
                matrix (`inversa_transformacion1`), generates the differential matrix.
                
                		- Shape: `(4, 4)` indicating that the matrix has 4 rows and 4 columns.
                		- dtype: `'float64'` indicating that the matrix elements are
                numerical values with a data type of float64.
                		- itemsize: 80 implies that each element in the matrix takes up
                80 bytes of memory.

        Returns:
            float: a matrix that has been transformed by multiplying its inverse
            by the second transformation matrix, with a scaling factor of 1000
            added to the top-right corner.

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
        1) searches for a YOLOv8 model in the current directory and surrounding
        directories using glob, 2) loads the model if found, or 3) prompts the
        user to select one of the available models (yolov8n, yolov8s, yolov8m,
        yolov8l, yolov8x) and downloads and converts it.

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
        1)loads a pre-trained YOLO model and exports it to TensorFlow format (TRT),
        then loads the resulting TRT model.

        Args:
            model_name (str): name of the YOLO model that needs to be converted
                to the TRT format.

        """
        self.v8_model = YOLO(model_name)
        # Export the model to TRT
        self.v8_model.export(format='engine', device='0')
        self.v8_model = YOLO(model_name + '.engine')
    def load_jointbdoe_model(self):
        """
        Is responsible for retrieving and preprocessing a JointBDOE model, as well
        as loading associated data from a given location.

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
        Transforms an input image into a PyTorch tensor, by reshaping it from HWC
        (Height x Width x Channels) to CHW (Height x Width x Number of Channels),
        normalizing the values to the range [0, 1], and expanding the tensor for
        batching if necessary.

        Args:
            image (ndarray or tensor object.): 3D tensor containing the original
                image data that is being transformed into a more suitable format
                for further processing.
                
                		- `image`: A numpy array representing the input image.
                		- `640`: The height of the desired output tensor.
                		- `stride`: The stride of the convolution kernel used in letterboxing.
                		- `auto`: A boolean indicating whether to automatically detect
                the format of the input image.

        Returns:
            float: a torch tensor representing the input image after conversion
            from a numpy array.

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
        Reads RGB data from a queue, performs depth estimation using a neural
        network, and appends the output to a new queue for further processing. It
        also waits for an event before continuing to the next iteration.

        Args:
            event (Event): event triggered by the main loop, causing the function
                to stop and check for pop messages from the read queue.

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
        Takes an image and a model as input, runs the model on the image to predict
        bounding boxes for objects in the image, and returns the predicted bounding
        boxes, orientations, and outputs of the model.

        Args:
            img0 (image.): 2D image that will be processed by the object detection
                model to obtain the predicted bounding boxes and class probabilities.
                
                		- `img0`: The input image for the inference.
                		- `show_conf`: An optional parameter that specifies whether to
                display confidence labels on the input image (True) or not (False).
            img_ori (2D image.): original image that is to be processed and analyzed
                by the object detection model.
                
                		- `img0`: This is the input image that is passed through the
                model for inference.
                		- `orientations`: This is a list of orientation boxes detected
                in the input image using the `get_orientation_data` function.
                		- `out_v8`: This is the predicted output from the V8 model, which
                is passed as an argument to the `predict` method along with other
                arguments.

        Returns:
            instance of the class `torch.Tensor: a triplet consisting of: (1) a
            prediction score map from the V8 model; (2) an array of orientation
            boxes; and (3) an array of orientations.
            
            	1/ `out_v8`: The predicted bounding box coordinates for objects in
            the input image. It is a tensor with shape `( batch_size, num_objects,
            4 )`, where `batch_size` is the number of images in the input batch,
            and `num_objects` is the number of objects detected in each image.
            	2/ `orientation_bboxes`: The predicted orientation of the bounding
            boxes for each object. It is a tensor with shape `( batch_size,
            num_objects, 3 )`, where `batch_size` is the number of images in the
            input batch, and `num_objects` is the number of objects detected in
            each image.
            	3/ `orientations`: The orientation of the bounding boxes for each
            object. It is a tensor with shape `( batch_size, num_objects )`, where
            `batch_size` is the number of images in the input batch, and `num_objects`
            is the number of objects detected in each image.
            
            	In summary, the output of the `inference_over_image` function contains
            information about the location, orientation, and class labels of objects
            in the input image.

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
        Generates a list of bounding boxes, confidence scores, and skeleton data
        for each person in an image based on keypoints and box coordinates provided
        in the output of a detection model.

        Args:
            result (int): 2D and 3D box coordinates and confidence scores of a
                person in an image, which are used to generate high-quality
                documentation for the code.

        Returns:
            list: a list of three lists: `pose_bboxes`, `pose_confidences`, and
            `skeletons`, along with a scalar value indicating the confidence score
            for each box.

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
        Processes a given set of detection results and extracts information about
        people and objects in the images, including their bounding boxes, classes,
        confidence levels, masks, and orientations. It returns a dictionary of
        people and objects with this information.

        Args:
            results (`object`.): 2D or 3D object detection output generated by an
                object detection algorithm, which contains multiple elements such
                as masks, boxes, confidence scores, classes, and hashes for each
                detected object.
                
                		- `masks`: A list of arrays with shape (N, 3), where N is the
                number of objects detected in the image. Each array represents a
                mask for an object, with values of either 0 or 1 indicating the
                presence or absence of an object at that position.
                		- `boxes`: A list of arrays with shape (N, 4), where N is the
                number of objects detected in the image. Each array represents a
                bounding box for an object, with x, y coordinates representing the
                top-left corner of the box and width and height representing the
                dimensions of the box.
                		- `confidences`: A list of arrays with shape (N,), where N is
                the number of objects detected in the image. Each array represents
                the confidence score for an object, with higher values indicating
                a more confident detection.
                		- `classes`: A list of arrays with shape (N,), where N is the
                number of objects detected in the image. Each array represents the
                class label for an object, with values corresponding to different
                classes (e.g., person, car, tree).
                		- `orientations`: A list of arrays with shape (N,), where N is
                the number of objects detected in the image. Each array represents
                the orientation of an object, with values ranging from -4 to 4
                representing different orientations around the center of the image.
                
                	It's important to note that these properties are only defined if
                `results` is a valid output from the `Segmentor` function. If
                `results` is not a valid output, then these properties may not be
                accurate or meaningful.
            color_image (ndarray of shape `(N, Y, X)`, where `N` represents the
                number of color channels (e.g., RGB), `Y` and `X` represent the
                height and width of the image in pixels, respectively.): 3D image
                that contains the object of interest, and it is used to generate
                masks and other elements of the output `people` and `objects` arrays.
                
                		- `shape`: Returns the shape of the color image as (`roi_ysize`,
                `roi_xsize`).
                		- `masks`: A list of arrays, each containing a binary mask for
                a person/object in the color image. Each array has shape (`len(boxes),
                3`), where `boxes` is the number of boxes (people or objects) in
                the color image.
                		- `poses`: A list of arrays, each containing a set of 4D pose
                coordinates (x, y, az, elev) for a person/object in the color
                image. Each array has shape (`len(bboxes), 4`).
                		- `confidences`: A list of floating-point values representing
                the confidence score of each person/object detection in the color
                image. Each value ranges from 0 to 1.
                		- `masks`: A list of binary arrays, each containing a mask for
                a person/object in the color image. Each array has shape (`len(boxes),
                255`), where `boxes` is the number of boxes (people or objects)
                in the color image.
                		- `classes`: A list of integers representing the class label for
                each person/object detection in the color image. Each value ranges
                from 0 to 9.
                		- `orientations`: An empty list, which will be populated with a
                single integer value indicating the orientation of the people or
                objects in the color image (-4).
                
                	Note that some of these properties can be altered or modified
                based on the requirements of the function and the input data.
            depth_image (3D array with shape `(Height, Width, Depth)`.): 3D point
                cloud of the scene, which is used to calculate the pose of objects
                in the scene through the distance field of the image.
                
                		- `shape`: (Height, Width, Channels) representing the dimensions
                of the depth image.
                		- `xysize`: The height of the image.
                		- `xsize`: The width of the image.
                
                	No further information is provided in the given code snippet about
                the properties of `depth_image`. Therefore, it can be assumed that
                `depth_image` is a numpy array or tensor representing a 3D depth
                map of an scene with objects.

        Returns:
            list: a pair of dictionaries, `people` and `objects`, containing various
            metadata and features about the detected objects in an image.

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
        1) extracts valid points from a depth image based on a provided bounding
        box center, 2) calculates distances between these valid points and the
        center of the bounding box, and 3) returns the median distance of the valid
        points.

        Args:
            mask (int): 3D point cloud mask, which is used to extract the valid
                3D points from the depth image.
            depth_image (3D-numpy array or tensor, which represents an image depth
                map or volume.): 2D depth image that is used to compute the person
                pose estimation.
                
                		- `depth_image`: This is an array of shape `(height, width)`
                representing the depth image.
                		- `shape`: This attribute provides the shape of the `depth_image`
                array.
                		- `bbox_center`: This variable computes the center of the bounding
                box for segmentation point calculation, and its coordinates are
                computed as (`width // 2`, `height // 2`).
                		- `segmentation_points`: This variable calculates the coordinates
                of segmentation points in the depth image using the `argwhere`
                function. The resulting array has shape `(n, 2)`, where `n` is the
                number of valid segmentation points.
                		- `p`: This variable computes the bbox width based on the aspect
                ratio of the depth image and is defined as `depth_image_shape[1]
                // 4.5`.
                		- `valid_points`: This variable filters out invalid points from
                the depth image based on the condition `segmentation_points[np.all(mask
                == 1, axis=-1)]` and assigns them to the `valid_points` array. The
                resulting array has shape `(n, 3)`, where `n` is the number of
                valid segmentation points.
                		- `distances`: This variable computes the distances between the
                points in the depth image using the `linalg.norm` function with
                axis `-1`. The resulting array has shape `(n,)`.
                		- `hist`: This variable computes a histogram of the distance
                values using the `histogram` function and provides the histogram
                values and bin edges as the `hist` and `edges` attributes, respectively.
                		- `pos_filtered_distances`: This variable filters out invalid
                distances based on the condition `distances >= edges[max_index]`
                and `distances <= edges[max_index + 1]`, and assigns them to the
                `pos_filtered_distances` array. The resulting array has shape `(n,)`.
                		- `person_dist`: This variable computes the mean distance of the
                filtered distances using the `mean` function and provides the mean
                distance as the output of the function. The resulting array has
                shape `(1,)`.

        Returns:
            float: a list of distances from the person to the camera, measured in
            meters.

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
        Takes a color image and converts it to HSV colorspace, creates a mask to
        exclude unwanted pixels based on saturation and value ranges, calculates
        the histogram of the remaining pixels, and returns the normalized histogram.

        Args:
            color (ndarray or numpy array of type `cv2.COLOR_RGB`.): 3-dimensional
                RGB color vector that is converted to HSV using cv2.cvtColor()
                before undergoing histogram calculation and normalization.
                
                		- `cv2.COLOR_RGB2HSV`: This is the conversion function used to
                transform the RGB color image into HSV color space.
                		- `color_hsv`: The resulting HSV color image.
                		- `color`: The original RGB color image.
                		- `mask`: A binary mask created by applying two logical operations
                on the HSV image: (1) selecting pixels with non-zero saturation,
                and (2) excluding pixels with values less than 256.
                		- `hist`: The resulting histogram calculated using the masked
                HSV image.

        Returns:
            np.ndarray: a normalized histogram of the input color image, excluding
            unwanted pixels.
            
            		- `hist`: A 3D numpy array with shape `(n_bins, n_samples, 3)`
            representing the histogram of the input color values. Each element in
            the array corresponds to a particular bin in the histogram and contains
            the number of samples that fall into that bin. The bins are spaced
            evenly over the range of possible colors.
            		- `normalize`: A boolean value indicating whether the histogram
            should be normalized or not. If set to `True`, the histogram is
            normalized to have a sum of 1 across all bins. If set to `False`, the
            histogram is returned in its original, unnormalized form.
            		- `range`: A list of 3 integers representing the range of values
            over which the histogram was computed. This can be used to compute the
            bin boundaries and determine the number of bins used in the histogram.

        """
        color_hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)

        # Create a mask to exclude pixels with Saturation = 0 and Value = 256
        mask = (color_hsv[:, :, 1] > 0) & (color_hsv[:, :, 2] < 256)

        # Calculate histogram using the mask to exclude unwanted pixels
        hist = cv2.calcHist([color_hsv], [0, 1, 2], mask.astype(np.uint8), [8, 8, 8], [0, 256, 0, 256, 0, 256])

        return self.normalize_histogram(hist)

    def normalize_histogram(self, hist):
        """
        Computes the normalization of a histogram by dividing the histogram values
        by the total number of pixels in the histogram.

        Args:
            hist (ndarray (i.e., array-like object).): 2D histogram of pixel values
                in an image, which is then used to calculate the normalized histogram
                by dividing it by the total number of pixels in the image.
                
                		- `total_pixels`: The total number of pixels in the histogram.
                		- `hist`: The input histogram with values represented as integers
                between 0 and 255.

        Returns:
            array of arrays, with each inner array representing a normalized
            histogram for a particular feature map within the input image: a
            normalized histogram representation of the input image.
            
            		- `total_pixels`: The total number of pixels in the histogram. (Passed
            as an argument and not mentioned again.)
            		- `normalized_hist`: The normalized histogram, represented as a
            probability distribution where each value is a fraction between 0 and
            1.

        """
        total_pixels = np.sum(hist)
        normalized_hist = hist / total_pixels
        return normalized_hist

    def get_orientation_data(self, processed_image, original_image):
        """
        Processes a input image and extracts orientation data using non-maximum
        suppression and scales the resulting prediction to the original image size.

        Args:
            processed_image (ndarray (a multidimensional array object).): 4D numpy
                array containing the pre-processing results of the input image,
                which is passed through a deep learning model to generate predictions.
                
                	1/ `augment`: A boolean variable that represents whether to apply
                data augmentation on the image before performing non-maximum suppression.
                	2/ `scales`: An integer array representing the scales for non-maximum
                suppression.
                	3/ `roi_xsize`: An integer variable representing the ROI (Region
                of Interest) X size, which is used to scale the coordinates.
                	4/ `num_angles`: An integer variable representing the number of
                angles for non-maximum suppression.
                	5/ `original_image`: A Python object (possibly a tensor) representing
                the original image before any modifications were applied.
            original_image (2D numpy array.): 2D image from which the ROI is being
                extracted.
                
                		- `shape`: The shape of the original image, which is `(height,
                width)` in pixels.
                		- `roi_xsize`: The size of the ROI (Region of Interest) in pixels,
                which is a parameter passed to the `model` function.
                		- `num_angles`: The number of angles in the predicted orientations,
                which is also a parameter passed to the `non_max_suppression` function.

        Returns:
            dict: a tuple of two arrays: `orientation_bboxes` and `orientations`.

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
        Updates and prints various timing information every 1000 milliseconds,
        including the frame rate (`fps`), alive time, period, current period,
        increment, and thread period.

        Args:
            alive_time (int): time elapsed since the last measurement, which is
                used to calculate the current interval and determine whether the
                thread should sleep or wake up.
            period (int): period of time between frames, and it is used to calculate
                the current frame period, the difference between the current and
                previous frame periods, and the clipping value for the thread period.

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
        Tests various interfaces provided by the `ifaces` module. Specifically,
        it checks if `TImage`, `TDepth`, `TRGBD`, `TGroundTruth`, `KeyPoint`,
        `Person`, and `PeopleData` classes are defined and implement their respective
        methods correctly.

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
        Sets the chosen track and updates the tracking data structures based on
        the given ID.

        Args:
            track (int): 1D array of ID numbers that correspond to objects in the
                `objects_read` list for which tracking information is to be collected
                and processed by the function.

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
        Updates an existing plot by clearing the existing data, resampling the x
        and y values within a specified range, and then plotting the updated data
        using `matplotlib`.

        Args:
            frame (Pandas DataFrame.): 2D frame to be plotted on the axis.
                
                		- `self.ax`: The Axes object used for plotting the data.
                		- `self.xs`: The x-axis values of the frame.
                		- `self.ys`: The y-axis values of the frame.
                		- `self.id_list`: A list of integers representing the unique IDs
                of the data points in the frame.
                
                	The function clears the existing plot, limits the x and y lists
                to 20 items each, and then plots the data using a custom marker
                and linestyle. Additionally, it sets the x- and y-axis labels and
                adjusts the bottom position of the subplots.

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

        console.print(f"UPDATE EDGE: {fr} to {type}", type, style='green')

    def update_edge_att(self, fr: int, to: int, type: str, attribute_names: [str]):
        console.print(f"UPDATE EDGE ATT: {fr} to {type} {attribute_names}", style='green')

    def delete_edge(self, fr: int, to: int, type: str):
        console.print(f"DELETE EDGE: {fr} to {type} {type}", style='green')
