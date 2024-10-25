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

from PySide6.QtCore import Qt
# from PySide6.QtCore import QTimer
# from PySide6.QtWidgets import QApplication
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

from ultralytics import YOLO
import torch
import itertools
import math

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
        super(SpecificWorker, self).__init__(proxy_map)

        if startup_check:
            self.startup_check()
        else:
            self.Period = 50
            self.thread_period = 50
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

            # lidar_odometry_data = self.lidarodometry_proxy.getPoseAndChange()
            # # Generate numpy transform matrix with the lidar odometry data
            # self.last_transform_matrix = np.array(
            #     [[lidar_odometry_data.pose.m00, lidar_odometry_data.pose.m01, lidar_odometry_data.pose.m02, lidar_odometry_data.pose.m03],
            #      [lidar_odometry_data.pose.m10, lidar_odometry_data.pose.m11, lidar_odometry_data.pose.m12, lidar_odometry_data.pose.m13],
            #      [lidar_odometry_data.pose.m20, lidar_odometry_data.pose.m21, lidar_odometry_data.pose.m22, lidar_odometry_data.pose.m23],
            #      [lidar_odometry_data.pose.m30, lidar_odometry_data.pose.m31, lidar_odometry_data.pose.m32, lidar_odometry_data.pose.m33]])

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
        # if self.reset:
        #     print("RESETTING TRACKS")
        #     self.tracker.tracked_stracks = []  # type: list[STrack]
        #     self.tracker.lost_stracks = []  # type: list[STrack]
        #     self.tracker.removed_stracks = []  # type: list[STrack]
        #     self.tracker.frame_id = 0
        #     time.sleep(2)
        #     stop_node = Node(agent_id=self.agent_id, type='intention', name="STOP")
        #     try:
        #         id_result = self.g.insert_node(stop_node)
        #         console.print('Person mind node created -- ', id_result, style='red')
        #         has_edge = Edge(id_result, self.g.get_node('Shadow').id, 'has', self.agent_id)
        #         self.g.insert_or_assign_edge(has_edge)
        #
        #         print(' inserted new node  ', id_result)
        #
        #     except:
        #         traceback.print_exc()
        #         print('cant update node or add edge RT')
        #     time.sleep(2)
        #     self.reset = False

        if self.inference_read_queue:
            start = time.time()
            out_v8_front, color_front, orientation_bboxes_front, orientations_front, depth_front, delta, alive_time, period, front_roi = self.inference_read_queue.pop()
            people_front, objects_front = self.get_segmentator_data(out_v8_front, color_front, depth_front)
            people_front = self.associate_orientation_with_segmentation(people_front, orientation_bboxes_front, orientations_front)

            # print("People front", people_front)
            # print("Objects front", objects_front)
            # seg_img = out_v8_front[0].plot()
            # cv2.imshow("seg_img", seg_img)
            # cv2.waitKey(1)

            # Fuse people and objects and equal it to self.objects_write
            self.objects = self.to_visualelements_interface(people_front, objects_front, alive_time)

            # front_objects = self.to_visualelements_interface(tracks, alive_time, front_roi)
            #
            # # Fuse front_objects and back_objects and equal it to self.objects_write
            self.visualelementspub_proxy.setVisualObjects(self.objects)
            # # If display is enabled, show the tracking results on the image
            #
            if self.display:
                img_front = self.display_data_tracks(color_front, self.objects.objects)
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
        inversa_transformacion1 = np.linalg.inv(transformacion1)

        # Calcular la matriz diferencial multiplicando la inversa de la primera
        # matriz de transformación por la segunda matriz de transformación
        matriz_diferencial = np.dot(inversa_transformacion1, transformacion2)
        # Trasponer la matriz diferencial
        matriz_diferencial[:3, 3] *= 1000
        return matriz_diferencial

    ###################### MODELS LOADING METHODS ######################
    def load_v8_model(self):
        pattern = os.path.join(os.getcwd(), "yolo11?-seg.pt")
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
            self.download_and_convert_v8_model("yolo11" + model_name + "-seg")
    def download_and_convert_v8_model(self, model_name):
        # Download model
        print("Downloading model:", model_name)
        self.v8_model = YOLO(model_name)
        # Export the model to TRT
        self.v8_model.export(format='engine', device='0')
        self.v8_model = YOLO(model_name + '.engine')
    def load_jointbdoe_model(self):
        try:
            self.device = select_device("0", batch_size=1)
            self.model = attempt_load(
                "/home/robocomp/software/JointBDOE/runs/JointBDOE/coco_s_1024_e500_t020_w005/weights/best.pt",
                map_location=self.device)
            self.stride = int(self.model.stride.max())
            with open("/home/robocomp/software/JointBDOE/data/JointBDOE_weaklabel_coco.yaml") as f:
                self.data = yaml.safe_load(f)  # load data dict
        except Exception as e:
            print("Error loading JointBDOE model", e)

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

                    image_front = self.camera360rgbd_proxy.getROI(960, 480, 960, 960, 960, 960)
                    roi_data_front = ifaces.RoboCompCamera360RGB.TRoi(xcenter=image_front.roi.xcenter, ycenter=image_front.roi.ycenter, xsize=image_front.roi.xsize, ysize=image_front.roi.ysize, finalxsize=image_front.roi.finalxsize, finalysize=image_front.roi.finalysize)

                    color_front = np.frombuffer(image_front.rgb, dtype=np.uint8).reshape(image_front.height, image_front.width, 3)
                    depth_front = np.frombuffer(image_front.depth, dtype=np.float32).reshape(image_front.height, image_front.width, 3)

                    # Process image for orientation DNN
                    front_img_tensor = self.convert_image_to_tensor(color_front)

                    # Calculate time difference.
                    delta = int(1000 * time.time() - image_front.alivetime)
                    data_package = [color_front, front_img_tensor, depth_front, delta, image_front.period, image_front.alivetime, roi_data_front]

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
        img_ori = letterbox(image, 640, stride=self.stride, auto=True)[0]
        img_ori = img_ori.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_ori = np.ascontiguousarray(img_ori)
        img_ori = torch.from_numpy(img_ori).to(self.device)
        img_ori = img_ori / 255.0  # 0 - 255 to 0.0 - 1.0

        if len(img_ori.shape) == 3:
            img_ori = img_ori[None]  # expand for batch dim
        return img_ori

    def inference_thread(self, event: Event):
        while not event.is_set():
            if self.rgb_read_queue:
                start = time.time()
                color_front, front_img_tensor, depth_front, delta, period, alive_time, front_roi = self.rgb_read_queue.pop()
                out_v8_front, orientation_bboxes_front, orientations_front = self.inference_over_image(color_front, front_img_tensor)
                self.inference_read_queue.append([out_v8_front, color_front, orientation_bboxes_front, orientations_front, depth_front, delta, alive_time, period, front_roi])
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
        orientation_bboxes, orientations = self.get_orientation_data(img_ori, img0)
        out_v8 = self.v8_model.predict(img0, show_conf=True)
        return out_v8, orientation_bboxes, orientations

    def to_visualelements_interface(self, people, objects, image_timestamp):
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
        object_counter = 0
        total_objects = []
        for i in range(len(people["bboxes"])):
            generic_attrs = {
                "score": str(people["confidences"][i]),
                "bbox_left": str(int(people["bboxes"][i][0])),
                "bbox_top": str(int(people["bboxes"][i][1])),
                "bbox_right": str(int(people["bboxes"][i][2])),
                "bbox_bot": str(int(people["bboxes"][i][3])),
                "x_pos": str(round(people["poses"][i][0], 2)),
                "y_pos": str(round(people["poses"][i][1], 2)),
                "z_pos": str(round(people["poses"][i][2], 2)),
                "orientation": str(round(float(people["orientations"][i]), 2))
            }

            mask_points = ifaces.RoboCompLidar3D.TDataImage(XArray=people["masks"][i][:, 0].tolist(),
                                                            YArray=people["masks"][i][:, 1].tolist(),
                                                            ZArray=people["masks"][i][:, 2].tolist())
            object_ = ifaces.RoboCompVisualElementsPub.TObject(id=object_counter, type=people["classes"][i],
                                                               attributes=generic_attrs, maskpoints=mask_points)
            total_objects.append(object_)
            object_counter += 1
        for i in range(len(objects["bboxes"])):
            generic_attrs = {
                "score": str(objects["confidences"][i]),
                "bbox_left": str(int(objects["bboxes"][i][0])),
                "bbox_top": str(int(objects["bboxes"][i][1])),
                "bbox_right": str(int(objects["bboxes"][i][2])),
                "bbox_bot": str(int(objects["bboxes"][i][3])),
                "x_pos": str(round(objects["poses"][i][0], 2)),
                "y_pos": str(round(objects["poses"][i][1], 2)),
                "z_pos": str(round(objects["poses"][i][2], 2)),
                "orientation": str(round(float(objects["orientations"][i]), 2))
            }
            mask_points = ifaces.RoboCompLidar3D.TDataImage(XArray=objects["masks"][i][:, 0].tolist(),
                                                            YArray=objects["masks"][i][:, 1].tolist(),
                                                            ZArray=objects["masks"][i][:, 2].tolist())
            #print("MASK POINTS", mask_points.XArray)
            # object_ = ifaces.RoboCompVisualElementsPub.TObject(id=int(track.track_id), type=track.clase, attributes=generic_attrs, image=self.mask_to_TImage(track.image, roi))
            object_ = ifaces.RoboCompVisualElementsPub.TObject(id=object_counter, type=objects["classes"][i],
                                                               attributes=generic_attrs, maskpoints=mask_points)
            total_objects.append(object_)
            object_counter += 1
        # print("IMAGE TIMESTAMP", image_timestamp)

        return ifaces.RoboCompVisualElementsPub.TData(timestampimage=image_timestamp, timestampgenerated=int(time.time() * 1000), period=self.Period, objects=total_objects)

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

                        if element_confidence > 0.4:
                            element_class = boxes[i].cls.cpu().numpy().astype(int)[0]
                            element_bbox = boxes[i].xyxy.cpu().numpy().astype(int)[0]
                            image_mask = np.zeros((roi_ysize, roi_xsize, 1), dtype=np.uint8)
                            act_mask = masks[i].astype(np.int32)
                            cv2.fillConvexPoly(image_mask, act_mask, (1, 1, 1))
                            image_mask_element = image_mask[element_bbox[1]:element_bbox[3],
                                                 element_bbox[0]:element_bbox[2]]
                            color_image_mask = color_image[element_bbox[1]:element_bbox[3],
                                               element_bbox[0]:element_bbox[2]]
                            # element_mask = self.get_mask_with_modified_background(image_mask_element, color_image_mask)
                            # element_hash = self.get_color_histogram(element_mask)
                            height, width, _ = image_mask_element.shape
                            depth_image_mask = depth_image[element_bbox[1]:element_bbox[3],
                                               element_bbox[0]:element_bbox[2]]
                            element_pose, filtered_depth_mask = self.get_mask_distance(image_mask_element, depth_image_mask)
                            if element_pose != [0, 0, 0]:
                                if element_class == 0:
                                    people["poses"].append(element_pose)
                                    people["bboxes"].append(element_bbox)
                                    people["confidences"].append(element_confidence)
                                    people["masks"].append(filtered_depth_mask)
                                    people["classes"].append(element_class)
                                    # people["hashes"].append(element_hash)
                                else:
                                    objects["bboxes"].append(element_bbox)
                                    objects["poses"].append(element_pose)
                                    objects["confidences"].append(element_confidence)
                                    objects["masks"].append(filtered_depth_mask)
                                    objects["classes"].append(element_class)
                                    # objects["hashes"].append(element_hash)

        people["orientations"] = [-4] * len(people["bboxes"])
        objects["orientations"] = [-4] * len(objects["bboxes"])
        return people, objects

    def get_mask_distance(self, mask, depth_image):
        # Get bbox center point
        # Get depth image shape and calculate bbox center
        depth_image_shape = depth_image.shape
        bbox_center = [depth_image_shape[1] // 2, depth_image_shape[0] // 2]
        segmentation_points = np.argwhere(np.all(mask == 1, axis=-1))[:, [1, 0]]
        total_depth_mask = depth_image[segmentation_points[:, 1], segmentation_points[:, 0]]
        total_depth_mask = total_depth_mask[np.any(total_depth_mask != [0, 0, 0], axis=-1)]

        p = depth_image_shape[1] // 4.5
        segmentation_points = segmentation_points[np.linalg.norm(segmentation_points - bbox_center, axis=1) < p]
        # Extraer directamente los puntos válidos (no ceros) y sus distancias en un solo paso.
        # Esto evitará tener que crear y almacenar matrices temporales y booleanas innecesarias.
        valid_points = depth_image[segmentation_points[:, 1], segmentation_points[:, 0]]
        valid_points = valid_points[np.any(valid_points != [0, 0, 0], axis=-1)]
        # print valid points
        if valid_points.size == 0:
            return [0, 0, 0], total_depth_mask
        distances = np.linalg.norm(valid_points, axis=-1)
        # Parece que intentas generar bins para un histograma basado en un rango de distancias.
        # Crear los intervalos directamente con np.histogram puede ser más eficiente.
        hist, edges = np.histogram(distances, bins=np.arange(np.min(distances), np.max(distances), 300))
        if hist.size == 0:
            pose = np.mean(valid_points, axis=0).tolist()
            return pose, total_depth_mask
        # Identificar el intervalo de moda y extraer los índices de las distancias que caen dentro de este intervalo.
        max_index = np.argmax(hist)
        pos_filtered_distances = np.logical_and(distances >= edges[max_index], distances <= edges[max_index + 1])
        # Filtrar los puntos válidos y calcular la media en un solo paso.
        person_dist = np.mean(valid_points[pos_filtered_distances], axis=0).tolist()
        return person_dist, total_depth_mask

    # def get_color_histogram(self, color):
    #     color =cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    #     hist = cv2.calcHist([color], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    #     return self.normalize_histogram(hist)

    #GPT VERSION: white pixels deleted
    def get_color_histogram(self, color):
        # Convert the color image to HSV
        color_hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)

        # Create a mask to exclude pixels with Saturation = 0 and Value = 256
        mask = (color_hsv[:, :, 1] > 0) & (color_hsv[:, :, 2] < 256)

        # Calculate histogram using the mask to exclude unwanted pixels
        hist = cv2.calcHist([color_hsv], [0, 1, 2], mask.astype(np.uint8), [8, 8, 8], [0, 256, 0, 256, 0, 256])

        return self.normalize_histogram(hist)

    def normalize_histogram(self, hist):
        total_pixels = np.sum(hist)
        normalized_hist = hist / total_pixels
        return normalized_hist

    def get_orientation_data(self, processed_image, original_image):
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

        if elements is None:
            return img
        img = img.astype(np.uint8)
        for element in elements:
            x0, y0, x1, y1 = map(int, [int(float(element.attributes["bbox_left"])), int(float(element.attributes["bbox_top"])), int(float(element.attributes["bbox_right"])), int(float(element.attributes["bbox_bot"]))])
            cls_ind = element.type
            color = (_COLORS[cls_ind] * 255).astype(np.uint8).tolist()
            # text = f'Class: {class_names[cls_ind]} - Score: {element.score * 100:.1f}% - ID: {element.id}'
            text = f'{float(element.attributes["x_pos"])} - {float(element.attributes["y_pos"])} - {float(element.attributes["z_pos"])} - {element.id} - {float(element.attributes["orientation"])} - {_OBJECT_NAMES[element.type]}'
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_ind]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]

            # Check if the img array is read-only
            if not img.flags.writeable:
                # Create a writable copy of the img array
                img = img.copy()

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

