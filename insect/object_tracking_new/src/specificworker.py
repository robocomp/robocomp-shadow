#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2024 by YOUR NAME HERE
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

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
from collections import deque
import threading
import traceback
import numpy as np
import cv2
import time
import os
import glob
from ultralytics import YOLO
import torch
import math
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/ByteTrack')
from byte_tracker import BYTETracker

###################### JOINTBDOE IMPORTS ######################
sys.path.append('/home/robocomp/software/JointBDOE')
from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
import yaml

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)
class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 66
        self.thread_period = 66
        if startup_check:
            self.startup_check()
        else:
            # Hz
            self.cont = 0
            self.last_time = time.time()
            self.fps = 0

            # Camera height pose
            self.camera_height = 1340

            ############## CAMERA ##############
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
                    print(" ratio {:.2f}".format(self.rgb_original.width / self.rgb_original.height))

                    if self.rgb_original.width > 1920 or self.rgb_original.height > 960:
                        print("Camera resolution too high. Probably RGBD camera is giving wrong values. Trying again...")
                        time.sleep(2)

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
            self.tracker = BYTETracker(frame_rate=15)
            self.objects = ifaces.RoboCompVisualElementsPub.TData()

            ############## MODELS ##############
            # Load YOLO model
            self.load_v8_model()
            # Load JointBDOE model
            self.load_jointbdoe_model()

            ############## FOVEA ##############
            self.tracked_id = -1
            # Control gains
            self.k1 = 0.4
            self.k2 = 0.15
            # Last error between the actual and the required ROIs for derivative control
            self.last_ROI_error = 0

            ############## THREADS ##############
            # Event for stopping threads
            self.event = threading.Event()

            # RGBD data
            # Queue for storing RGBD data
            self.RGBD_queue = deque(maxlen=1)
            # Thread for obtaining RGBD data
            self.RGBD_thread = threading.Thread(target=self.getRGBD, args=[self.event], daemon=True)
            self.RGBD_thread.start()
            time.sleep(2)

            # Objects data
            # Queue for storing objects data
            self.objects_queue = deque(maxlen=1)
            # Thread for obtaining objects data
            self.objects_thread = threading.Thread(target=self.getObjects, args=[self.event], daemon=True)
            self.objects_thread.start()
            time.sleep(2)

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        self.RGBD_thread.join()
        self.objects_thread.join()

    def setParams(self, params):
        try:
            self.display = params["display"] == "true" or params["display"] == "True"
        except:
            print("Error reading config params")
            traceback.print_exc()

###################### THREADS ######################
    # Method for obtaining RGBD data
    def getRGBD(self, event: threading.Event):
        while not event.is_set():
            try:
                # start = time.time()
                # Get RGBD data
                roi_data = {"roi_xcenter": self.roi_xcenter, "roi_ycenter": self.roi_ycenter,
                            "roi_xsize": self.roi_xsize, "roi_ysize": self.roi_ysize}
                rgbd = self.camera360rgbd_proxy.getROI(
                    roi_data["roi_xcenter"], roi_data["roi_ycenter"], roi_data["roi_xsize"],
                    roi_data["roi_ysize"], roi_data["roi_xsize"], roi_data["roi_ysize"]
                )
                color = np.frombuffer(rgbd.rgb, dtype=np.uint8).reshape(rgbd.height, rgbd.width, 3)
                depth = np.frombuffer(rgbd.depth, dtype=np.float32).reshape(rgbd.height, rgbd.width, 3)
                # Convert color to JointBDOE format
                color_orientation = self.convert_image_to_jointbdoe(color)
                # Put RGBD data in the queue
                self.RGBD_queue.append((color, color_orientation, depth, rgbd.period, rgbd.alivetime, roi_data))
                # If the current ROI is not the target ROI, call the method to move towards the target ROI.
                if (
                        roi_data["roi_xcenter"] != self.target_roi_xcenter or
                        roi_data["roi_ycenter"] != self.target_roi_ycenter or
                        roi_data["roi_xsize"] != self.target_roi_xsize or
                        roi_data["roi_ysize"] != self.target_roi_xsize
                ):
                    self.from_act_roi_to_target()
                event.wait(self.thread_period / 1000)
                # print("RGBD time:", time.time() - start, "ms")
            except Ice.Exception as e:
                traceback.print_exc()
                print(e)
    # Method for converting img to JointBDOE format
    def convert_image_to_jointbdoe(self, color):
        img_ori = letterbox(color, 640, stride=self.stride, auto=True)[0]
        img_ori = img_ori.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_ori = np.ascontiguousarray(img_ori)
        img_ori = torch.from_numpy(img_ori).to(self.device)
        img_ori = img_ori / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img_ori.shape) == 3:
            img_ori = img_ori[None]  # expand for batch dim
        return img_ori
    def from_act_roi_to_target(self):
        """
            This method modifies the actual Region of Interest (ROI) to approach the target ROI. It calculates the differences
            between the actual and target ROIs and then adjusts the actual ROI based on these differences. The adjustment is
            performed in both the x and y dimensions, and both for size and center location.

            The method also updates the last ROI error value to be used in future calculations.
            """
        # Get errors from ROI sizes and centers
        x_diff = abs(self.target_roi_xcenter - self.roi_xcenter)
        y_diff = abs(self.target_roi_ycenter - self.roi_ycenter)
        x_size_diff = abs(self.target_roi_xsize - self.roi_xsize)
        y_size_diff = abs(self.target_roi_ysize - self.roi_ysize)

        # aux_x_diff defined for setting pixel speed and calculate derivated component
        aux_x_diff = x_diff if x_diff < self.rgb_original.width / 2 else self.rgb_original.width - x_diff
        x_der_diff = int(self.k2 * abs(self.last_ROI_error - aux_x_diff))

        x_mod_speed = np.clip(int(self.k1 * aux_x_diff), 0, 22) + x_der_diff
        y_mod_speed = np.clip(int(self.k1 * y_diff), 0, 20)

        x_size_mod_speed = np.clip(int(0.03 * x_size_diff), 0, 8)
        y_size_mod_speed = np.clip(int(0.03 * y_size_diff), 0, 8)

        if self.roi_xcenter < self.target_roi_xcenter:
            self.roi_xcenter -= x_mod_speed if x_diff > self.rgb_original.width / 2 else -x_mod_speed
        elif self.roi_xcenter > self.target_roi_xcenter:
            self.roi_xcenter += x_mod_speed if x_diff > self.rgb_original.width / 2 else -x_mod_speed

        self.roi_xcenter %= self.rgb_original.width

        if self.roi_ycenter < self.target_roi_ycenter:
            self.roi_ycenter += y_mod_speed
        elif self.roi_ycenter > self.target_roi_ycenter:
            self.roi_ycenter -= y_mod_speed

        if self.roi_xsize < self.target_roi_xsize:
            self.roi_xsize += x_size_mod_speed
        elif self.roi_xsize > self.target_roi_xsize:
            self.roi_xsize -= x_size_mod_speed

        if self.roi_ysize < self.target_roi_ysize:
            self.roi_ysize += y_size_mod_speed
        elif self.roi_ysize > self.target_roi_ysize:
            self.roi_ysize -= y_size_mod_speed

        self.last_ROI_error = aux_x_diff
    # Method for obtaining objects data
    def getObjects(self, event: threading.Event):
        while not event.is_set():
            # Get RGBD queue data
            if self.RGBD_queue:
                # start = time.time()
                color, color_orientation, depth, period, alive_time, roi_data = self.RGBD_queue.pop()
                out_v8 = self.v8_model.predict(color, show_conf=True)
                orientation_bboxes, orientations = self.get_orientation_data(color_orientation, color)
                self.objects_queue.append((out_v8, orientation_bboxes, orientations, color, depth, period, alive_time, roi_data))
                event.wait(self.thread_period / 1000)
                # print("Inference time:", time.time() - start, "ms")
    # Method for obtaining orientation data
    def get_orientation_data(self, processed_image, original_image):
        out_ori = self.model(processed_image, augment=True, scales=[self.roi_xsize / 640])[0]
        out = non_max_suppression(out_ori, 0.3, 0.5, num_angles=self.data['num_angles'])
        orientation_bboxes = scale_coords(processed_image.shape[2:], out[0][:, :4], original_image.shape[:2]).cpu().numpy().astype(int)  # native-space pred
        orientations = (out[0][:, 6:].cpu().numpy() * 360) - 180   # N*1, (0,1)*360 --> (0,360)
        return orientation_bboxes, orientations
    @QtCore.Slot()
    def compute(self):
        if self.event.is_set():
            self.RGBD_thread.join()
            self.objects_thread.join()
        # Get objects data in different variables
        if self.objects_queue:
            # start = time.time()
            out_v8, orientation_bboxes, orientations, color, depth, period, alive_time, roi_data = self.objects_queue.pop()
            people, objects = self.get_segmentator_data(out_v8, color, depth)
            people = self.associate_orientation_with_segmentation(people, orientation_bboxes, orientations)
            for key in objects:
                objects[key].extend(people[key])
            tracks = self.tracker.update(np.array(objects["confidences"]),
                                np.array(objects["bboxes"]),
                                np.array(objects["classes"]),
                                np.array(objects["poses"]),
                                np.array(objects["orientations"]))
            self.objects = self.to_visualelements_interface(tracks, alive_time, roi_data)

            self.visualelementspub_proxy.setVisualObjects(self.objects)
            if self.tracked_id != -1:
                self.set_roi_dimensions(self.objects, roi_data)
            # print("Postprocess time:", time.time() - start, "ms")
            # cv2.imshow("color", color)
            # cv2.waitKey(1)
            self.show_fps(alive_time, period)
    ###################### MODELS LOADING METHODS ######################
    def load_v8_model(self):
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
        except:
            print("Error loading JointBDOE model")
            exit(1)
    ###################### SEGMENTATION METHODS ######################
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
                        if element_confidence > 0.2:
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
    def get_mask_with_modified_background(self, mask, image):
        """
        A method that removes the background of person ROI considering the segmentation mask and fill it with white color

        Args:
            mask: black and white mask obtained with the segmentation
            image: RGB image corresponding to person ROI
        """
        masked_image = mask * image
        is_black_pixel = np.logical_and(masked_image[:, :, 0] == 0, masked_image[:, :, 1] == 0,
                                        masked_image[:, :, 2] == 0)
        masked_image[is_black_pixel] = [255, 255, 255]

        return masked_image
    def get_color_histogram(self, color):
        # Convert the color image to HSV
        color_hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
        # Create a mask to exclude pixels with Saturation = 0 and Value = 256
        mask = (color_hsv[:, :, 1] > 0) & (color_hsv[:, :, 2] < 256)
        # Calculate histogram using the mask to exclude unwanted pixels
        hist = cv2.calcHist([color_hsv], [0, 1, 2], mask.astype(np.uint8), [8, 8, 8], [0, 256, 0, 256, 0, 256])
        # Normalize the histogram
        total_pixels = np.sum(hist)
        normalized_hist = hist / total_pixels
        return normalized_hist
    def get_mask_distance(self, mask, depth_image):
        # Get bbox center point
        # Get depth image shape and calculate bbox center
        depth_image_shape = depth_image.shape
        bbox = [0, 0, depth_image_shape[1], depth_image_shape[0]]
        bbox_center = [depth_image_shape[1] // 2, depth_image_shape[0] // 2]
        segmentation_points = np.argwhere(np.all(mask == 1, axis=-1))[:, [1, 0]]
        # p = bbox width / 4
        p = depth_image_shape[1] // 4
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
    ###################### VE METHODS ######################
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
            x_pose = round(track.mean[0], 2)
            y_pose = round(track.mean[1], 2)
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
            # print("generic_attrs", generic_attrs)
            # object_ = ifaces.RoboCompVisualElementsPub.TObject(id=int(track.track_id), type=track.clase, attributes=generic_attrs, image=self.mask_to_TImage(track.image, roi))
            object_ = ifaces.RoboCompVisualElementsPub.TObject(id=int(track.track_id), type=track.clase,
                                                               attributes=generic_attrs)
            objects.append(object_)
        # print("IMAGE TIMESTAMP", image_timestamp)
        return ifaces.RoboCompVisualElementsPub.TData(timestampimage=image_timestamp, timestampgenerated=int(time.time() * 1000), period=self.Period, objects=objects)
    def mask_to_TImage(self, mask, roi):
        y, x, _ = mask.shape
        roi = ifaces.RoboCompCamera360RGB.TRoi(xcenter=roi["roi_xcenter"], ycenter=roi["roi_ycenter"],
                                                    xsize=roi["roi_xsize"], ysize=roi["roi_ysize"],
                                                    finalxsize=roi["roi_xsize"],
                                                    finalysize=roi["roi_ysize"])
        return ifaces.RoboCompCamera360RGB.TImage(image=mask.tobytes(), height=y, width=x, roi=roi)

    # Calculate image ROI for element centering
    def set_roi_dimensions(self, objects, roi):
        """
            Set Region of Interest (ROI) based on objects.

            Args:
                objects (list): List of detected objects. Each object contains information about its position and size.

            The method goes through the list of objects and when it finds the object that matches the tracked_id,
            it calculates the desired ROI based on the object's position and size. The ROI is then stored in the class's attributes.
            """
        for object in objects.objects:
            if object.id == self.tracked_id:
                # roi = object.image.roi
                # x_roi_offset = roi.xcenter - roi.xsize / 2
                # y_roi_offset = roi.ycenter - roi.ysize / 2
                # x_factor = roi.xsize / roi.finalxsize
                # y_factor = roi.ysize / roi.finalysize

                x_roi_offset = roi["roi_xcenter"] - roi["roi_xsize"] / 2
                y_roi_offset = roi["roi_ycenter"] - roi["roi_ysize"] / 2
                x_factor = 1
                y_factor = 1


                left = int(float(object.attributes["bbox_left"]) * x_factor + x_roi_offset)
                right = int(float(object.attributes["bbox_right"]) * x_factor + x_roi_offset)

                top = int(float(object.attributes["bbox_top"]) * y_factor + y_roi_offset)
                bot = int(float(object.attributes["bbox_bot"]) * y_factor + y_roi_offset)

                self.target_roi_xcenter = (left + (right - left) / 2) % self.rgb_original.width
                self.target_roi_ycenter = (top + (bot - top) / 2)
                self.target_roi_ysize = np.clip(int((bot - top) * 2), 0, self.rgb_original.height)
                self.target_roi_xsize = self.target_roi_ysize
                return
        self.tracked_id = -1
        self.target_roi_xcenter = self.rgb_original.width // 2
        self.target_roi_ycenter = self.rgb_original.height // 2
        self.target_roi_xsize = self.rgb_original.width // 2
        self.target_roi_ysize = self.rgb_original.height
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
    def startup_check(self):
        print(f"Testing RoboCompCamera360RGBD.TRoi from ifaces.RoboCompCamera360RGBD")
        test = ifaces.RoboCompCamera360RGBD.TRoi()
        print(f"Testing RoboCompCamera360RGBD.TRGBD from ifaces.RoboCompCamera360RGBD")
        test = ifaces.RoboCompCamera360RGBD.TRGBD()
        print(f"Testing RoboCompVisualElementsPub.TObject from ifaces.RoboCompVisualElementsPub")
        test = ifaces.RoboCompVisualElementsPub.TObject()
        print(f"Testing RoboCompVisualElementsPub.TData from ifaces.RoboCompVisualElementsPub")
        test = ifaces.RoboCompVisualElementsPub.TData()
        QTimer.singleShot(200, QApplication.instance().quit)


    # =============== Methods for Component SubscribesTo ================
    # ===================================================================

    #
    # SUBSCRIPTION to setTrack method from SegmentatorTrackingPub interface
    #
    def SegmentatorTrackingPub_setTrack(self, target):
        if target.id == -1:
            self.tracked_id = -1
            self.target_roi_xcenter = self.rgb_original.width // 2
            self.target_roi_ycenter = self.rgb_original.height // 2
            self.target_roi_xsize = self.rgb_original.width // 2
            self.target_roi_ysize = self.rgb_original.height
            return
        self.tracked_id = target.id


    # ===================================================================
    # ===================================================================



    ######################
    # From the RoboCompCamera360RGBD you can call this methods:
    # self.camera360rgbd_proxy.getROI(...)

    ######################
    # From the RoboCompCamera360RGBD you can use this types:
    # RoboCompCamera360RGBD.TRoi
    # RoboCompCamera360RGBD.TRGBD

    ######################
    # From the RoboCompVisualElementsPub you can publish calling this methods:
    # self.visualelementspub_proxy.setVisualObjects(...)

    ######################
    # From the RoboCompVisualElementsPub you can use this types:
    # RoboCompVisualElementsPub.TObject
    # RoboCompVisualElementsPub.TData


