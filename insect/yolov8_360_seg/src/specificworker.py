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
import copy

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import numpy as np
import time
import cv2
from threading import Thread, Event
import traceback
import queue
import sys
import yaml
import copy

sys.path.append('/home/robocomp/software/JointBDOE')

from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

sys.path.append('/home/robocomp/robocomp/components/robocomp-shadow/insect/hash_tracker/HashTrack')
import matching

from ultralytics import YOLO
import torch


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

_ROBOLAB_NAMES = ['person', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'bottle', 'wine glass', 'cup',
                  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                  'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                  'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                  'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                  'toothbrush']

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
            self.Period = 1
            self.thread_period = 10
            self.display = False
            self.yolo_model_name = 'yolov8m-seg.pt'
            # self.yolo_model_name = 'yolov8n-pose.pt'

            self.model_v8 = YOLO(self.yolo_model_name)

            self.device = select_device("0", batch_size=1)
            self.model = attempt_load(
                "/home/robocomp/software/JointBDOE/runs/JointBDOE/coco_s_1024_e500_t020_w005/weights/best.pt",
                map_location=self.device)
            self.stride = int(self.model.stride.max())
            with open("/home/robocomp/software/JointBDOE/data/JointBDOE_weaklabel_coco.yaml") as f:
                self.data = yaml.safe_load(f)  # load data dict

            # OJO, comentado en el main
            self.setParams(params)

            # read test image to get sizes
            started_camera = False
            while not started_camera:
                try:
                    self.rgb_original = self.camera360rgb_proxy.getROI(-1, -1, -1, -1, -1, -1)

                    print("Camera specs:")
                    print(" width:", self.rgb_original.width)
                    print(" height:", self.rgb_original.height)
                    print(" depth", self.rgb_original.depth)
                    print(" focalx", self.rgb_original.focalx)
                    print(" focaly", self.rgb_original.focaly)
                    print(" period", self.rgb_original.period)
                    print(" ratio {:.2f}".format(self.rgb_original.width/self.rgb_original.height))

                    # Image ROI require parameters
                    self.final_xsize = 640
                    self.final_ysize = 640
                    self.roi_xsize = self.rgb_original.width // 2
                    self.roi_ysize = self.rgb_original.height
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
                    print(e, "Trying again...")
                    time.sleep(2)

            self.window_name = "Yolo Segmentator"

            print("Loaded YOLO model: ", self.yolo_model_name)

            # Hz
            self.cont = 0
            self.last_time = time.time()
            self.fps = 0

            # interface swap objects
            self.objects_read = []
            self.objects_write = []

            # ID to track
            self.tracked_element = None
            self.tracked_id = None

            # Last error between the actual and the required ROIs for derivative control
            self.last_ROI_error = 0

            # Control gains
            self.k1 = 0.4
            self.k2 = 0.15

            # camera read thread
            self.rgb = None
            self.read_queue = queue.Queue(1)
            self.event = Event()
            self.read_thread = Thread(target=self.get_rgb_thread, args=["camera_top", self.event],
                                      name="read_queue", daemon=True)
            self.read_thread.start()

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        try:
            self.classes = [0]
            self.yolo_model = params["yolo_model"]
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
        # Get image with depth
        data = self.read_queue.get()
        if self.depth_flag:
            rgb, depth, alive_time, period = data
        else:
            rgb, alive_time, period = data

        img0 = copy.deepcopy(rgb)

        init = time.time()
        bboxes, confidences, associated_orientations, masks, classes = self.inference_over_image(rgb)

        print("TIEMPO INFERENCIA:", time.time() - init)

        # Set ROI dimensions if tracking an object
        if self.tracked_id is not None:
            self.set_roi_dimensions(self.objects_read)
        #
        self.objects_write = ifaces.RoboCompVisualElements.TObjects()

        masks = self.get_mask_with_modified_background(masks, bboxes, img0)

        self.objects_write = self.create_interface_data(bboxes, confidences, classes, masks, associated_orientations)
        self.objects_write = self.visualelements_proxy.getVisualObjects(self.objects_write)
        #
        # # swap
        self.objects_write, self.objects_read = self.objects_read, self.objects_write

        # If display is enabled, show the tracking results on the image
        if self.display:
            img = self.display_data_tracks(rgb, self.objects_read)
            cv2.imshow(self.window_name, img)
            cv2.waitKey(1)


        # Show FPS and handle Keyboard Interruption
        try:
            self.show_fps(alive_time, period)
        except KeyboardInterrupt:
            self.event.set()

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
                # Get ROI from the camera.
                self.rgb = self.camera360rgb_proxy.getROI(
                    self.roi_xcenter, self.roi_ycenter, self.roi_xsize,
                    self.roi_ysize, self.final_xsize, self.final_ysize
                )

                # Convert image data to numpy array and reshape it.
                color = np.frombuffer(self.rgb.image, dtype=np.uint8).reshape(self.rgb.height, self.rgb.width, 3)

                # Calculate time difference.
                delta = int(1000 * time.time() - self.rgb.alivetime)

                # Prepare the data package to be put into the queue.
                data_package = [color, delta, self.rgb.period]

                # If the depth flag is set, insert the depth information into the data package.
                if self.depth_flag:
                    data_package.insert(1, self.rgb.depth)

                # Put the data package into the queue.
                self.read_queue.put(data_package)

                # If the current ROI is not the target ROI, call the method to move towards the target ROI.
                if (
                        self.roi_xcenter != self.target_roi_xcenter or
                        self.roi_ycenter != self.target_roi_ycenter or
                        self.roi_xsize != self.target_roi_xsize or
                        self.roi_ysize != self.target_roi_xsize
                ):
                    self.from_act_roi_to_target()

                event.wait(self.thread_period / 1000)
            except Ice.Exception as e:
                traceback.print_exc()
                print(e, "Error communicating with Camera360RGB")

    def get_mask_with_modified_background(self, masks, bboxes, image):
        output_masks = []
        if len(masks) == len(bboxes):
            for i in range(len(masks)):
                image_mask = np.zeros((self.rgb.height, self.rgb.width, 3), dtype=np.uint8)
                act_mask = masks[i].astype(np.int32)
                act_bbox = bboxes[i]
                cv2.fillConvexPoly(image_mask, act_mask, (1, 1, 1))
                masked_image = image_mask * image
                roi = masked_image[act_bbox[1]:act_bbox[3], act_bbox[0]:act_bbox[2]]
                h, w, _ = roi.shape
                background_color = roi[int(h / 3), int(w / 2)]
                black_pixels = np.where(np.all(roi == [0, 0, 0], axis=-1))
                roi[black_pixels] = background_color
                output_masks.append(roi)
        return output_masks


    # Modify actual ROI data to converge in the target ROI dimensions
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

        # print("")
        # print("x_diff", x_diff, "target_roi_xcenter", self.target_roi_xcenter, "roi_xcenter", self.roi_xcenter)
        # print("self.last_ROI_error", self.last_ROI_error, "x_der_diff", x_der_diff)

        x_mod_speed = np.clip(int(self.k1 * aux_x_diff), 0, 22) + x_der_diff
        y_mod_speed = np.clip(int(self.k1 * y_diff), 0, 20)

        # print("x_mod_speed", x_mod_speed)

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

    def inference_over_image(self, img):
        img0 = copy.deepcopy(img)
        img = letterbox(img, 640, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0

        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Make inference with both models
        out_ori = self.model(img, augment=True, scales=[self.roi_xsize / 640])[0]
        orientation_bboxes, orientations = self.get_orientation_data(out_ori, img, img0)

        out_v8 = self.model_v8.predict(img0, classes=0, show_conf=True)

        # YOLO V8 data processing
        if "pose" in self.yolo_model_name:
            bboxes, confidences, skeletons, classes = self.get_pose_data(out_v8)
        else:
            bboxes, confidences, masks, classes = self.get_segmentator_data(out_v8)

        matches = self.associate_orientation_with_segmentation(orientation_bboxes, bboxes)

        associated_orientations = []
        for i in range(len(matches)):
            for j in range(len(matches)):
                if i == matches[j][1]:
                    associated_orientations.append(np.deg2rad(orientations[matches[j][0]][0]))
                    break

        return bboxes, confidences, associated_orientations, masks, classes

    def associate_orientation_with_segmentation(self, seg_bboxes, ori_bboxes):
        dists = matching.v_iou_distance(seg_bboxes, ori_bboxes)
        matches, unmatched_a, unmatched_b = matching.linear_assignment(dists, 0.9)
        return matches

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

    def get_segmentator_data(self, result):
        segmentation_bboxes = []
        segmentation_masks = []
        segmentation_confidences = []
        segmentation_classes = []
        for result in result:
            if result.masks != None and result.boxes != None:
                masks = result.masks.xy
                boxes = result.boxes
                if len(masks) == len(boxes):
                    for i in range(len(boxes)):
                        person_bbox = boxes[i].xyxy.cpu().numpy().astype(int)[0]
                        segmentation_bboxes.append(person_bbox)
                        segmentation_classes.append(boxes[i].cls.cpu().numpy()[0])
                        segmentation_confidences.append(boxes[i].conf.cpu().numpy()[0])
                        segmentation_masks.append(masks[i].astype(int))
        return segmentation_bboxes, segmentation_confidences, segmentation_masks, segmentation_classes

    def get_orientation_data(self, results, processed_image, original_image):
        out = non_max_suppression(results, 0.3, 0.5, num_angles=self.data['num_angles'])
        orientation_bboxes = scale_coords(processed_image.shape[2:], out[0][:, :4], original_image.shape[:2]).cpu().numpy().astype(int)  # native-space pred
        orientations = (out[0][:, 6:].cpu().numpy() * 360) - 180   # N*1, (0,1)*360 --> (0,360)
        return orientation_bboxes, orientations

    def yolov8_objects(self, blob):
        """
        This method infers objects in the given image blob using YOLO (You Only Look Once) version 8 object detection model.

        Args:
            blob (numpy array): A preprocessed image blob ready for object detection.

        Returns:
            dets (numpy array): An array with detected objects information. Each object is represented with a 1D array:
                                [x1, y1, x2, y2, score, class_id]. The first four values denote the bounding box coordinates,
                                the fifth value is the confidence score of the detection, and the last value is the detected
                                object's class index.

        The method begins by feeding the blob into the YOLOv8 model's inference engine. The model returns several outputs:
        1) 'num': number of detected objects,
        2) 'final_boxes': a 2D array where each row corresponds to a detected object and contains its bounding box
           coordinates,
        3) 'final_scores': a 1D array containing the confidence scores for each detected object,
        4) 'final_cls_inds': a 1D array containing the class index for each detected object.

        These outputs are then processed and concatenated into a single array 'dets' for convenience, and this array is returned.
        """
        data = self.yolo_object_predictor.infer(blob)
        print(data[0])
        exit()
        # skeleton = self.yolo_pose_predictor.infer(blob)
        # print(skeleton)
        # num, final_boxes, final_scores, final_cls_inds = data
        # final_boxes = np.reshape(final_boxes, (-1, 4))
        # dets = np.concatenate([final_boxes[:num[0]], final_scores[:num[0]].reshape(-1, 1),
        #                        final_cls_inds[:num[0]].reshape(-1, 1)], axis=-1)
        # return dets

    def create_interface_data(self, boxes, scores, cls_inds, mask_rois, orientations): #Optimizado
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
        objects_write = ifaces.RoboCompVisualElements.TObjects()

        # Extracting desired indices, scores, boxes, and classes in one go
        desired_data = [(i, score, box, cls, roi, orientation) for i, (score, box, cls, roi, orientation) in enumerate(zip(scores, boxes, cls_inds, mask_rois, orientations)) if
                        cls in self.classes]
        for i, score, box, cls, roi, orientation in desired_data:
            act_object = ifaces.RoboCompVisualElements.TObject()
            act_object.type = int(cls)
            if int(cls) == 0:
                act_object.person = ifaces.RoboCompPerson.TPerson(orientation = float(orientation))
            act_object.left = int(box[0])
            act_object.top = int(box[1])
            act_object.right = int(box[2])
            act_object.bot = int(box[3])
            act_object.score = float(score)
            act_object.image = self.get_bbox_image_data(roi)
            objects_write.append(act_object)
        return objects_write

    def get_bbox_image_data(self, image):
        bbox_image = ifaces.RoboCompCamera360RGB.TImage()
        bbox_image.image = image.tobytes()
        bbox_image.height, bbox_image.width, _ = image.shape
        bbox_image.roi = ifaces.RoboCompCamera360RGB.TRoi(xcenter=self.rgb.roi.xcenter,
                                                                ycenter=self.rgb.roi.ycenter,
                                                                xsize=self.rgb.roi.xsize, ysize=self.rgb.roi.ysize,
                                                                finalxsize=self.rgb.roi.finalxsize,
                                                                finalysize=self.rgb.roi.finalysize)
        return bbox_image

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
                self.target_roi_ysize = np.clip(int((bot - top)*4), 0, self.rgb_original.height)
                self.target_roi_xsize = self.target_roi_ysize
                return

        self.tracked_element = None
        self.tracked_id = None
        self.target_roi_xcenter = self.rgb_original.width // 2
        self.target_roi_ycenter = self.rgb_original.height // 2
        self.target_roi_xsize = self.rgb_original.width // 2
        self.target_roi_ysize = self.rgb_original.height

    ###############################################################
    def pre_process(self, image, input_size, swap=(2, 0, 1)):
        """
            Preprocesses an image for object detection.

            The preprocessing steps include padding the image to a given size, reversing the color channels
            from RGB to BGR, normalizing pixel values to [0,1], and rearranging the dimensions based on the 'swap' parameter.

            Args:
                image (numpy array): The original image to be preprocessed.
                input_size (tuple): The desired image size after padding, in the format (height, width).
                swap (tuple, optional): The order to which the image dimensions should be rearranged.

            Returns:
                padded_img (numpy array): The preprocessed image ready for object detection.
            """
        padded_img = np.ones((input_size[0], input_size[1], 3))
        img = np.array(image).astype(np.float32)
        padded_img[: int(img.shape[0]), : int(img.shape[1])] = img
        padded_img = padded_img[:, :, ::-1] # Swap color channels from RGB to BGR
        padded_img /= 255.0 # Normalize pixel values to [0,1]
        padded_img = padded_img.transpose(swap) # Swap dimensions
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img

    # def track(self, boxes, scores, cls_inds, depth_image):
    #     final_boxes = []
    #     final_scores = []
    #     final_cls_ids = []
    #     final_ids = []
    #     tracked_boxes = []
    #     tracked_scores = []
    #     tracked_ids = []
    #     tracked_clases = []
    #
    #     if self.depth_flag:
    #         online_targets = self.bytetrack_proxy.getTargetswithdepth(scores, boxes, depth_image, cls_inds)
    #     else:
    #         online_targets = self.bytetrack_proxy.getTargets(scores, boxes, cls_inds)
    #     for t in online_targets:
    #         tlwh = t.tlwh
    #         tid = t.trackid
    #         vertical = tlwh[2] / tlwh[3] > 1.6
    #         if tlwh[2] * tlwh[3] > 10 and not vertical:
    #             tracked_boxes.append(tlwh)
    #             tracked_ids.append(tid)
    #             tracked_scores.append(t.score)
    #             tracked_clases.append(t.clase)
    #     if tracked_boxes:
    #         tracked_boxes = np.asarray(tracked_boxes)
    #         tracked_boxes[:, 2] = tracked_boxes[:, 0] + tracked_boxes[:, 2]
    #         tracked_boxes[:, 3] = tracked_boxes[:, 1] + tracked_boxes[:, 3]
    #
    #         # we replace the original person boxes by the tracked ones
    #         # non_people_cls_inds = [i for i, cls in enumerate(cls_inds) if cls != 0]  # index of non-person elements
    #         final_boxes = tracked_boxes  # non-person boxes + tracked people
    #         final_scores = tracked_scores
    #         final_ids =  tracked_ids
    #         final_cls_ids = tracked_clases
    #
    #     return final_boxes, final_scores, final_cls_ids, final_ids

    def track(self, boxes, scores, cls_inds, depth_image):
        """
        Function to track objects in the image. If depth information is available, it is used to enhance tracking accuracy.

        Args:
            boxes (numpy array): Bounding box coordinates for detected objects.
            scores (numpy array): Confidence scores for detected objects.
            cls_inds (numpy array): Class indices for detected objects.
            depth_image (numpy array): Depth image, if available.

        Returns:
            final_boxes (numpy array): Updated bounding boxes after tracking.
            final_scores (numpy array): Updated confidence scores after tracking.
            final_cls_ids (numpy array): Updated class indices after tracking.
            final_ids (numpy array): Unique track IDs for tracked objects.
        """
        online_targets = self.bytetrack_proxy.getTargetswithdepth(scores, boxes, depth_image,
                                                                  cls_inds) if self.depth_flag else self.bytetrack_proxy.getTargets(
            scores, boxes, cls_inds)

        targets_data = [(t.tlwh, t.trackid, t.score, t.clase) for t in online_targets if
                        t.tlwh[2] * t.tlwh[3] > 10 and t.tlwh[2] / t.tlwh[3] <= 1.6]

        if targets_data:
            tracked_boxes, tracked_ids, tracked_scores, tracked_clases = zip(*targets_data)
            tracked_boxes = np.stack(tracked_boxes)
            tracked_boxes[:, 2] += tracked_boxes[:, 0]
            tracked_boxes[:, 3] += tracked_boxes[:, 1]

            final_boxes = tracked_boxes
            final_scores = list(tracked_scores)
            final_ids = list(tracked_ids)
            final_cls_ids = list(tracked_clases)

        return final_boxes, final_scores, final_cls_ids, final_ids

    def post_process(self, final_boxes, final_scores, final_cls_inds, final_inds):
        data = ifaces.RoboCompYoloObjects.TData()
        data.objects = []
        data.people = []

        for i in range(len(final_boxes)):
            box = final_boxes[i]
            ibox = ifaces.RoboCompYoloObjects.TBox()
            ibox.type = int(final_cls_inds[i])
            ibox.id = int(final_inds[i])
            ibox.prob = final_scores[i]
            ibox.left = int(box[0])
            ibox.top = int(box[1])
            ibox.right = int(box[2])
            ibox.bot = int(box[3])

            data.objects.append(ibox)
        return data

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
        for element in elements:
            x0, y0, x1, y1 = map(int, [element.left, element.top, element.right, element.bot])
            cls_ind = element.type
            color = (_COLORS[cls_ind] * 255).astype(np.uint8).tolist()

            # text = f'Class: {class_names[cls_ind]} - Score: {element.score * 100:.1f}% - ID: {element.id}'
            text = f'{cls_ind} - {element.id} - {element.person.orientation}'
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

    # =============== Methods for Component SubscribesTo ================
    # ===================================================================

    #
    # SUBSCRIPTION to setTrack method from SegmentatorTrackingPub interface
    #
    def SegmentatorTrackingPub_setTrack(self, track):
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
    # From the RoboCompCamera360RGB you can call this methods:
    # self.camera360rgb_proxy.getROI(...)

    ######################
    # From the RoboCompVisualElements you can use this types:
    # RoboCompVisualElements.TRoi
    # RoboCompVisualElements.TObject
