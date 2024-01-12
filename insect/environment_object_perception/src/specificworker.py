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
# from PIL import Image
import copy
# import math

print(os.path.abspath(__file__) )
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/HashTrack')
from hash import HashTracker
from basetrack import TrackState
import matching
from ultralytics import YOLO
import torch
import itertools

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
            self.Period = 200
            self.thread_period = 200
            self.display = False
            
            self.yolo_model_name = 'yolov8s-seg.engine'
            
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
            
            self.window_name = "Yolo Segmentator"
            
            print("Loaded YOLO model: ", self.yolo_model_name)

            # Hz
            self.cont = 0
            self.last_time = time.time()
            self.fps = 0
            
            # interface swap objects
            self.objects_read = ifaces.RoboCompVisualElementsPub.TObjects()
            self.objects_write = ifaces.RoboCompVisualElementsPub.TObjects()
            
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
            self.lidar_data = None

            self.rgb_queue_len = 1
            
            self.rgb_read_queue = deque(maxlen=self.rgb_queue_len)
            self.inference_read_queue = deque(maxlen=1)

            self.event = Event()
            
            self.last_time_with_data = time.time()
            self.last_lidar_stamp = time.time()

            self.image_read_thread = Thread(target=self.get_rgb_thread, args=["camera_top", self.event],
                                      name="rgb_read_queue", daemon=True)
            self.image_read_thread.start()
                        
            self.inference_execution_thread = Thread(target=self.inference_thread, args=[self.event],
                                      name="inference_read_queue", daemon=True)
            self.inference_execution_thread.start()
            
            
            #TRACKER
            self.tracker = HashTracker(frame_rate=20)

            # Best time stamp
            self.lowest_timestamp = 0

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

        if self.inference_read_queue:
            start = time.time()
            out_v8_front, color_front, orientation_bboxes_front, orientations_front, out_v8_back, color_back, orientation_bboxes_back, orientations_back, depth_front, depth_back, delta, alive_time, period, front_roi, back_roi = self.inference_read_queue.pop()
            objects_front = self.get_segmentator_data(out_v8_front, color_front, depth_front)
            objects_back = self.get_segmentator_data(out_v8_back, color_back, depth_back)

            matches, unm_a, unm_b = self.associate_orientation_with_segmentation(objects_front["bboxes"], orientation_bboxes_front)
            for i in range(len(matches)):
                objects_front["orientations"][matches[i][0]] = np.deg2rad(orientations_front[matches[i][1]][0])
            matches, unm_a, unm_b = self.associate_orientation_with_segmentation(objects_back["bboxes"], orientation_bboxes_back)
            for i in range(len(matches)):
                objects_back["orientations"][matches[i][0]] = np.deg2rad(orientations_back[matches[i][1]][0])

            tracks_front = self.tracker.update(np.array(objects_front["confidences"]),
                                np.array(objects_front["bboxes"]),
                                np.array(objects_front["classes"]),
                                np.array(objects_front["masks"], dtype=object),
                                np.array(objects_front["hashes"]),
                                np.array(objects_front["poses"]),
                                np.array(objects_front["orientations"]))
            
            tracks_back = self.tracker.update(np.array(objects_back["confidences"]),
                                np.array(objects_back["bboxes"]),
                                np.array(objects_back["classes"]),
                                np.array(objects_back["masks"], dtype=object),
                                np.array(objects_back["hashes"]),
                                np.array(objects_back["poses"]),
                                np.array(objects_back["orientations"]))

            # Compare tracks_front and tracks_back to remove tracks with same ID
            for track_front in tracks_front:
                for track_back in tracks_back:
                    if track_front.track_id == track_back.track_id:
                        if track_front.score > track_back.score:
                            tracks_back.remove(track_back)
                        else:
                            tracks_front.remove(track_front)
                        break

            front_objects = self.to_visualelements_interface(tracks_front, alive_time, front_roi)
            back_objects = self.to_visualelements_interface(tracks_back, alive_time, back_roi)

            # Fuse front_objects and back_objects and equal it to self.objects_write
            self.objects_write = front_objects
            self.objects_write.objects += back_objects.objects

            self.visualelementspub_proxy.setVisualObjects(self.objects_write)

            # If display is enabled, show the tracking results on the image
            if self.display:
                img_front = self.display_data_tracks(color_front, self.objects_write_front.objects)
                img_back = self.display_data_tracks(color_back, self.objects_write_back.objects)
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
                    if (time.time() - start) > 0.1:
                        print("Time exceded get image")
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
                color_front, front_img_tensor, color_back, back_img_tensor, depth_front, depth_back, delta, period, alive_time, front_roi, back_roi = self.rgb_read_queue.pop()
                out_v8_front, orientation_bboxes_front, orientations_front = self.inference_over_image(color_front, front_img_tensor)
                out_v8_back, orientation_bboxes_back, orientations_back = self.inference_over_image(color_back, back_img_tensor)
                self.inference_read_queue.append([out_v8_front, color_front, orientation_bboxes_front, orientations_front, out_v8_back, color_front, orientation_bboxes_back, orientations_back, depth_front, depth_back, delta, alive_time, period, front_roi, back_roi])
                if (time.time() - start) > 0.1:
                    print("Time exceded inference")
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
        out_v8 = self.model_v8.predict(img0, show_conf=True)
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
            # object_ = ifaces.RoboCompVisualElements.TObject(
            #     id=int(track.track_id), score=float(track.score),
            #     left=int(track.bbox[0]), top=int(track.bbox[1]),
            #     right=int(track.bbox[2]),
            #     bot=int(track.bbox[3]), type=track.clase,
            #     image=self.mask_to_TImage(track.image, roi), person = ifaces.RoboCompPerson.TPerson(orientation = round(float(track.orientation), 2)))
            # # speed_module = math.sqrt(round(track.speed[0], 2) ** 2 + round(track.speed[1], 2) ** 2)
            # object_.x = object_.x[0]
            # object_.y = object_.y[0]
            # object_.metrics = ifaces.RoboCompVisualElements.TMetrics(track.metrics)
            x_pose = round(track.mean[0], 2) if track.kalman_initiated else round(track._pose[0], 2),
            y_pose = round(track.mean[1], 2) if track.kalman_initiated else round(track._pose[1], 2),
            z_pose = 0
            generic_attrs = {
                "score": str(track.score),
                "bbox_left": str(int(track.bbox[0])),
                "bbox_top": str(int(track.bbox[1])),
                "bbox_right": str(int(track.bbox[2])),
                "bbox_bot": str(int(track.bbox[3])),
                "x_pos": str(x_pose[0]),
                "y_pos": str(y_pose[0]),
                "z_pos": str(z_pose),
                "orientation": str(round(float(track.orientation), 2))
            }
            object_ = ifaces.RoboCompVisualElementsPub.TObject(id=int(track.track_id), type=track.clase, attributes=generic_attrs, image=self.mask_to_TImage(track.image, roi))
            objects.append(object_)
        # print("IMAGE TIMESTAMP", image_timestamp)
        visual_elements = ifaces.RoboCompVisualElementsPub.TData(timestampimage=image_timestamp, timestampgenerated=int(time.time() * 1000), period=self.Period, objects=objects)
        return visual_elements

    def mask_to_TImage(self, mask, roi):
        y, x, _ = mask.shape
        return ifaces.RoboCompCamera360RGB.TImage(image=mask.tobytes(), height=y, width=x, roi=roi)

    def associate_orientation_with_segmentation(self, seg_bboxes, ori_bboxes):
        dists = matching.v_iou_distance(seg_bboxes, ori_bboxes)
        matches, unmatched_a, unmatched_b = matching.linear_assignment(dists, 0.9)
        return matches, unmatched_a, unmatched_b

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
        objects = {"bboxes" : [], "poses" : [], "confidences" : [], "masks" : [], "classes" : [], "orientations": [], "hashes" : []}
        roi_ysize, roi_xsize, _ = color_image.shape
        for result in results:
            if result.masks != None and result.boxes != None:
                masks = result.masks.xy
                boxes = result.boxes
                if len(masks) == len(boxes):
                    # print("LEN MASKS", len(boxes))
                    for i in range(len(boxes)):
                        element_confidence = boxes[i].conf.cpu().numpy()[0]
                        # print("element_confidence", element_confidence)
                        if element_confidence > 0.4:
                            element_class = boxes[i].cls.cpu().numpy().astype(int)[0]
                            element_bbox = boxes[i].xyxy.cpu().numpy().astype(int)[0]
                            image_mask = np.zeros((roi_ysize, roi_xsize, 1), dtype=np.uint8)
                            act_mask = masks[i].astype(np.int32)
                            cv2.fillConvexPoly(image_mask, act_mask, (1, 1, 1))
                            image_mask_element = image_mask[element_bbox[1]:element_bbox[3], element_bbox[0]:element_bbox[2]]
                            color_image_mask = color_image[element_bbox[1]:element_bbox[3], element_bbox[0]:element_bbox[2]]
                            element_mask = self.get_mask_with_modified_background(image_mask_element, color_image_mask)
                            element_hash = self.get_color_histogram(element_mask)
                            height, width, _ = image_mask_element.shape
                            depth_image_mask = depth_image[element_bbox[1] :element_bbox[3], element_bbox[0]:element_bbox[2]]
                            element_pose = self.get_mask_distance(image_mask_element, depth_image_mask, element_bbox)
                            objects["bboxes"].append(element_bbox)
                            objects["poses"].append(element_pose)
                            objects["confidences"].append(element_confidence)
                            objects["masks"].append(element_mask)             
                            objects["classes"].append(element_class) 
                            objects["hashes"].append(element_hash)  
                            
        objects["orientations"] = [-4] * len(objects["bboxes"])
        return objects

    def get_mask_distance(self, mask, depth_image, bbox):
        segmentation_points = np.argwhere(np.all(mask == 1, axis=-1))[:, [1, 0]]
        # print("SEGMENTATOON", segmentation_points)
        # Extraer directamente los puntos válidos (no ceros) y sus distancias en un solo paso.
        # Esto evitará tener que crear y almacenar matrices temporales y booleanas innecesarias.
        valid_points = depth_image[segmentation_points[:, 1], segmentation_points[:, 0]]
        valid_points = valid_points[np.any(valid_points != [0, 0, 0], axis=-1)]
        # print("valid_points",valid_points )
        if valid_points.size == 0:
            return [0, 0, 0]
        distances = np.linalg.norm(valid_points, axis=-1)
        # Parece que intentas generar bins para un histograma basado en un rango de distancias.
        # Crear los intervalos directamente con np.histogram puede ser más eficiente.
        hist, edges = np.histogram(distances, bins=np.arange(np.min(distances), np.max(distances), 300))
        if hist.size == 0:
            return [0, 0, 0]
        # Identificar el intervalo de moda y extraer los índices de las distancias que caen dentro de este intervalo.
        max_index = np.argmax(hist)

        pos_filtered_distances = np.logical_and(distances >= edges[max_index], distances <= edges[max_index + 1])
        # Filtrar los puntos válidos y calcular la media en un solo paso.
        person_dist = np.mean(valid_points[pos_filtered_distances], axis=0)
        return person_dist.tolist()

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
        for element in elements:
            x0, y0, x1, y1 = map(int, [element.left, element.top, element.right, element.bot])
            cls_ind = element.type
            color = (_COLORS[cls_ind] * 255).astype(np.uint8).tolist()
            # text = f'Class: {class_names[cls_ind]} - Score: {element.score * 100:.1f}% - ID: {element.id}'
            text = f'{element.x} - {element.y} - {element.id} - {element.person.orientation} - {_OBJECT_NAMES[element.type]}'
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
