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

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import numpy as np
import time
import cv2
from threading import Thread, Event
import traceback
import queue
from shapely.geometry import box
from collections import defaultdict
import itertools

sys.path.append('/home/robocomp/software/TensorRT-For-YOLO-Series')
from utils.utils import BaseEngine



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
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 1
        self.thread_period = 10
        self.display = True

        if startup_check:
            self.startup_check()
        else:
            # trt
            self.yolo_object_predictor = BaseEngine(engine_path='yolov8.trt')

            # Hz
            self.cont = 0
            self.last_time = time.time()
            self.fps = 0

            # camera read thread
            self.read_queue = queue.Queue(1)
            self.event = Event()
            self.read_thread = Thread(target=self.get_rgb_thread, args=["camera_top", self.event], 
                                      name="read_queue", daemon=True)
            self.read_thread.start()

            # result data
            self.objects_write = ifaces.RoboCompYoloObjects.TData()
            self.objects_read = ifaces.RoboCompYoloObjects.TData()

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        try:
            self.classes = []
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
        if self.depth_flag:
            rgb, depth, blob, alive_time, period = self.read_queue.get()
        else:
            rgb, blob, alive_time, period = self.read_queue.get()

        dets = self.yolov7_objects(blob)

        if dets is not None:
            boxes, scores, cls_inds = self.dets_to_robocompifaces(dets[:, :4], dets[:, 4], dets[:, 5])

        if not self.display and not self.depth_flag:
            self.bytetrack_proxy.setTargets(scores, boxes, cls_inds, "Yolo") # ByteTracker without return and no depth

        elif not self.display:
            _ = self.bytetrack_proxy.getTargetswithdepth(scores, boxes, depth, cls_inds) # ByteTracker with depth

        else:
            tracked_boxes, tracked_scores, tracked_cls_inds, tracked_inds = self.track(boxes, scores, cls_inds,
                                                                                           False) if not self.depth_flag else self.track(boxes, scores, cls_inds,
                                                                                           depth)
            self.objects_write = self.post_process(tracked_boxes, tracked_scores, tracked_cls_inds, tracked_inds)

            self.objects_write, self.objects_read = self.objects_read, self.objects_write

            rgb = self.display_data(rgb, tracked_boxes, tracked_scores, tracked_cls_inds, tracked_inds,
                                        class_names=self.yolo_object_predictor.class_names)

            cv2.imshow("Detected Objects", rgb)
            cv2.waitKey(1)


    def dets_to_robocompifaces(self, boxes, scores, cls_inds):
        desired_inds = [i for i, cls in enumerate(cls_inds) if
                        cls in self.classes]  # index of elements that match desired classes
        desired_scores = scores[desired_inds]
        desired_boxes = boxes[desired_inds]
        desired_clases = cls_inds[desired_inds]
        boxes = ifaces.RoboCompByteTrack.Boxes()
        for i in desired_boxes:
            box = ifaces.RoboCompByteTrack.Box(i)
            boxes.append(box)
        scores = ifaces.RoboCompByteTrack.Scores(desired_scores)
        cls_inds = ifaces.RoboCompByteTrack.Clases([int(a) for a in desired_clases])
        return boxes, scores, cls_inds


    def get_rgb(self, name):
        try:
            rgb = self.camerargbdsimple_proxy.getImage(name)
            frame = np.frombuffer(rgb.image, dtype=np.uint8)
            frame = frame.reshape((rgb.height, rgb.width, 3))
        except:
            print("Error communicating with CameraRGBDSimple")
            traceback.print_exc()
            return
        return frame

    def get_rgb_thread(self, camera_name: str, event: Event):
        while not event.isSet():
            try:
                rgbd = self.camerargbdsimple_proxy.getAll(camera_name)
                color = np.frombuffer(rgbd.image.image, dtype=np.uint8).reshape(rgbd.image.height, rgbd.image.width, 3)
                blob = self.pre_process(color, (640, 640))
                delta = int(1000 * time.time() - rgbd.image.alivetime)
                if self.depth_flag:
                    # depth = np.frombuffer(rgbd.depth.depth, dtype=np.float32).reshape(rgbd.depth.height,
                    #                                                                   rgbd.depth.width, 1)
                    self.read_queue.put([color, rgbd.depth, blob, delta, rgbd.image.period])
                else:
                    self.read_queue.put([color, blob, delta, rgbd.image.period])
                event.wait(self.thread_period/1000)
            except:
                print("Error communicating with CameraRGBDSimple")
                traceback.print_exc()

    ###############################################################
    def pre_process(self, image, input_size, swap=(2, 0, 1)):
        padded_img = np.ones((input_size[0], input_size[1], 3))
        img = np.array(image).astype(np.float32)
        padded_img[: int(img.shape[0]), : int(img.shape[1])] = img
        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img

    def yolov7_objects(self, blob):
        data = self.yolo_object_predictor.infer(blob)
        num, final_boxes, final_scores, final_cls_inds = data
        final_boxes = np.reshape(final_boxes, (-1, 4))
        dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1),
                               np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        return dets

    def track(self, boxes, scores, cls_inds, depth_image):
        final_boxes = []
        final_scores = []
        final_cls_ids = []
        final_ids = []
        tracked_boxes = []
        tracked_scores = []
        tracked_ids = []
        tracked_clases = []

        if self.depth_flag:
            online_targets = self.bytetrack_proxy.getTargetswithdepth(scores, boxes, depth_image, cls_inds)
        else:
            online_targets = self.bytetrack_proxy.getTargets(scores, boxes, cls_inds)
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.trackid
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > 10 and not vertical:
                tracked_boxes.append(tlwh)
                tracked_ids.append(tid)
                tracked_scores.append(t.score)
                tracked_clases.append(t.clase)
        if tracked_boxes:
            tracked_boxes = np.asarray(tracked_boxes)
            tracked_boxes[:, 2] = tracked_boxes[:, 0] + tracked_boxes[:, 2]
            tracked_boxes[:, 3] = tracked_boxes[:, 1] + tracked_boxes[:, 3]

            # we replace the original person boxes by the tracked ones
            # non_people_cls_inds = [i for i, cls in enumerate(cls_inds) if cls != 0]  # index of non-person elements
            final_boxes = tracked_boxes  # non-person boxes + tracked people
            final_scores = tracked_scores
            final_ids =  tracked_ids
            final_cls_ids = tracked_clases

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

    def display_data(self, img, boxes, scores, cls_inds, inds, class_names=None):
        #print(len(inds), len(boxes))
        for i in range(len(boxes)):
            if inds[i] == -1:
                continue
            bb = boxes[i]
            cls_ids = int(cls_inds[i])
            ids = inds[i]
            score = scores[i]
            x0 = int(bb[0])
            y0 = int(bb[1])
            x1 = int(bb[2])
            y1 = int(bb[3])
            color = (_COLORS[cls_ids] * 255).astype(np.uint8).tolist()
            text = 'Class: {} - Score: {:.1f}% - ID: {}'.format(class_names[cls_ids], score*100, ids)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_ids]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            txt_bk_color = (_COLORS[cls_ids] * 255 * 0.7).astype(np.uint8).tolist()
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

    # ===================================================================
    #
    # IMPLEMENTATION of getImage method from YoloObjects interface
    #
    def YoloObjects_getImage(self):
        # ret = RoboCompYoloObjects.RoboCompCameraRGBDSimple::TImage()
        #
        # write your CODE here
        #
        return ret
        #

    # IMPLEMENTATION of getYoloJointNames method from YoloObjects interfa
    #
    def YoloObjects_getYoloJointData(self):
        ret = ifaces.RoboCompYoloObjects.TJointData()
        ret.jointNames = {}
        for i, jnt in enumerate(_JOINT_NAMES):
            ret.jointNames[i] = jnt
        ret.connections = []
        for a, b in _CONNECTIONS:
            conn = ifaces.RoboCompYoloObjects.TConnection()
            conn.first = a
            conn.second = b
            ret.connections.append(conn)
        return ret

    # IMPLEMENTATION of getYoloObjectNames method from YoloObjects interf
    #
    def YoloObjects_getYoloObjectNames(self):
        return self.yolo_object_predictor.class_names

    # IMPLEMENTATION of getYoloObjects method from YoloObjects interface
    #
    def YoloObjects_getYoloObjects(self):
        return self.objects_read

    ######################
    # From the RoboCompCameraRGBDSimple you can call this methods:
    # self.camerargbdsimple_proxy.getAll(...)
    # self.camerargbdsimple_proxy.getDepth(...)
    # self.camerargbdsimple_proxy.getImage(...)

    ######################
    # From the RoboCompCameraRGBDSimple you can use this types:
    # RoboCompCameraRGBDSimple.TImage
    # RoboCompCameraRGBDSimple.TDepth
    # RoboCompCameraRGBDSimple.TRGBD

    ######################
    # From the RoboCompHumanCameraBody you can use this types:
    # RoboCompHumanCameraBody.TImage
    # RoboCompHumanCameraBody.TGroundTruth
    # RoboCompHumanCameraBody.KeyPoint
    # RoboCompHumanCameraBody.Person
    # RoboCompHumanCameraBody.PeopleData

 ######################
    # From the RoboCompByteTrack you can call this methods:
    # self.bytetrack_proxy.getTargets(...)

    ######################
    # From the RoboCompByteTrack you can use this types:
    # RoboCompByteTrack.Targets


