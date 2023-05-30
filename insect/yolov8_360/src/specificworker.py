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
                print(" ratio {:.2f}.format(image.width/image.height)")

                # Image ROI require parameters
                self.final_xsize = 640
                self.final_ysize = 640
                self.roi_xsize = self.rgb_original.width // 2
                self.roi_ysize = self.rgb_original.height
                self.roi_xcenter = self.rgb_original.width // 2
                self.roi_ycenter = self.rgb_original.height // 2

                #Target ROI size
                self.target_roi_xsize = self.roi_xsize
                self.target_roi_ysize = self.roi_ysize
                self.target_roi_xcenter = self.roi_xcenter
                self.target_roi_ycenter = self.roi_ycenter

                started_camera = True
            except Ice.Exception as e:
                traceback.print_exc()
                print(e, "Trying again...")
                time.sleep(2)

        if startup_check:
            self.startup_check()
        else:
            self.window_name = "Yolo Segmentator"

            # trt
            self.yolo_object_predictor = BaseEngine(engine_path='yolov8.trt')

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

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        try:
            self.classes = []
            self.display = params["display"] == "true" or params["display"] == "True"
            self.depth_flag = params["depth"] == "true" or params["depth"] == "True"
            self.read_thread = Thread(target=self.get_rgb_thread, args=["camera_top", self.event],
                                      name="read_queue", daemon=True)
            self.read_thread.start()
            with open(params["classes-path-file"]) as f:
                [self.classes.append(_OBJECT_NAMES.index(line.strip())) for line in f.readlines()]
            print("Params read. Starting...", params)
        except:
            print("Error reading config params")
            traceback.print_exc()

        return True

    @QtCore.Slot()
    def compute(self):
        # Get image with depth
        if self.depth_flag:
            rgb, depth, blob, alive_time, period = self.read_queue.get()
        else:
            rgb, blob, alive_time, period = self.read_queue.get()

        if self.tracked_id != None:
            self.set_roi_dimensions(self.objects_read)

        dets = self.yolov8_objects(blob)

        if dets is not None:
            self.create_interface_data(dets[:, :4], dets[:, 4], dets[:, 5])
            if self.display:
                img = self.display_data_tracks(rgb, self.objects_read,
                                        class_names=self.yolo_object_predictor.class_names)
                cv2.imshow(self.window_name, img)
                cv2.waitKey(1)
        # FPS
        try:
            self.show_fps(alive_time, period)
        except KeyboardInterrupt:
            self.event.set()

    ######################################################################################################3

    def get_rgb_thread(self, camera_name: str, event: Event):
        while not event.is_set():
            try:
                # print("ACT ROI:", self.roi_xcenter, self.roi_ycenter, self.roi_xsize, self.roi_ysize, self.final_xsize, self.final_ysize)
                # print("TARGET DATA:", self.tracked_id, self.target_roi_xcenter, self.target_roi_ycenter, self.target_roi_xsize, self.target_roi_ysize)
                self.rgb = self.camera360rgb_proxy.getROI(self.roi_xcenter, self.roi_ycenter, self.roi_xsize,
                                                      self.roi_ysize, self.final_xsize, self.final_ysize)
                color = np.frombuffer(self.rgb.image, dtype=np.uint8).reshape(self.rgb.height, self.rgb.width, 3)
                blob = self.pre_process(color, (640, 640))
                delta = int(1000 * time.time() - self.rgb.alivetime)
                if self.depth_flag:
                    # depth = np.frombuffer(rgbd.depth.depth, dtype=np.float32).reshape(rgbd.depth.height,
                    #                                                                   rgbd.depth.width, 1)
                    self.read_queue.put([color, self.rgb.depth, blob, delta, self.rgb.period])
                else:
                    self.read_queue.put([color, blob, delta, self.rgb.period])
                if self.roi_xcenter != self.target_roi_xcenter or self.roi_ycenter != self.target_roi_ycenter or self.roi_xsize != self.target_roi_xsize or self.roi_ysize != self.target_roi_xsize:
                    self.from_act_roi_to_target()

                event.wait(self.thread_period/1000)
            except:
                print("Error communicating with Camera360RGB")
                traceback.print_exc()

    def from_act_roi_to_target(self):

        # self.target_roi_xcenter_list.put(x_diff_act)
        # total = sum(self.target_roi_xcenter_list.queue)
        # contador = self.target_roi_xcenter_list.qsize()
        # x_diff = total / contador if contador > 0 else 0
        # if self.target_roi_xcenter_list.full():
        #     self.target_roi_xcenter_list.get()
        # print(list(self.target_roi_xcenter_list.queue))
        # print("LIST SIZE:", contador)
        # print("LIST SIZE:", x_diff)

        x_diff = abs(self.target_roi_xcenter - self.roi_xcenter)
        y_diff = abs(self.target_roi_ycenter - self.roi_ycenter)
        x_size_diff = abs(self.target_roi_xsize - self.roi_xsize)
        y_size_diff = abs(self.target_roi_ysize - self.roi_ysize)

        # aux_x_diff defined for setting pixel speed and calculate derivated component
        aux_x_diff = x_diff if x_diff < self.rgb_original.width / 2 else self.rgb_original.width - x_diff
        x_der_diff = int(self.k2 * abs(self.last_ROI_error - aux_x_diff))

        print("")
        print("x_diff", x_diff, "target_roi_xcenter", self.target_roi_xcenter, "roi_xcenter", self.roi_xcenter)
        print("self.last_ROI_error", self.last_ROI_error, "x_der_diff", x_der_diff)


        x_mod_speed = np.clip(int(self.k1 * aux_x_diff), 0, 22) + x_der_diff
        y_mod_speed = np.clip(int(self.k1 * y_diff), 0, 20)

        print("x_mod_speed", x_mod_speed)

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

    def yolov8_objects(self, blob):
        data = self.yolo_object_predictor.infer(blob)
        num, final_boxes, final_scores, final_cls_inds = data
        final_boxes = np.reshape(final_boxes, (-1, 4))
        dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1),
                               np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        return dets

    def create_interface_data(self, boxes, scores, cls_inds):
        self.objects_write = ifaces.RoboCompVisualElements.TObjects()
        desired_inds = [i for i, cls in enumerate(cls_inds) if True]
                        #cls in self.classes]  # index of elements that match desired classes
        desired_scores = scores[desired_inds]
        desired_boxes = boxes[desired_inds]
        desired_clases = cls_inds[desired_inds]
        for index in range(len(desired_scores)):
            act_object = ifaces.RoboCompVisualElements.TObject()
            act_object.type = int(desired_clases[index])
            act_object.left = int(desired_boxes[index][0])
            act_object.top = int(desired_boxes[index][1])
            act_object.right = int(desired_boxes[index][2])
            act_object.bot = int(desired_boxes[index][3])
            act_object.score = desired_scores[index]
            act_object.roi = ifaces.RoboCompVisualElements.TRoi(xcenter=self.rgb.roi.xcenter, ycenter=self.rgb.roi.ycenter,
                                                                xsize=self.rgb.roi.xsize, ysize=self.rgb.roi.ysize,
                                                                finalxsize=self.rgb.roi.finalxsize, finalysize=self.rgb.roi.finalysize)
            self.objects_write.append(act_object)
        self.objects_write = self.visualelements_proxy.getVisualObjects(self.objects_write)
        # swap
        self.objects_write, self.objects_read = self.objects_read, self.objects_write

    def set_roi_dimensions(self, objects):
        for object in objects:
            if object.id == self.tracked_id:
                x_roi_offset = object.roi.xcenter - object.roi.xsize / 2
                y_roi_offset = object.roi.ycenter - object.roi.ysize / 2
                x_factor = object.roi.xsize / object.roi.finalxsize
                y_factor = object.roi.ysize / object.roi.finalysize

                left = int(object.left * x_factor + x_roi_offset)
                right = (object.right * x_factor + x_roi_offset)

                top = int(object.top * y_factor + y_roi_offset)
                bot = int(object.bot * y_factor + y_roi_offset)

                self.target_roi_xcenter = (left + (right - left)/2) % self.rgb_original.width
                self.target_roi_ycenter = (top + (bot - top)/2)
                self.target_roi_ysize = np.clip(int((bot - top)*4), 0, self.rgb_original.height)
                self.target_roi_xsize = self.target_roi_ysize
                return

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
            # if inds[i] == -1:
            #     continue
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

    def display_data_tracks(self, img, elements, class_names=None):
        #print(len(inds), len(boxes))
        for i in elements:
            # if inds[i] == -1:
            #     continue
            x0 = int(i.left)
            y0 = int(i.top)
            x1 = int(i.right)
            y1 = int(i.bot)
            color = (_COLORS[i.type] * 255).astype(np.uint8).tolist()
            text = 'Class: {} - Score: {:.1f}% - ID: {}'.format(class_names[i.type], i.score*100, i.id)
            txt_color = (0, 0, 0) if np.mean(_COLORS[i.type]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            txt_bk_color = (_COLORS[i.type] * 255 * 0.7).astype(np.uint8).tolist()
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
        if track == -1:
            self.tracked_element = None
            self.tracked_id = None
            self.target_roi_xcenter = self.rgb_original.width // 2
            self.target_roi_ycenter = self.rgb_original.height // 2
            self.target_roi_xsize = self.rgb_original.width // 2
            self.target_roi_ysize = self.rgb_original.height
            return

        for track_obj in self.objects_read:
            if track_obj.id == track:
                self.target_roi_xcenter_list = queue.Queue(10)
                self.tracked_element = track_obj
                self.tracked_id = track
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
