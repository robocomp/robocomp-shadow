#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2023 by YOUR NAME HERE
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

import sys
from PySide6.QtCore import QTimer
from PySide6.QtGui import QPixmap, QIcon, QPolygon
from PySide6.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import numpy as np
import traceback
import cv2
import time
import itertools
import queue
from dwa_optimizer import DWA_Optimizer
sys.path.append('/home/robocomp/robocomp/lib')
console = Console(highlight=False)
from dataclasses import dataclass
from glviewer import GLViewer

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
        self.Period = 100
        if startup_check:
            self.startup_check()
        else:
            # 3D viewer
            self.glviewer = GLViewer(self.ui.frame_3d, [])

            self.visual_objects = None
            self.segmented_img = None
            self.target_object = None
            self.selected_object = None

            self.yolo = True
            self.semantic = True

            # ROI parameters
            @dataclass
            class TRoi:
                final_xsize: int = 0
                final_ysize: int = 0
                xcenter: int = 0
                ycenter: int = 0
                xsize: int = 0
                ysize: int = 0
            self.roi = TRoi()

            # Hz
            self.cont = 0
            self.last_time = time.time()
            self.fps = 0

            # get camera specs
            try:
                self.rgb = self.camera360rgb_proxy.getROI(-1, -1, -1, -1, -1, -1)

                print("Camera specs:")
                print(" width:", self.rgb.width)
                print(" height:", self.rgb.height)
                print(" depth", self.rgb.depth)
                print(" focalx", self.rgb.focalx)
                print(" focaly", self.rgb.focaly)
                print(" period", self.rgb.period)
                print(" ratio {:.2f}".format(self.rgb.width/self.rgb.height))
            except Ice.Exception as e:
                traceback.print_exc()
                print(e, "Cannot connect to camera. Aborting")
                sys.exit()

            # camera matrix
            rx = np.deg2rad(20)  # get from camera_proxy
            cam_to_robot = self.make_matrix_rt(rx, 0, 0, 0, -15, 1580)  # converts points in camera CS to robot CS
            #cam_to_robot = self.make_matrix_rt(0, 0, 0, 0, 0, 1580)  # converts points in camera CS to robot CS
            robot_to_cam = np.linalg.inv(cam_to_robot)

            # visual objects
            self.visual_objects_queue = queue.Queue(1)
            self.total_objects = []

            # optimizer
            #self.dwa_optimizer = DWA_Optimizer(robot_to_cam, self.rgb.focalx, self.rgb.focaly, (self.rgb.height, self.rgb.width, self.rgb.depth))
            self.dwa_optimizer = DWA_Optimizer(robot_to_cam, self.rgb.focalx, self.rgb.focaly, (384, 384, 3))

            # signals
            self.selected_index = None
            self.buttons = [self.ui.pushButton_leftleft, self.ui.pushButton_left, self.ui.pushButton_centre,
                            self.ui.pushButton_right, self.ui.pushButton_rightright]
            pixmap = QPixmap("icons/vector-simple-arrow-sign-icon.png")
            icon = QIcon(pixmap)
            self.ui.pushButton_centre.setIcon(icon)
            pixmap = QPixmap("icons/th.png")
            icon = QIcon(pixmap)
            self.ui.pushButton_right.setIcon(icon)

            # bins
            plusbins = np.geomspace(0.001, 1, 10)
            minusbins = -np.geomspace(0.0001, 1, 10)
            self.bins = np.append(minusbins[::-1], plusbins)  # compose both gemspaces from -1 to 1 with high density close to 0

            self.ui.frame_2d.mousePressEvent = self.set_target_with_mouse

            self.publishing_time = 0.6
            self.last_publish_time = time.time()

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)
    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        try:
            self.yolo = (params["yolo"] == "True") or (params["yolo"] == "true")
            self.semantic = (params["semantic"] == "True") or (params["semantic"] == "true")
            print("Params:", params)

            # semantic_segmentation
            if self.semantic:
                try:
                    self.semantic_classes = self.maskelements_proxy.getNamesofCategories()
                except Ice.Exception as e:
                    traceback.print_exc()
                    print(e, "Cannot connect to MaskElements interface. Aborting")
                    sys.exit()
        except:
            traceback.print_exc()
            print("Error reading config params")
        return True

    @QtCore.Slot()
    def compute(self):
        # Get camera frame
        frame = self.read_image()

        # Get masks from mask2former. Moved to DWA components


        # Get objects from yolo and mask2former, and transform them
        if self.yolo:
            yolo_objects = self.read_yolo_objects()
        else:
            yolo_objects = []
        if self.semantic:
            sm_objects = self.read_ss_objects()

            sm_masks = self.get_ss_masks(["floor"])
        else:
            sm_objects = []
            sm_masks = []
        self.total_objects = self.transform_and_concatenate_objects(yolo_objects, sm_objects)
        # check if selected_object is in self.total_objects
        if self.selected_object is not None and len(self.total_objects) > 0:
            found, self.selected_object = self.check_selected_in_new_objects(self.total_objects, self.selected_object)
            if not found:
                print("Warning: target LOST")

        # publish target
        if time.time() - self.last_publish_time > self.publishing_time:
            self.publish_target(self.selected_object)
            self.last_publish_time = time.time()

        # Draw image
        self.draw_frame(frame, self.total_objects, sm_masks)
        self.show_fps()

    #########################################################################
    def set_target_with_mouse(self, event):
        x = int(event.pos().x()*(self.rgb.width / self.ui.frame_2d.width()))
        y = int(event.pos().y()*(self.rgb.height / self.ui.frame_2d.height()))
        self.selected_object = None
        # print(list(self.mask2former.labels.keys())[list(self.mask2former.labels.values()).index(self.segmented_img[y, x].item())])
        # check if clicked point on yolo object. If so, set it as the new target object
        for b in self.total_objects:
            if x >= b.left and x < b.right and y >= b.top and y < b.bot:
                self.selected_object = b
                print("Selected target id = ", self.selected_object.id)
                return

    def check_selected_in_new_objects(self, elems, target):
        for e in elems:
            #print(len(elems), e.id, target.id)
            if e.id == target.id:
                target = e
                return True, target
        return False, None

    def publish_target(self, target):
        if target is not None:
            try:
                self.segmentatortrackingpub_proxy.setTrack(self.selected_object)
            except Ice.Exception as e:
                traceback.print_exc()
                print(e)
        else:
            self.segmentatortrackingpub_proxy.setTrack(ifaces.RoboCompVisualElements.TObject(id=-1))

    def read_image(self):
        frame = None
        try:
            # rgb = self.camera360rgb_proxy.getROI(512, 256, 512, 430, 384, 384)
            rgb = self.camera360rgb_proxy.getROI(-1, -1, -1, -1, -1, -1)
            frame = np.frombuffer(rgb.image, dtype=np.uint8).reshape((rgb.height, rgb.width, 3))
        except Ice.Exception as e:
            traceback.print_exc()
            print(e)
        return frame

    def get_ss_masks(self, class_names):
        # read masks from SemanticSegmentator
        ss_masks = []
        try:
            ss_masks_ = self.maskelements_proxy.getMasks(class_names)
            ss_masks = self.convert_from_2d_to_3d_mask_points(ss_masks_)
        except Ice.Exception as e:
            traceback.print_exc()
            print(e)
        return ss_masks

    def convert_from_2d_to_3d_mask_points(self, masks):
        masks_ = []
        for mask in masks:
            mask_img = np.frombuffer(mask.image, dtype=np.uint8).reshape((mask.height, mask.width, 1))
            mask_box = self.do_box(mask_img)
            if any(numero < 0 for numero in mask_box) or (mask_box[0] == mask_box[1]) or (mask_box[2] == mask_box[3]):
                return masks_
            print(mask_box)
            # Bbox points converted
            top_left_x, top_left_y = self.to_3d(mask_box[0], mask_box[2])
            top_right_x, top_right_y = self.to_3d(mask_box[1], mask_box[2])
            bot_left_x, bot_left_y = self.to_3d(mask_box[0], mask_box[3])
            bot_right_x, bot_right_y = self.to_3d(mask_box[1], mask_box[3])

            if top_left_x < bot_left_x:
                # Move negative x points to positive section
                if top_left_x < 0:
                    top_right_x += abs(top_left_x)
                    bot_right_x += abs(top_left_x)
                    bot_left_x += abs(top_left_x)
                    top_left_x += abs(top_left_x)
                else:
                    top_right_x -= top_left_x
                    bot_right_x -= top_left_x
                    bot_left_x -= top_left_x
                    top_left_x = 0
            else:
                if bot_left_x < 0:
                    top_right_x += abs(bot_left_x)
                    top_left_x += abs(bot_left_x)
                    bot_right_x += abs(bot_left_x)
                    bot_left_x += abs(bot_left_x)
                else:
                    top_right_x -= bot_left_x
                    top_left_x = 0
                    bot_right_x -= bot_left_x
                    bot_left_x -= bot_left_x

            image_polygon_width = top_right_x * 100 if top_right_x > bot_right_x else bot_right_x * 100

            print("ANCHO Y ALTO:", image_polygon_width, int(top_left_y * 100))
            polygon = QPolygon()
            polygon << QPoint(int(top_left_x * 100), int(top_left_y * 100)) << QPoint(int(top_right_x * 100), int(top_right_y * 100)) << QPoint(int(bot_right_x * 100), int(bot_right_y * 100)) << QPoint(int(bot_left_x * 100), int(bot_left_y * 100))

            print(int(top_left_x * 100), int(top_left_y * 100))
            print(int(top_right_x * 100), int(top_right_y * 100))
            print(int(bot_left_x * 100), int(bot_left_y * 100))
            print(int(bot_right_x * 100), int(bot_right_y * 100))
            if image_polygon_width > 100 and  int(top_left_y * 100) > 100:
                masks_.append(self.draw_polygon(polygon, int(top_left_y * 100), int(image_polygon_width)))
        return masks_
            # min_x = 0
            # max_x = 0
            # max_y = 0
            # mask_real_points = []
            # for y, row in enumerate(mask_img):
            #     x_values = np.where(row == 255)
            #     # Get maximun distance
            #     if len(x_values[0]) > 0:
            #         for x in x_values[0]:
            #             act_x, act_y = self.to_3d(x, y)
            #             mask_real_points.append([int(round(act_x, 2) * 100), int(round(act_y, 2) * 100)])
            #             if act_y > max_y:
            #                 max_y = act_y
            #             if act_x < min_x:
            #                 min_x = act_x
            #             if act_x > max_x:
            #                 max_x = act_x
            # y_array_dim = int(round(max_y, 2) * 100)
            # x_array_dim = int(round(2 * abs(min_x), 2) * 100) if abs(min_x) > abs(max_x) else int(round(2 * abs(max_x), 2) * 100)
            # print(x_array_dim, y_array_dim)
            # converted = np.zeros((y_array_dim + 1, x_array_dim + 1, 1), dtype=np.uint8)
            # for point in mask_real_points:
            #     converted[point[1]][point[0] + int(x_array_dim / 2)] = 255
            # converted = cv2.resize(converted, (int((x_array_dim + 1) / 2), int((y_array_dim + 1)/2)))

    def do_box(self, mask):
        box = cv2.boundingRect(mask)
        left = int(box[0])
        right = left + int(box[2]) - 1
        top = int(box[1])
        bot = top + int(box[3]) - 1
        return [left, right, top, bot]

    def draw_polygon(self, polygon, height, width):

        image = QImage(width, height, QImage.Format_RGB32)
        image.fill(Qt.black)

        # Crear un QPainter para dibujar en la imagen
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)

        fill_color = QColor(0, 255, 0)  # Rojo
        painter.setBrush(fill_color)
        outline_color = QColor(0, 0, 0)  # Azul
        painter.setPen(outline_color)
        painter.drawPolygon(polygon)
        painter.end()
        return image


    def to_3d(self, x, y):
        world_y = 128 * 1.35 / (y - 192)
        world_x = world_y * (x - 192) / 128
        return round(world_x, 2), round(world_y, 2)
    def read_yolo_objects(self):
        yolo_objects = ifaces.RoboCompVisualElements.TObjects()
        try:
            yolo_objects = self.visualelements_proxy.getVisualObjects([])
        except Ice.Exception as e:
            traceback.print_exc()
            print(e)
        return yolo_objects

    def read_ss_objects(self):
        ss_objects = ifaces.RoboCompVisualElements.TObjects()
        try:
            ss_objects = self.visualelements1_proxy.getVisualObjects([])
        except Ice.Exception as e:
            traceback.print_exc()
            print(e)
        return ss_objects

    def transform_and_concatenate_objects(self, yolo_objects, ss_objects):    #TODO: replace by list of lists
        # Concatenate objects from YOLO and ss
        total_objects = list(itertools.chain(yolo_objects, ss_objects))

        # Transform objects to main image size
        for element in total_objects:
            roi = element.image.roi
            final_xsize = roi.finalxsize
            final_ysize = roi.finalysize
            roi_xcenter = roi.xcenter
            roi_ycenter = roi.ycenter
            roi_xsize = roi.xsize
            roi_ysize = roi.ysize
            x_roi_offset = roi_xcenter-roi_xsize/2
            y_roi_offset = roi_ycenter-roi_ysize/2
            x_factor = roi_xsize / final_xsize
            y_factor = roi_ysize / final_ysize
            element.left = int(element.left * x_factor + x_roi_offset) % self.rgb.width
            element.right = int(element.right * x_factor + x_roi_offset) % self.rgb.width
            element.top = int(element.top*y_factor + y_roi_offset)
            element.bot = int(element.bot*y_factor + y_roi_offset)
        return total_objects

    def check_human_interface(self, candidates):
        # pre-conditions
        if any([button.isDown() for button in self.buttons]) is not True:
            return False, None, candidates
        if len(candidates) == 0:
            return False, None, candidates
        # core
        control = None
        self.selected_index = None
        ref_curvatures = [-0.2, -0.1, 0.0, 0.1, 0.2]
        # search paths close to reference curvature

        for c, button, i in zip(ref_curvatures, self.buttons, range(len(self.buttons))):
            # get the 5 closest elements by curvature and select the one with minimum loss
            selected = sorted(candidates, key=lambda item: np.abs(item["params"][2] - c))[0:5]
            # local_index = np.argmin(params[candidates][:, 3])
            selected = sorted(selected, key=lambda item: item["loss"])
            # advance, _, _, _ = params[candidates][local_index]
            # print("selected", len(selected))
            advance, _, _ = selected[0]["params"]
            # check here that loss or arc is enough
            if advance > 100:
                button.setEnabled(True)
                if button.isDown():
                    # self.selected_index = candidates[local_index]
                    self.selected_index = 0
                    # print("Down:", i, params[candidates][local_index])
                    print("Down:", i, selected[0]["params"])
                    control = selected[0]["params"][0:2]
                    advance, rotation, curvature = selected[0]["params"]
                    # print("Arrow:", advance, rotation, curvature, loss, "dist:", np.linalg.norm(targets[self.selected_index][-1]))
                    print("Arrow:", advance, rotation, curvature, selected[0]["loss"])
                    return True, control, selected

        return False, control, candidates

    def check_visual_target(self, candidates):
        # pre-conditions
        if len(candidates) == 0:
            return False, None, candidates
        if self.selected_object is None:
            return False, None, candidates

        # if a visual target is already selected, update it with one of the new objects
        if self.previous_yolo_id is not None:
            try:
                # find the object in self.yolo_objects that has the same id
                current = next(x for x in self.yolo_objects if x.score > 0.7 and x.id == self.previous_yolo_id)
                print("CheckVisualTarget:", current.id, current.type)
                self.selected_object = current
            except:
                self.previous_yolo_id = None
                print("CheckVisualTarget,no match found. Resetting id")
                return False, None, candidates
                self.selected_object = None
        else:
            current = self.selected_object

        # select closest direction to target
        center_object = np.array([current.x, current.y])
        selected = sorted(candidates, key=lambda item: (np.linalg.norm(item["trajectory"][-1] - center_object)))
        control = selected[0]["params"][0:2]
        self.previous_yolo_id = self.selected_object.id
        return True, control, [selected[0]]


    def sigmoid(self, rot):
        xset = 0.4
        yset = 0.3
        s = -xset * xset / np.log(yset)
        return np.exp(-rot * rot / s)

    def distance_to_target(self):
        center_object = np.array([self.selected_object.x, self.selected_object.y])
        dist = np.linalg.norm(center_object)
        if dist > 1500:
            return 1, dist
        else:
            return (1/1500.0) * dist, dist

    def show_fps(self):
        if time.time() - self.last_time > 1:
            self.last_time = time.time()
            cur_period = int(1000./self.cont)
            #delta = (-1 if (period - cur_period) < -1 else (1 if (period - cur_period) > 1 else 0))
            print("Freq:", self.cont, "ms. Curr period:", cur_period)
            #self.thread_period = np.clip(self.thread_period+delta, 0, 200)
            self.cont = 0
        else:
            self.cont += 1

    def draw_frame(self, frame, objects, masks):
        # alpha = 0.95
        # for s, c in enumerate(candidates):
        #     # alt = c["mask"]
        #     # alt_lane = cv2.cvtColor(alt, cv2.COLOR_GRAY2BGR)
        #     # if frame.shape != alt_lane.shape:
        #     #     print("Image and mask have different size. Returning")
        #     #     return
        #     # color = (0, 0, 255) if s == self.selected_index else (100, 100, 100)
        #     # alt_lane[np.all(alt_lane == (255, 255, 255), axis=-1)] = color
        #     # # we have to combine with the ROI part of the image, not all
        #     # roi_frame = frame[]
        #     # roi_frame = cv2.addWeighted(roi_frame, alpha, alt_lane, 1 - alpha, 0)
        #     # roi_frame.copyTo(frame[])
        #     center_x = frame.shape[0]//2
        #     cv2.polylines(frame, [c["projected_polygon"].astype(dtype='int32')+[center_x, 0]], False, (255, 255, 255))

        frame = self.draw_objects(frame, objects)
        self.glviewer.process_elements(objects)
        if len(masks) > 0:
            self.glviewer.update_floor_plane(masks[0])

        # Convert the OpenCV image to a QImage. WATCH that the ratio is lost
        frame_r = cv2.resize(frame, (self.ui.frame_2d.width(), self.ui.frame_2d.height()), cv2.INTER_LINEAR)
        height, width, channel = frame_r.shape
        q_image = QImage(frame_r.data, width, height, channel*width, QImage.Format_BGR888)
        # Display the QImage in the QLabel
        self.ui.frame_2d.setPixmap(QPixmap.fromImage(q_image))

    def draw_objects(self, image, objects):
        if len(objects) == +0:
            return image
        for element in objects:
            x0 = int(element.left)
            y0 = int(element.top)
            x1 = int(element.right)
            y1 = int(element.bot)
            color = (0, 255, 0)
            line = 2
            if self.selected_object and element.id == self.selected_object.id:
                color = (0, 0, 255)
                line = 4
            if x1 - x0 < 0:
                cv2.rectangle(image, (x0, y0), (self.rgb.width, y1), color, line)
                cv2.rectangle(image, (0, y0), (x1, y1), color, line)
            else:
                cv2.rectangle(image, (x0, y0), (x1, y1), color, line)
            text = 'Cls: {} - Score: {:.1f}% - ID: {} - Depth: {:.1f}mm'.format(element.type, element.score * 100, element.id, element.depth)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(
                image,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                (255, 0, 0),
                -1
            )
            cv2.putText(image, text, (x0, y0 + txt_size[1]), font, 0.4, (0, 255, 0), thickness=1)
        return image

    def draw_semantic_segmentation(self, winname, color_image, seg, yolo_rois, seg_rois):
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        palette = np.array(self.mask2former.color_palette)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # Convert to BGR
        color_seg = color_seg[..., ::-1]

        # Show image + mask
        img = np.array(color_image) * 0.6 + color_seg * 0.4
        img = img.astype(np.uint8)
        txt_color = (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        if yolo_rois is not None:
            for r in yolo_rois:
                text = 'YOLO {}-{}'.format(r.type, r.id)
                txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                rect = [r.left, r.top, r.right-r.left, r.bot-r.top]
                cv2.rectangle(img, rect, (255, 0, 0), 2)
                cv2.putText(img, text, (r.left, r.top + txt_size[1]), font, 0.4, txt_color, thickness=1)
        if seg_rois is not None:
            for r in seg_rois:
                text = '{}-{}'.format(r.type, r.id)
                txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                cv2.rectangle(img, (r.left, r.top, r.right, r.bot), (0, 0, 255), 2)
                cv2.putText(img, text, (r.left, r.top + txt_size[1]), font, 0.4, txt_color, thickness=1)
        return img

    def draw_yolo_boxes(self, img, yolo_objects, class_names):
        for box in yolo_objects:
            if box.score < 0.6:
                continue
            x0 = box.left
            y0 = box.top
            x1 = box.right
            y1 = box.bot
            # color = (_COLORS[box.type] * 255).astype(np.uint8).tolist()
            color = (255, 0, 0)
            text = '{} - {:.1f}%'.format(class_names[box.type], box.score * 100)
            txt_color = (255, 0, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    def make_matrix_rt(self, roll, pitch, heading, x0, y0, z0):
        a = roll
        b = pitch
        g = heading
        mat = np.array([[np.cos(b) * np.cos(g),
                         (np.sin(a) * np.sin(b) * np.cos(g) + np.cos(a) * np.sin(g)), (np.sin(a) * np.sin(g) -
                                                                                       np.cos(a) * np.sin(b) * np.cos(
                        g)), x0],
                        [-np.cos(b) * np.sin(g), (np.cos(a) * np.cos(g) - np.sin(a) * np.sin(b) * np.sin(g)),
                         (np.sin(a) * np.cos(g) + np.cos(a) * np.sin(b) * np.sin(g)), y0],
                        [np.sin(b), -np.sin(a) * np.cos(b), np.cos(a) * np.cos(b), z0],
                        [0, 0, 0, 1]])
        return mat

    def stop_robot(self):
        try:
            self.omnirobot_proxy.setSpeedBase(0, 0, 0)
        except Ice.Exception as e:
            traceback.print_exc()
            print(e, "Error connecting to omnirobot")

    def insert_element(self):
        self.modelEntity = Qt3DCore.QEntity(self.glviewer.grid_entity)
        self.modelMesh = Qt3DRender.QMesh()
        self.modelMesh.setSource(QUrl.fromLocalFile('low_poly_person_by_Rochus.stl'))

        self.modelTransform = Qt3DCore.QTransform()
        self.modelTransform.setScale(0.25)
        self.modelTransform.setRotationX(-90)
        self.modelTransform.setTranslation(QVector3D(QPointF(3,3)))

        self.modelEntity.addComponent(self.modelMesh)
        self.modelEntity.addComponent(self.modelTransform)
        self.modelEntity.addComponent(self.material)

    ########################################################################
    def startup_check(self):
        print(f"Testing RoboCompCameraRGBDSimple.Point3D from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.Point3D()
        print(f"Testing RoboCompCameraRGBDSimple.TPoints from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TPoints()
        print(f"Testing RoboCompCameraRGBDSimple.TImage from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TImage()
        print(f"Testing RoboCompCameraRGBDSimple.TDepth from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TDepth()
        print(f"Testing RoboCompCameraRGBDSimple.TRGBD from ifaces.RoboCompCameraRGBDSimple")
        test = ifaces.RoboCompCameraRGBDSimple.TRGBD()
        print(f"Testing RoboCompMPC.Point from ifaces.RoboCompMPC")
        test = ifaces.RoboCompMPC.Point()
        print(f"Testing RoboCompMPC.Control from ifaces.RoboCompMPC")
        test = ifaces.RoboCompMPC.Control()
        print(f"Testing RoboCompOmniRobot.TMechParams from ifaces.RoboCompOmniRobot")
        test = ifaces.RoboCompOmniRobot.TMechParams()
        print(f"Testing RoboCompSemanticSegmentation.TBox from ifaces.RoboCompSemanticSegmentation")
        test = ifaces.RoboCompSemanticSegmentation.TBox()
        print(f"Testing RoboCompVisualElements.TObject from ifaces.RoboCompVisualElements")
        test = ifaces.RoboCompVisualElements.TObject()
        QTimer.singleShot(200, QApplication.instance().quit)


# MPC
        # try:
        #     path = [ifaces.RoboCompMPC.Point(x=x, y=y) for x, y in current_target[1:11]]
        #     control = self.mpc_proxy.newPath(path)
        #     print("Control", control)
        # except Ice.Exception as e:
        #     traceback.print_exc()
        #     print(e, "Error connecting to MPC")
        #     return

        # # omnirobot
        # if control.valid:
        #     try:
        #         self.omnirobot_proxy.setSpeedBase(control.side, control.adv, control.rot)
        #     except Ice.Exception as e:
        #         traceback.print_exc()
        #         print(e, "Error connecting to omnirobot")

    # def ground_objects(self, objects, seg_objects):
    #     # update, add, delete cycle against objects. We assume seg_objects are already being tracked by ByteTracker
    #     # so their ids are consistent
    #     # update
    #     local_objs = []
    #     for obj in objects:
    #         for s_obj in seg_objects:
    #             if obj.id == s_obj.id:
    #                 local_objs.append(s_obj)
    #     # remove matched objects from seg_objects
    #     for obj in local_objs:
    #         seg_objects.remove(obj)
    #     # add

    ######################
    # From the RoboCompCamera360RGB you can call this methods:
    # self.camera360rgb_proxy.getROI(...)

    ######################
    # From the RoboCompMPC you can call this methods:
    # self.mpc_proxy.newPath(...)

    ######################
    # From the RoboCompMPC you can use this types:
    # RoboCompMPC.Point
    # RoboCompMPC.Control

    ######################
    # From the RoboCompMaskElements you can call this methods:
    # self.maskelements_proxy.getMasks(...)
    # self.maskelements_proxy.getNamesofCategories(...)

    ######################
    # From the RoboCompMaskElements you can use this types:
    # RoboCompMaskElements.TRoi
    # RoboCompMaskElements.TMask

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
    # From the RoboCompVisualElements you can call this methods:
    # self.visualelements_proxy.getVisualObjects(...)

    ######################
    # From the RoboCompVisualElements you can use this types:
    # RoboCompVisualElements.TRoi
    # RoboCompVisualElements.TObject

    ######################
    # From the RoboCompVisualElements you can call this methods:
    # self.visualelements1_proxy.getVisualObjects(...)

    ######################
    # From the RoboCompVisualElements you can use this types:
    # RoboCompVisualElements.TRoi
    # RoboCompVisualElements.TObject

    ######################
    # From the RoboCompSegmentatorTrackingPub you can publish calling this methods:
    # self.segmentatortrackingpub_proxy.setTrack(...)

    #def thread_frame_capture(self, queue, segmentator):
    #     while True:
    #         try:
    #             #image = self.camerargbdsimple_proxy.getImage("/Shadow/camera_top")
    #             rgbd = self.camerargbdsimple_proxy.getAll("/Shadow/camera_top")  # TODO: cambiar a variable
    #             depth_frame = np.frombuffer(rgbd.depth.depth, dtype=np.float32).reshape((rgbd.depth.height, rgbd.depth.width, 1))
    #             frame = np.frombuffer(rgbd.image.image, dtype=np.uint8).reshape((rgbd.image.height, rgbd.image.width, 3))
    #             #frame = cv2.resize(frame, (384, 384))
    #             seg_objects = self.semanticsegmentation_proxy.getInstances()
    #             segmented_img = self.semanticsegmentation_proxy.getSegmentedImage()
    #             segmented_img = np.frombuffer(segmented_img.image, dtype=np.uint8).reshape(segmented_img.height, segmented_img.width)
    #             mask_img = self.semanticsegmentation_proxy.getMaskedImage("floor")
    #             mask_img = np.frombuffer(mask_img.image, dtype=np.uint8).reshape(mask_img.height, mask_img.width)
    #             #mask_img, segmented_img, seg_objects = segmentator.process(frame, depth_frame, rgbd.depth.focalx, rgbd.depth.focaly)
    #             #pythonyolo_objects = self.read_yolo_objects()
    #             # if yolo_objects:
    #             #     self.draw_yolo_boxes(frame, yolo_objects, self.yolo_object_names)
    #             queue.put([frame, mask_img, segmented_img, [], seg_objects])
    #             time.sleep(0.050)
    #         except Ice.Exception as e:
    #             traceback.print_exc()
    #             print(e)


# if self.selected_index is not None:
#     target = np.array(targets[self.selected_index])
#     points = self.dwa_optimizer.project_polygons([target])[0]
#     for p in points:
#         cv2.circle(frame, np.array(p).astype(int), 5, (0, 0, 255))

# compute rois for doors
# inst = instance_img['segments_info']
# door_ids = [v['id'] for v in inst if v['label_id'] == labels["door"] and v['score'] > 0.7]
# inst_img = instance_img['segmentation']
# for door_id in door_ids:
#     mask = np.zeros((inst_img.shape[0], inst_img.shape[1], 1), dtype=np.uint8)
#     mask[inst_img == door_id] = 255
#     mask_23 = cv2.boundingRect(mask)
#     cv2.rectangle(frame_new, mask_23, (0, 0, 255), 2)

# cv2.imshow(winname, frame)
# cv2.waitKey(2)

# def draw_frame(self, winname, frame, alternatives, segmented_img, instance_img, labels, curvatures, targets):
#     alpha = 0.8
#
#     for s, alt in enumerate(alternatives):
#         alt_lane = cv2.cvtColor(alt, cv2.COLOR_GRAY2BGR)
#         if s == self.selected_index:
#             color = (0, 0, 255)
#         else:
#             color = (100, 100, 100)
#         alt_lane[np.all(alt_lane == (255, 255, 255), axis=-1)] = color
#         frame = cv2.addWeighted(frame, alpha, alt_lane, 1 - alpha, 0)
#
#     if self.selected_index is not None:
#         target = np.array(targets[self.selected_index])
#         points = self.dwa_optimizer.project_polygons([target])[0]
#         for p in points:
#             cv2.circle(frame, np.array(p).astype(int), 5, (0, 0, 255))
#
#     # compute rois for doors
#     # inst = instance_img['segments_info']
#     # door_ids = [v['id'] for v in inst if v['label_id'] == labels["door"] and v['score'] > 0.7]
#     # inst_img = instance_img['segmentation']
#     # for door_id in door_ids:
#     #     mask = np.zeros((inst_img.shape[0], inst_img.shape[1], 1), dtype=np.uint8)
#     #     mask[inst_img == door_id] = 255
#     #     mask_23 = cv2.boundingRect(mask)
#     #     cv2.rectangle(frame_new, mask_23, (0, 0, 255), 2)
#
#     cv2.imshow(winname, frame)
#     cv2.waitKey(2)


  # take control actions
        # active, control, selected = self.check_human_interface(candidates)
        # if not active:
        #     active, control, selected = self.check_visual_target(candidates)
        #     if not active:
        #         self.draw_frame(self.winname, frame, candidates)
        #         cv2.imshow(self.winname, frame)
        #         cv2.waitKey(2)
        #         return
        #     else:
        #         print("Compute: num candidates:", len(candidates), "selected:", selected is not None, self.yolo_object_names[self.selected_object.type])

        #self.control_2(control)

        # def control(self, curvatures, targets, controls):
        #     # we need to assign curvatures to buttons
        #     self.selected_index = None
        #     digitized = np.digitize(curvatures, self.bins) // 4  # (10+10)/5
        #     for i, b in enumerate(self.buttons):
        #         b.setEnabled(i in digitized)
        #         if b.isDown():
        #             self.selected_index = int(np.where(digitized == i)[0][0])
        #     if self.selected_index is not None:
        #         current_target = targets[self.selected_index]
        #         control = controls[self.selected_index]
        #         print(control, curvatures[self.selected_index], "dist:",
        #               np.linalg.norm(targets[self.selected_index][-1]))
        #     elif self.selected_object is not None:
        #         # select closest direction to target
        #         center_object = np.array([self.selected_object.x, self.selected_object.y])
        #         object_index = np.argmin(np.linalg.norm(targets[-1] - center_object))
        #         control = controls[object_index]
        #     else:
        #         self.stop_robot()
        #         return
        #
        #     # omnirobot
        #     adv, rot = control  # TODO: move limit to sampler
        #     if adv > 1000:
        #         adv = 1000
        #     side = 0
        #
        #     try:
        #         print(side, adv, rot * 2)
        #         self.omnirobot_proxy.setSpeedBase(side, adv, rot * 2)
        #     except Ice.Exception as e:
        #         traceback.print_exc()
        #         print(e, "Error connecting to omnirobot")
