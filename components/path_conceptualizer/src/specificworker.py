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

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import numpy as np
import traceback
import cv2
import time
import threading
from enum import Enum
from queue import Queue
from dwa_optimizer import DWA_Optimizer
from mask2former import Mask2Former
from floodfill_segmentator import Floodfill_Segmentator
sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 50
        if startup_check:
            self.startup_check()
        else:
            # camera matrix
            try:
                image = self.camerargbdsimple_proxy.getImage("/Shadow/camera_top")
                print("Camera specs:")
                print(" width:", image.width)
                print(" height:", image.height)
                print(" depth", image.depth)
                print(" focalx", image.focalx)
                print(" focaly", image.focaly)
                print(" period", image.period)
            except Ice.Exception as e:
                traceback.print_exc()
                print(e, "Cannot connect to camera. Aborting")

            rx = np.deg2rad(20)  # get from camera_proxy
            cam_to_robot = self.make_matrix_rt(rx, 0, 0, 0, -15, 1580)  # converts points in camera CS to robot CS
            robot_to_cam = np.linalg.inv(cam_to_robot)

            # optimizer
            self.dwa_optimizer = DWA_Optimizer(robot_to_cam, image.focalx, image.focaly, (image.height, image.width, image.depth))
            # while cv2.waitKey(25) & 0xFF != ord('q'):
            #      pass

            self.mask2former = Mask2Former()
            self.floodfill = Floodfill_Segmentator()

            # start frame thread
            self.buffer = []
            self.ready = False
            self.frame_queue = Queue(1)     # Hay que poner 1
            self.thread_frame = threading.Thread(target=self.thread_frame_capture, args=(self.frame_queue, self.floodfill), daemon=True)
            self.thread_frame.start()
            print("Video frame started")

            self.winname = "Path Concept"
            cv2.namedWindow(self.winname)
            cv2.setMouseCallback(self.winname, self.mouse_click)

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

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)
    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        #	self.innermodel = InnerModel(params["InnerModelPath"])
        return True

    @QtCore.Slot()
    def compute(self):
        frame, mask_img, segmented_img, instance_img = self.frame_queue.get()
        self.segmented_img = segmented_img
        #self.draw_semantic_segmentation(self.winname, segmented_img, frame)
        alternatives, curvatures, targets = self.dwa_optimizer.optimize(loss=self.target_function_mask, mask_img=mask_img)
        self.draw_frame(self.winname, frame, alternatives, segmented_img, instance_img, self.mask2former.labels,
                        curvatures, targets)
        self.control(curvatures, targets)

        return True

#########################################################################
    def control(self, curvatures, targets):
        # we need to assign curvatures to buttons
        self.selected_index = None
        bins = np.arange(-1, 1, 2/5)
        digitized = np.digitize(curvatures, bins)
        print(bins, digitized, curvatures)
        for i, b in enumerate(self.buttons):
            if i not in digitized:
                b.setEnabled(False)
            else:
                b.setEnabled(True)
                if b.isDown():
                    self.selected_index = int(curvatures[digitized == i][0])

        # if self.ui.pushButton_left.isDown():
        #     self.selected_index = np.argmax(curvatures)
        # elif self.ui.pushButton_leftleft.isDown():
        #         self.selected_index = np.argmax(curvatures)
        # elif self.ui.pushButton_centre.isDown():
        #     self.selected_index = np.argmin(np.abs(curvatures))  # closer to 0
        # elif self.ui.pushButton_right.isDown():
        #     self.selected_index = np.argmin(curvatures)
        # else:
        #     self.selected_index = None

        if self.selected_index is not None:
            current_target = targets[self.selected_index]
            target = current_target[len(current_target)//2]
            rot = np.arctan2(target[0], target[1])
            adv = np.linalg.norm(target)
            try:
                #self.omnirobot_proxy.setSpeedBase(0, adv, rot)
                print("Control", adv, rot)
            except Ice.Exception as e:
                traceback.print_exc()
                print(e)

    def target_function_mask(self, mask_img, mask_poly):
        result = cv2.bitwise_and(mask_img, mask_poly)
        lane_size = np.count_nonzero(mask_poly)
        segmented_size = np.count_nonzero(mask_img)
        inliers = np.count_nonzero(result)
        loss = abs(segmented_size - lane_size) + 5 * abs(lane_size - inliers)  # + 300*abs(curvature)
        return float(loss)

    def thread_frame_capture(self, queue, segmentator):
        while True:
            try:
                image = self.camerargbdsimple_proxy.getImage("/Shadow/camera_top")
                frame = np.frombuffer(image.image, dtype=np.uint8).reshape((image.height, image.width, 3))
                frame = cv2.resize(frame, (384, 384))
                mask_img, segmented_img, instance_img = segmentator.process(frame)
                #mask_img, segmented_img, instance_img = self.mask2former.process(frame)
                queue.put([frame, mask_img, segmented_img, instance_img])
                time.sleep(0.050)
            except Ice.Exception as e:
                traceback.print_exc()
                print(e)

    def draw_frame(self, winname, frame, alternatives, segmented_img, instance_img, labels, curvatures, targets):
        alpha = 0.8

        for s, alt in enumerate(alternatives):
            alt_lane = cv2.cvtColor(alt, cv2.COLOR_GRAY2BGR)
            if s == self.selected_index:
                color = (0, 0, 255)
            else:
                color = (100, 100, 100)
            alt_lane[np.all(alt_lane == (255, 255, 255), axis=-1)] = color
            frame = cv2.addWeighted(frame, alpha, alt_lane, 1 - alpha, 0)

        if self.selected_index is not None:
            target = np.array(targets[self.selected_index])
            points = self.dwa_optimizer.project_polygons([target])[0]
            for p in points:
                cv2.circle(frame, np.array(p).astype(int), 5, (0, 0, 255))

        # compute rois for doors
        # inst = instance_img['segments_info']
        # door_ids = [v['id'] for v in inst if v['label_id'] == labels["door"] and v['score'] > 0.7]
        # inst_img = instance_img['segmentation']
        # for door_id in door_ids:
        #     mask = np.zeros((inst_img.shape[0], inst_img.shape[1], 1), dtype=np.uint8)
        #     mask[inst_img == door_id] = 255
        #     mask_23 = cv2.boundingRect(mask)
        #     cv2.rectangle(frame_new, mask_23, (0, 0, 255), 2)

        cv2.imshow(winname, frame)
        cv2.waitKey(2)

    def draw_semantic_segmentation(self, winname, seg, color_image):
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        palette = np.array(self.mask2former.color_palette)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # Convert to BGR
        color_seg = color_seg[..., ::-1]

        # Show image + mask
        img = np.array(color_image) * 0.5 + color_seg * 0.5
        img = img.astype(np.uint8)
        cv2.imshow(winname, np.asarray(img))
        cv2.waitKey(2)

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

    #########################################################################################3
    def mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_point = (x, y)
            print("Clicked:", self.selected_point)
            print(list(self.mask2former.labels.keys())[list(self.mask2former.labels.values()).index(self.segmented_img[y, x].item())])

    def slot_button_left(self):
        #self.human_choice = self.Human_Choices.left
        pass

    def slot_button_right(self):
        #self.human_choice = self.Human_Choices.right
        pass

    def slot_button_centre(self):
        #self.human_choice = self.Human_Choices.centre
        pass

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
        QTimer.singleShot(200, QApplication.instance().quit)


    ######################
    # From the RoboCompCameraRGBDSimple you can call this methods:
    # self.camerargbdsimple_proxy.getAll(...)
    # self.camerargbdsimple_proxy.getDepth(...)
    # self.camerargbdsimple_proxy.getImage(...)
    # self.camerargbdsimple_proxy.getPoints(...)

    ######################
    # From the RoboCompCameraRGBDSimple you can use this types:
    # RoboCompCameraRGBDSimple.Point3D
    # RoboCompCameraRGBDSimple.TPoints
    # RoboCompCameraRGBDSimple.TImage
    # RoboCompCameraRGBDSimple.TDepth
    # RoboCompCameraRGBDSimple.TRGBD


