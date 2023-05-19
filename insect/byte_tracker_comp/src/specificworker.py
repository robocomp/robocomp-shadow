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
import traceback

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *

import interfaces as ifaces
import numpy as np
import cv2

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)
sys.path.append('/home/robolab/robocomp/components/robocomp-shadow/insect/byte_tracker_comp/ByteTrack')
# from yolox.tracker.byte_tracker_depth import BYTETracker as BYTETrackerDepth
from yolox.tracker.byte_tracker import BYTETracker

# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 1
        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)
        from queue import Queue

        self.process_queue = Queue(maxsize = 1)

        self.image_height = 640
        self.image_width = 640
        self.publisher = 0
        from queue import Queue

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        try:
            if params["depth_flag"] == "true" or params["depth_flag"] == "True":
                self.tracker = BYTETrackerDepth(frame_rate=30)
            else:
                self.tracker = BYTETracker(frame_rate=30)
        except:
            traceback.print_exc()
            print("Error reading config params")

        return True


    @QtCore.Slot()
    def compute(self):
        # Get elements data from queue
        data = self.process_queue.get()
        # Get tracks from Bytetrack and convert data to VisualElements interface
        processed_data = self.to_visualelements_interface(self.tracker.update_original(np.array(data["scores"]), np.array(data["boxes"]), np.array(data["clases"])))

        img = self.read_image()
        img = self.display_data(img, processed_data)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        try:
            self.visualelements_proxy.setVisualObjects(processed_data, self.publisher)
        except:
            print("Error communicating with BYTETRACK")
            traceback.print_exc()
            return

    def read_image(self):
        rgb = self.camera360rgb_proxy.getROI(920, 460, 920, 920, 640, 640)
        rgb_frame = np.frombuffer(rgb.image, dtype=np.uint8).reshape((rgb.height, rgb.width, 3))
        img = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        return img

    def display_data(self, img, data):
        image_shape = img.shape
        for element in data:
            x0 = int(element.left*(image_shape[0]/self.image_width))
            y0 = int(element.top*(image_shape[1]/self.image_height))
            x1 = int(element.right*(image_shape[0]/self.image_width))
            y1 = int(element.bot*(image_shape[1]/self.image_height))
            cv2.rectangle(img, (x0, y0), (x1, y1), (0,255,0), 2)
            text = 'Class: {} - Score: {:.1f}% - ID: {}'.format(element.type, element.score*100, element.id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                (255, 0, 0),
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, (0,255,0), thickness=1)
        return img

    def to_visualelements_interface(self, tracks):
        ret = ifaces.RoboCompVisualElements.TObjects()
        for track in tracks:
            target = ifaces.RoboCompVisualElements.TObject()
            target.id = track.track_id
            target.score = track.score
            target.left = int(track.tlwh[0])
            target.top = int(track.tlwh[1])
            target.right = int(track.tlwh[0]+track.tlwh[2])
            target.bot = int(track.tlwh[1]+track.tlwh[3])
            target.type = track.clase
            ret.append(target)
        return ret

    def startup_check(self):
        print(f"Testing RoboCompYoloObjects.TBox from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TBox()
        print(f"Testing RoboCompYoloObjects.TKeyPoint from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TKeyPoint()
        print(f"Testing RoboCompYoloObjects.TPerson from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TPerson()
        print(f"Testing RoboCompYoloObjects.TConnection from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TConnection()
        print(f"Testing RoboCompYoloObjects.TJointData from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TJointData()
        print(f"Testing RoboCompYoloObjects.TData from ifaces.RoboCompYoloObjects")
        test = ifaces.RoboCompYoloObjects.TData()
        QTimer.singleShot(200, QApplication.instance().quit)

    def VisualElements_setVisualObjects(self, visualObjects, publisher):
        target_data_dict = {"scores" : [], "boxes" : [], "clases" : []}
        for object in visualObjects:
            target_data_dict["scores"].append(object.score)
            target_data_dict["boxes"].append([object.left, object.top, object.right, object.bot])
            target_data_dict["clases"].append(object.type)
        self.process_queue.put(target_data_dict)
        self.publisher = publisher
        if publisher == 0:
            self.image_height = 640
            self.image_width = 640
        elif publisher == 1:
            self.image_height = 384
            self.image_width = 384
    #
    # IMPLEMENTATION of getTargets method from ByteTrack interface
    #
    # def ByteTrack_getTargets(self, ps, pb, clases):
    #     ret = ifaces.RoboCompByteTrack.OnlineTargets()
    #     scores = np.array(ps)
    #     boxes = np.array(pb)
    #     clases = np.array(clases)
    #     for i in self.tracker.update_original(scores, boxes, clases):
    #         target = ifaces.RoboCompByteTrack.Targets()
    #         tlwh = ifaces.RoboCompByteTrack.Box(i.tlwh)
    #         target.trackid = i.track_id
    #         target.score = i.score
    #         target.tlwh = tlwh
    #         target.clase = i.clase
    #         ret.append(target)
    #     return ret
    # #
    # # IMPLEMENTATION of getTargetswithdepth method from ByteTrack interface
    # #
    # def ByteTrack_getTargetswithdepth(self, ps, pb, depth, clases):
    #     ret = ifaces.RoboCompByteTrack.OnlineTargets()
    #     depth = np.frombuffer(depth.depth, dtype=np.float32).reshape(depth.height, depth.width, 1)
    #     scores = np.array(ps)
    #     boxes = np.array(pb)
    #     clases = np.array(clases)
    #     for i in self.tracker.update2(scores, boxes, depth, clases):
    #         target = ifaces.RoboCompByteTrack.Targets()
    #         tlwh = ifaces.RoboCompByteTrack.Box(i.tlwh)
    #         target.trackid = i.track_id
    #         target.score = i.score
    #         target.tlwh = tlwh
    #         target.clase = i.clase
    #         ret.append(target)
    #     return ret
    # # ===================================================================
    # # ===================================================================
    #
    # def ByteTrack_setTargets(self, ps, pb, clases, sender):
    #
    #
    # def ByteTrack_allTargets(self):
    #     return self.read_tracks

    ######################
    # From the RoboCompByteTrack you can use this types:
    # RoboCompByteTrack.Targets


