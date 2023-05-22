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
import time

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)
sys.path.append('/home/robocomp/robocomp/components/robocomp-shadow/insect/byte_tracker/ByteTrack')
# from yolox.tracker.byte_tracker_depth import BYTETracker as BYTETrackerDepth
from yolox.tracker.byte_tracker import BYTETracker

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 100

        # ROI parameters. Must be filled up
        self.final_xsize = 0
        self.final_ysize = 0
        self.roi_xcenter = 0
        self.roi_ycenter = 0
        self.roi_xsize = 0
        self.roi_ysize = 0

        if startup_check:
            self.startup_check()
        else:
            self.objects_read = []
            self.objects_write = []
            self.display = False

            # Hz
            self.cont = 1
            self.last_time = time.time()
            self.fps = 0

            # read test image to get sizes
            try:
                rgb = self.camera360rgb_proxy.getROI(-1, -1, -1, -1, -1, -1)
                self.center_x = rgb.width // 2
                self.center_y = rgb.height // 2
                print("Camera specs:")
                print(" width:", rgb.width)
                print(" height:", rgb.height)
                print(" depth", rgb.depth)
                print(" focalx", rgb.focalx)
                print(" focaly", rgb.focaly)
                print(" period", rgb.period)
                print(" ratio {:.2f}.format(image.width/image.height)")
            except Ice.Exception as e:
                traceback.print_exc()
                print(e, "Aborting...")
                sys.exit()

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):
        try:
            if params["depth_flag"] == "true" or params["depth_flag"] == "True":
                self.tracker = BYTETrackerDepth(frame_rate=30)
            else:
                self.tracker = BYTETracker(frame_rate=30)
            if params["display"] == "true" or params["display"] == "True":
                self.display = True
        except:
            traceback.print_exc()
            print("Error reading config params")

        return True


    @QtCore.Slot()
    def compute(self):
        # Read visual elements from segmentator
        data = self.read_visual_objects()

        # Get tracks from Bytetrack and convert data to VisualElements interface
        processed_data = self.to_visualelements_interface(self.tracker.update_original(np.array(data["scores"]),
                                                          np.array(data["boxes"]), np.array(data["clases"])))

        # read image
        if self.display:
            img = self.read_image()
            img = self.display_data(img, processed_data)
            if img is not None:
                cv2.imshow("Image", img)
                cv2.waitKey(1)

        self.show_fps()

    #########################################################################################################
    def read_visual_objects(self):
        data = {"scores": [], "boxes": [], "clases": []}
        try:
            visual_objects = self.visualelements_proxy.getVisualObjects()
            for object in visual_objects:
                data["scores"].append(object.score)
                data["boxes"].append([object.left, object.top, object.right, object.bot])
                data["clases"].append(object.type)

            # get roi params from firs visual object since all are the same
            if visual_objects:
                roi = visual_objects[0].roi
                self.final_xsize = roi.finalxsize
                self.final_ysize = roi.finalysize
                self.roi_xcenter = roi.xcenter
                self.roi_ycenter = roi.ycenter
                self.roi_xsize = roi.xsize
                self.roi_ysize = roi.ysize

        except Ice.Exception as e:
            traceback.print_exc()
            print(e, "Error reading from Visual Objects interface")
        return data

    def to_visualelements_interface(self, tracks):
        self.objects_write = ifaces.RoboCompVisualElements.TObjects()
        for track in tracks:
            target = ifaces.RoboCompVisualElements.TObject()
            target.id = track.track_id
            target.score = track.score
            target.left = int(track.tlwh[0])
            target.top = int(track.tlwh[1])
            target.right = int(track.tlwh[0]+track.tlwh[2])
            target.bot = int(track.tlwh[1]+track.tlwh[3])
            target.type = track.clase
            self.objects_write.append(target)

        # swap
        self.objects_write, self.objects_read = self.objects_read, self.objects_write
        return self.objects_read

    def read_image(self):
        #rgb = self.camera360rgb_proxy.getROI(self.center_x, self.center_y, 512, 430, 640, 640)
        rgb = self.camera360rgb_proxy.getROI(-1, -1, -1, -1, -1, -1)
        rgb_frame = np.frombuffer(rgb.image, dtype=np.uint8).reshape((rgb.height, rgb.width, 3))
        return rgb_frame

    def display_data(self, image, objects):
        if len(objects) == +0:
            return
        xfactor = 1024/self.final_xsize
        yfactor = 512/self.final_ysize
        print(xfactor, yfactor)
        for element in objects:
            x0 = int(element.left*xfactor)
            y0 = int(element.top*yfactor)
            x1 = int(element.right*xfactor)
            y1 = int(element.bot*yfactor)
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
            text = 'Class: {} - Score: {:.1f}% - ID: {}'.format(element.type, element.score*100, element.id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(
                image,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                (255, 0, 0),
                -1
            )
            cv2.putText(image, text, (x0, y0 + txt_size[1]), font, 0.4, (0,255,0), thickness=1)
        return image

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
    ##############################################################################################

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

    ##############################################################################################
    # IMPLEMENTATION of getVisualObjects method from VisualElements interface
    ##############################################################################################

    def VisualElements_getVisualObjects(self):
        return self.objects_read


    ##############################################################################################

    # def VisualElements_setVisualObjects(self, visualObjects, publisher):
    #     target_data_dict = {"scores" : [], "boxes" : [], "clases" : []}
    #     for object in visualObjects:
    #         target_data_dict["scores"].append(object.score)
    #         target_data_dict["boxes"].append([object.left, object.top, object.right, object.bot])
    #         target_data_dict["clases"].append(object.type)
    #     self.process_queue.put(target_data_dict)
    #     self.publisher = publisher
    #     if publisher == 0:
    #         self.image_height = 640
    #         self.image_width = 640
    #     elif publisher == 1:
    #         self.image_height = 384
    #         self.image_width = 384
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


