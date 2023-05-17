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
from yolox.tracker.byte_tracker_depth import BYTETracker as BYTETrackerDepth
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
        self.yolo_queue = Queue(maxsize = 1)
        self.ss_queue = Queue(maxsize = 1)
        self.yolo_tracks = []
        self.ss_tracks = []
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
        # Track YOLO data
        yolo_rois = self.yolo_queue.get()
        self.yolo_tracks = self.to_bytetrack_interface(self.tracker.update_original(yolo_rois["scores"], yolo_rois["boxes"], yolo_rois["clases"]))

        # Track SS data
        ss_rois = self.ss_queue.get()
        self.ss_tracks = self.to_bytetrack_interface(self.tracker.update_original(ss_rois["scores"], ss_rois["boxes"], ss_rois["clases"]))

    def to_bytetrack_interface(self, tracks):
        ret = ifaces.RoboCompByteTrack.TOnlineTargets()
        for track in tracks:
            target = ifaces.RoboCompByteTrack.Targets()
            target.trackid = track.track_id
            target.score = track.score
            target.tlwh = ifaces.RoboCompByteTrack.Box(track.tlwh)
            target.clase = track.clase
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




    #
    # IMPLEMENTATION of getTargets method from ByteTrack interface
    #
    def ByteTrack_getTargets(self, ps, pb, clases):
        ret = ifaces.RoboCompByteTrack.OnlineTargets()
        scores = np.array(ps)
        boxes = np.array(pb)
        clases = np.array(clases)
        for i in self.tracker.update_original(scores, boxes, clases):
            target = ifaces.RoboCompByteTrack.Targets()
            tlwh = ifaces.RoboCompByteTrack.Box(i.tlwh)
            target.trackid = i.track_id
            target.score = i.score
            target.tlwh = tlwh
            target.clase = i.clase
            ret.append(target)
        return ret
    #
    # IMPLEMENTATION of getTargetswithdepth method from ByteTrack interface
    #
    def ByteTrack_getTargetswithdepth(self, ps, pb, depth, clases):
        ret = ifaces.RoboCompByteTrack.OnlineTargets()
        depth = np.frombuffer(depth.depth, dtype=np.float32).reshape(depth.height, depth.width, 1)
        scores = np.array(ps)
        boxes = np.array(pb)
        clases = np.array(clases)
        for i in self.tracker.update2(scores, boxes, depth, clases):
            target = ifaces.RoboCompByteTrack.Targets()
            tlwh = ifaces.RoboCompByteTrack.Box(i.tlwh)
            target.trackid = i.track_id
            target.score = i.score
            target.tlwh = tlwh
            target.clase = i.clase
            ret.append(target)
        return ret
    # ===================================================================
    # ===================================================================

    def ByteTrack_setTargets(self, ps, pb, clases, sender):
        target_data_dict = {"scores" : np.array(ps)}, {"boxes" : np.array(pb)}, {"clases" : np.array(clases)}
        if sender == "yolo":
            self.yolo_queue.put(target_data_dict)
        if sender == "semantic_segmentator":
            self.ss_queue.put(target_data_dict)

    def ByteTrack_allTargets(self):
        return self.yolo_tracks.extend(self.ss_tracks)

    ######################
    # From the RoboCompByteTrack you can use this types:
    # RoboCompByteTrack.Targets


