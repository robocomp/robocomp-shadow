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
from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces
import numpy as np
import traceback
import cv2
import time
import itertools
import queue
sys.path.append('/home/robocomp/robocomp/lib')
console = Console(highlight=False)
from dataclasses import dataclass


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
            
            # visual objects
            self.visual_objects_queue = queue.Queue(1)
            self.total_objects = []

           
            # bins
            plusbins = np.geomspace(0.001, 1, 10)
            minusbins = -np.geomspace(0.0001, 1, 10)
            self.bins = np.append(minusbins[::-1], plusbins)  # compose both gemspaces from -1 to 1 with high density close to 0

            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)
            self.image_width = 1920
            
            
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
        # Get masks from mask2former. Moved to DWA components
        #candidates = self.compute_candidates(["floor"])

        # Get objects from yolo and mask2former, and transform them
        if self.yolo:
            yolo_objects = self.read_yolo_objects()
        else:
            yolo_objects = []
        if self.semantic:
            sm_objects = self.read_ss_objects()
        else:
            sm_objects = []
        self.total_objects = self.transform_and_concatenate_objects(yolo_objects, sm_objects)

        # publish target
        if self.selected_object is not None:
            self.publish_target(self.selected_object)
        self.show_fps()

    #########################################################################

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
            element.left = int(element.left * x_factor + x_roi_offset) % self.image_width
            element.right = int(element.right * x_factor + x_roi_offset) % self.image_width
            element.top = int(element.top*y_factor + y_roi_offset)
            element.bot = int(element.bot*y_factor + y_roi_offset)
        return total_objects


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


    ########################################################################
    def startup_check(self):
        print(f"Testing RoboCompSemanticSegmentation.TBox from ifaces.RoboCompSemanticSegmentation")
        test = ifaces.RoboCompSemanticSegmentation.TBox()
        print(f"Testing RoboCompVisualElements.TObject from ifaces.RoboCompVisualElements")
        test = ifaces.RoboCompVisualElements.TObject()
        QTimer.singleShot(200, QApplication.instance().quit)

    
    # =============== Methods for Component SubscribesTo ================
    # ===================================================================

    #
    # SUBSCRIPTION to setTrack method from SegmentatorTrackingPub interface
    #
    def SegmentatorTrackingPub_setTrack(self, target):
    
        if target is None:
            print("None target received")
            return
        if target.id == -1:
            print("Empty target received")
            self.selected_object = target
            return
        if len(self.total_objects) == 0:
            print("No visible objects. Cannot assign the target", target.id)
            return
        found, self.selected_object = self.check_selected_in_new_objects(self.total_objects, target)
        if not found:
            print("Warning: target LOST")
            return
        print("target assigned", target.id)    
        return
        
